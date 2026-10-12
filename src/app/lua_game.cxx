#include "app/lua_game.hxx"

#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

#include "app/lua_game_api.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    constexpr char const *default_entry = "assets/scripts/main.lua";

    auto read_data_file(std::string_view relative) -> std::optional<std::string> {
        auto const path = Paths::current().data(relative);

        if (!path) {
            return std::nullopt;
        }

        std::ifstream file{path->absolute(), std::ios::binary};

        if (!file) {
            return std::nullopt;
        }

        std::ostringstream contents;

        contents << file.rdbuf();

        return std::move(contents).str();
    }

    constexpr float default_fov_degrees = 60.0F;
    constexpr float default_near_clip = 0.1F;
    constexpr float default_far_clip = 1000.0F;
}

struct LuaGame::Impl {
    GameHost host;
    LuaGameHost bindings;
    std::vector<std::pair<std::string, lua_CFunction>> native_modules;
    std::unique_ptr<LuaRuntime> runtime;

    std::string fatal_error;

    Scene const *bound_scene = nullptr;

    mutable CameraParams last_camera{};
    mutable bool have_camera = false;

    auto call(char const *name, std::span<double const> numbers = {}) -> void {
        if (runtime) {
            static_cast<void>(runtime->call(name, numbers));
        }
    }
};

LuaGame::LuaGame() : impl_(std::make_unique<Impl>()) {}

LuaGame::~LuaGame() = default;

auto LuaGame::add_native_module(std::string name, lua_CFunction opener) -> void {
    impl_->native_modules.emplace_back(std::move(name), opener);
}

auto LuaGame::attach_host(GameHost host) -> void {
    impl_->host = std::move(host);
    impl_->bindings.host = impl_->host;
}

auto LuaGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    auto &self = *impl_;

    scene.get_registry().clear();

    self.bindings.scene = &scene;
    self.bindings.renderer = &renderer;
    self.bindings.engine_models = &engine_models;
    self.bindings.animation = std::make_unique<LuaAnimationSystem>();
    self.bindings.collider_slots.clear();
    self.bindings.mesh_colliders.clear();
    self.bound_scene = nullptr;
    self.have_camera = false;
    self.fatal_error.clear();

    self.runtime.reset();

    auto created = LuaRuntime::create();

    if (!created) {
        self.fatal_error = created.error();
        error("[lua] {}", self.fatal_error);

        return;
    }

    self.runtime = std::make_unique<LuaRuntime>(std::move(*created));
    self.runtime->set_host(&self.bindings);
    self.runtime->set_source_loader([](std::string const &module_path) -> std::optional<std::string> {
        return read_data_file("assets/scripts/" + module_path + ".lua");
    });

    for (auto const &[name, opener]: self.native_modules) {
        self.runtime->register_native_module(name, opener);
    }

    open_lua_game_api(*self.runtime);

    auto const entry = self.host.script_entry.empty() ? std::string{default_entry} : self.host.script_entry;
    auto const source = read_data_file(entry);

    if (!source) {
        self.fatal_error = "the entry script '" + entry + "' was not found in the data directory";
        error("[lua] {}", self.fatal_error);

        return;
    }

    info("[lua] Loading '{}'", entry);

    if (auto const loaded = self.runtime->run_main(entry, *source); !loaded) {
        self.fatal_error = loaded.error();
        error("[lua] {}", self.fatal_error);

        return;
    }

    self.call("on_populate");
}

auto LuaGame::on_update(Scene &scene, float delta_time) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    if (self.bound_scene != &scene) {
        self.bound_scene = &scene;
        self.call("on_bind");
    }

    update_lua_game_physics(self.bindings);

    std::array const arguments{static_cast<double>(delta_time)};

    self.call("on_update", arguments);

    if (self.bindings.renderer != nullptr) {
        self.bindings.animation->update(scene, *self.bindings.renderer, delta_time);
    }

    // A frame's mouse movement is seen by one update.
    self.bindings.mouse_delta_x = 0.0;
    self.bindings.mouse_delta_y = 0.0;
}

auto LuaGame::on_key_released(Scene & , KeyReleasedEvent const &event) -> void {
    impl_->bindings.keys_down.erase(event.key);
}

auto LuaGame::on_mouse_moved(Scene & , MouseMovedEvent const &event) -> void {
    impl_->bindings.mouse_delta_x += event.delta_x;
    impl_->bindings.mouse_delta_y += event.delta_y;
}

auto LuaGame::on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;
    self.bindings.keys_down.insert(event.key);

    std::array const arguments{static_cast<double>(event.key), static_cast<double>(event.modifiers)};

    self.call("on_key", arguments);
}

auto LuaGame::on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    std::array const arguments{static_cast<double>(event.button)};

    self.call("on_mouse_button", arguments);
}

auto LuaGame::wants_cursor() const -> bool { return impl_->runtime && impl_->runtime->callback_flag("wants_cursor"); }

auto LuaGame::on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    std::array const arguments{event.ndc_x, event.ndc_y, event.inside ? 1.0 : 0.0};

    self.call("on_cursor", arguments);
}

auto LuaGame::on_ui(Scene &scene, Renderer &renderer) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;
    self.bindings.renderer = &renderer;

    self.call("on_ui");

    auto const error_text = !self.fatal_error.empty() ? self.fatal_error
                            : self.runtime            ? self.runtime->last_error()
                                                      : std::string{};

    if (error_text.empty()) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2{16.0F, 16.0F}, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.9F);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0F, 0.45F, 0.4F, 1.0F});

    if (ImGui::Begin("Lua error##lua_error", nullptr,
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::TextWrapped("%s", error_text.c_str());
        ImGui::TextDisabled("Fix the script and press Ctrl+R to reload.");
    }

    ImGui::End();
    ImGui::PopStyleColor();
}

auto LuaGame::camera(Scene const & , float aspect_ratio) const -> CameraParams {
    auto &self = *impl_;

    glm::vec3 eye{0.0F, 6.0F, -8.0F};
    glm::vec3 target{0.0F};
    float fov = default_fov_degrees;
    float near_clip = default_near_clip;
    float far_clip = default_far_clip;

    if (self.runtime && self.runtime->has_callback("camera")) {
        std::array const arguments{static_cast<double>(aspect_ratio)};

        if (self.runtime->call("camera", arguments, 1).ok) {
            auto *const state = self.runtime->state();
            auto const table = lua_gettop(state);

            if (lua_istable(state, table)) {
                auto const eye_value = lua_field_vec3(state, table, "eye", {eye.x, eye.y, eye.z});
                auto const target_value = lua_field_vec3(state, table, "target", {target.x, target.y, target.z});

                eye = {eye_value.x, eye_value.y, eye_value.z};
                target = {target_value.x, target_value.y, target_value.z};
                fov = static_cast<float>(lua_field_number(state, table, "fov", fov));
                near_clip = static_cast<float>(lua_field_number(state, table, "near", near_clip));
                far_clip = static_cast<float>(lua_field_number(state, table, "far", far_clip));
            }

            self.runtime->pop(1);
        }
    }

    CameraParams params{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov), aspect_ratio, near_clip, far_clip),
            .near_clip = near_clip,
            .far_clip = far_clip,
            .vertical_fov_radians = glm::radians(fov),
    };

    self.last_camera = params;
    self.have_camera = true;
    self.bindings.inverse_view_projection = glm::inverse(params.projection * params.view);

    return params;
}
