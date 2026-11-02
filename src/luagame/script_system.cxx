#include "luagame/script_system.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

#include "assets/file_change_queue.hxx"
#include "assets/file_change_watcher.hxx"
#include "luagame/lua_game_api.hxx"
#include "luagame/lua_http.hxx"
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

    // A .luarc.json in the project root, so a Lua language server (lua_ls) opened on a script finds the engine's API
    // stubs (lua-api/) and the callbacks and libraries a script gets. It is the server's root marker; an existing file is
    // left alone.
    auto write_editor_config(std::filesystem::path const &project_root, std::filesystem::path const &engine_root) -> void {
        auto const config = project_root / ".luarc.json";
        auto const stubs = engine_root / "lua-api";
        std::error_code ignored;

        if (std::filesystem::exists(config, ignored) || !std::filesystem::is_directory(stubs, ignored)) {
            return;
        }

        std::ofstream file{config};

        if (!file) {
            return;
        }

        file << "{\n"
                "  \"$schema\": \"https://raw.githubusercontent.com/LuaLS/vscode-lua/master/setting/schema.json\",\n"
                "  \"runtime\": { \"version\": \"Lua 5.4\" },\n"
                "  \"workspace\": { \"library\": [\""
             << stubs.generic_string()
             << "\"], \"checkThirdParty\": false },\n"
                "  \"diagnostics\": {\n"
                "    \"globals\": [\"scene\", \"assets\", \"camera\", \"ui\", \"game\", \"entity\", \"compute\", \"key\", "
                "\"mouse\", \"physics\", \"animation\", \"render\", \"http\", \"text\"],\n"
                "    \"disable\": [\"lowercase-global\"]\n"
                "  }\n"
                "}\n";
    }

    constexpr float default_fov_degrees = 60.0F;
    constexpr float default_near_clip = 0.1F;
    constexpr float default_far_clip = 1000.0F;
}

struct ScriptSystem::Impl {
    // One script: its own Lua state, so its globals (its callbacks among them) are its own. The first instance also runs
    // the behaviours (assets/scripts/behaviours), and exists even when the project has no script.
    struct Instance {
        std::string path;
        std::unique_ptr<LuaRuntime> runtime;
        // The `http` library scripts get; requests belong to one Lua state.
        LuaHttp http;
    };

    GameHost host;
    LuaGameHost bindings;
    std::vector<std::pair<std::string, lua_CFunction>> native_modules;

    std::vector<std::unique_ptr<Instance>> instances;
    // The instance whose on_camera the game camera comes from, found when asked.
    mutable Instance *camera_instance = nullptr;

    std::string fatal_error;

    Scene const *bound_scene = nullptr;

    mutable CameraParams last_camera{};
    mutable bool have_camera = false;

    // Script files changing on disk (editors save in several steps; the queue waits for quiet).
    FileChangeQueue changes;
    FileChangeWatcher watcher;

    auto call(char const *name, std::span<double const> numbers = {}) -> void {
        for (auto const &instance: instances) {
            static_cast<void>(instance->runtime->call(name, numbers));
        }
    }

    auto first_error() const -> std::string {
        if (!fatal_error.empty()) {
            return fatal_error;
        }

        for (auto const &instance: instances) {
            if (!instance->runtime->last_error().empty()) {
                return instance->runtime->last_error();
            }
        }

        return bindings.components ? bindings.components->last_error() : std::string{};
    }
};

ScriptSystem::ScriptSystem() : impl_(std::make_unique<Impl>()) {}

ScriptSystem::~ScriptSystem() = default;

auto ScriptSystem::add_native_module(std::string name, lua_CFunction opener) -> void {
    impl_->native_modules.emplace_back(std::move(name), opener);
}

auto ScriptSystem::set_scripts(std::vector<std::string> scripts) -> void {
    impl_->host.scripts = scripts;
    impl_->bindings.host.scripts = std::move(scripts);
}

auto ScriptSystem::attach_host(GameHost host) -> void {
    impl_->host = std::move(host);
    impl_->bindings.host = impl_->host;
}

auto ScriptSystem::on_populate(Scene &scene, Renderer *renderer, EngineModels const *engine_models) -> void {
    auto &self = *impl_;

    scene.get_registry().clear();

    self.bindings.scene = &scene;
    self.bindings.renderer = renderer;
    self.bindings.engine_models = engine_models;
    self.bindings.animation = std::make_unique<LuaAnimationSystem>();
    self.bindings.collider_slots.clear();
    self.bindings.mesh_colliders.clear();
    self.bound_scene = nullptr;
    self.have_camera = false;
    self.fatal_error.clear();

    self.bindings.components.reset();
    self.camera_instance = nullptr;
    self.watcher.stop();
    self.instances.clear();

    // The scripts to run: the project's list, else assets/scripts/main.lua when there is one (so saving a main.lua into a
    // project is enough), else none (the behaviours still need a Lua state).
    auto paths = self.host.scripts;

    if (paths.empty()) {
        if (auto const main_script = Paths::current().data("assets/scripts/main.lua");
            main_script && std::filesystem::exists(main_script->absolute())) {
            paths.emplace_back("assets/scripts/main.lua");
        }
    }

    // A project's scripts directory has to exist for the watcher to see a script saved into it later.
    if (!self.host.player_mode && Paths::current().in_project()) {
        std::error_code ignored;
        std::filesystem::create_directories(Paths::current().data_root() / "assets" / "scripts", ignored);

        write_editor_config(Paths::current().data_root(), Paths::current().engine_root());
    }

    auto const make_instance = [&](std::string const &path) -> Impl::Instance * {
        auto created = LuaRuntime::create();

        if (!created) {
            self.fatal_error = created.error();
            error("[lua] {}", self.fatal_error);

            return nullptr;
        }

        auto instance = std::make_unique<Impl::Instance>();

        instance->path = path;
        instance->runtime = std::make_unique<LuaRuntime>(std::move(*created));
        instance->runtime->set_host(&self.bindings);
        instance->runtime->set_source_loader([](std::string const &module_path) -> std::optional<std::string> {
            return read_data_file("assets/scripts/" + module_path + ".lua");
        });

        for (auto const &[name, opener]: self.native_modules) {
            instance->runtime->register_native_module(name, opener);
        }

        open_lua_game_api(*instance->runtime);
        instance->http.open(*instance->runtime);

        self.instances.push_back(std::move(instance));

        return self.instances.back().get();
    };

    if (paths.empty()) {
        make_instance({});
    }

    for (auto const &path: paths) {
        make_instance(path);
    }

    if (self.instances.empty()) {
        return;
    }

    self.bindings.components = std::make_unique<BehaviourSystem>(
            *self.instances.front()->runtime, self.bindings, [](std::string const &name) {
                return read_data_file("assets/scripts/behaviours/" + name + ".lua");
            });

    for (auto const &instance: self.instances) {
        if (instance->path.empty()) {
            continue;
        }

        auto const source = read_data_file(instance->path);

        if (!source) {
            self.fatal_error = "the script '" + instance->path + "' was not found in the data directory or a mount";
            error("[lua] {}", self.fatal_error);

            continue;
        }

        info("[lua] Loading '{}'", instance->path);

        if (auto const loaded = instance->runtime->run_main(instance->path, *source); !loaded) {
            self.fatal_error = instance->path + ": " + loaded.error();
            error("[lua] {}", self.fatal_error);
        }
    }

    // Reload when a script, or a module one requires, changes: every data root's assets/scripts directory.
    auto const directories = Paths::current().data_directories("assets/scripts");

    if (!self.host.player_mode && !directories.empty() && !self.watcher.start(self.changes, directories)) {
        warn("[lua] could not watch the scripts for changes; they will not reload on save");
    }

    for (auto const &instance: self.instances) {
        static_cast<void>(instance->runtime->call("on_populate"));
    }
}

auto ScriptSystem::poll_changes() -> bool {
    auto const changed = impl_->changes.drain();

    return std::ranges::any_of(changed, [](std::filesystem::path const &path) { return path.extension() == ".lua"; });
}

auto ScriptSystem::on_update(Scene &scene, float delta_time) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    if (self.bound_scene != &scene) {
        self.bound_scene = &scene;
        self.call("on_bind");
    }

    update_lua_game_physics(self.bindings);

    // Finished http requests call their on_done before the frame's on_update sees the results.
    for (auto const &instance: self.instances) {
        instance->http.update(*instance->runtime);
    }

    std::array const arguments{static_cast<double>(delta_time)};

    self.call("on_update", arguments);

    if (self.bindings.components) {
        self.bindings.components->update(delta_time);
    }

    if (self.bindings.renderer != nullptr) {
        self.bindings.animation->update(scene, *self.bindings.renderer, delta_time);
    }

    // A frame's mouse movement is seen by one update.
    self.bindings.mouse_delta_x = 0.0;
    self.bindings.mouse_delta_y = 0.0;
}

auto ScriptSystem::on_play_stop(Scene &scene) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    if (self.bindings.components) {
        self.bindings.components->stop();
    }
}

auto ScriptSystem::behaviour_names() -> std::span<std::string const> {
    auto &self = *impl_;

    return self.bindings.components ? self.bindings.components->script_names() : std::span<std::string const>{};
}

auto ScriptSystem::behaviour_schema(std::string const &name) -> ScriptSchema const * {
    auto &self = *impl_;

    return self.bindings.components ? self.bindings.components->schema(name) : nullptr;
}

auto ScriptSystem::reload_behaviours() -> void {
    if (impl_->bindings.components) {
        impl_->bindings.components->reload();
    }
}

auto ScriptSystem::clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_editor_into_runtime<Components::Behaviour>(editor_scene, runtime_scene);
}

auto ScriptSystem::subscribe_input(InputContext const &ctx) -> void {
    listen(ctx, &ScriptSystem::on_key_pressed);
    listen(ctx, &ScriptSystem::on_key_released);
    listen(ctx, &ScriptSystem::on_mouse_moved, true);
    listen(ctx, &ScriptSystem::on_mouse_button_pressed);
}

auto ScriptSystem::on_key_released(Scene & , KeyReleasedEvent const &event) -> void {
    impl_->bindings.keys_down.erase(event.key);
}

auto ScriptSystem::on_mouse_moved(Scene & , MouseMovedEvent const &event) -> void {
    impl_->bindings.mouse_delta_x += event.delta_x;
    impl_->bindings.mouse_delta_y += event.delta_y;
}

auto ScriptSystem::on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;
    self.bindings.keys_down.insert(event.key);

    std::array const arguments{static_cast<double>(event.key), static_cast<double>(event.modifiers)};

    self.call("on_key", arguments);
}

auto ScriptSystem::on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    std::array const arguments{static_cast<double>(event.button)};

    self.call("on_mouse_button", arguments);
}

auto ScriptSystem::wants_cursor() const -> bool {
    return std::ranges::any_of(impl_->instances, [](auto const &instance) { return instance->runtime->callback_flag("wants_cursor"); });
}

auto ScriptSystem::on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;

    std::array const arguments{event.ndc_x, event.ndc_y, event.inside ? 1.0 : 0.0};

    self.call("on_cursor", arguments);
}

auto ScriptSystem::on_ui(Scene &scene, Renderer &renderer) -> void {
    auto &self = *impl_;

    self.bindings.scene = &scene;
    self.bindings.renderer = &renderer;

    if (self.bindings.components) {
        self.bindings.components->poll();
    }

    self.call("on_ui");

    auto const error_text = self.first_error();

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

namespace {
    // `on_camera` is the callback of a script that defines global functions: there `camera` is the engine's library.
    auto camera_callback_of(LuaRuntime const &runtime) -> char const * {
        if (runtime.has_callback("on_camera")) {
            return "on_camera";
        }

        return runtime.has_callback("camera") ? "camera" : nullptr;
    }
}

auto ScriptSystem::has_camera() const -> bool {
    auto const &self = *impl_;

    self.camera_instance = nullptr;

    for (auto const &instance: self.instances) {
        if (camera_callback_of(*instance->runtime) != nullptr) {
            self.camera_instance = instance.get();

            return true;
        }
    }

    return false;
}

auto ScriptSystem::ok() const -> bool { return !impl_->instances.empty() && impl_->first_error().empty(); }

auto ScriptSystem::error_text() const -> std::string { return impl_->first_error(); }

auto ScriptSystem::camera(Scene const & , float aspect_ratio) const -> CameraParams {
    auto &self = *impl_;

    glm::vec3 eye{0.0F, 6.0F, -8.0F};
    glm::vec3 target{0.0F};
    float fov = default_fov_degrees;
    float near_clip = default_near_clip;
    float far_clip = default_far_clip;
    glm::vec3 up{0.0F, 1.0F, 0.0F};
    float ortho_height = 0.0F;

    auto *const camera_runtime = has_camera() ? self.camera_instance->runtime.get() : nullptr;

    if (camera_runtime != nullptr) {
        std::array const arguments{static_cast<double>(aspect_ratio)};

        if (camera_runtime->call(camera_callback_of(*camera_runtime), arguments, 1).ok) {
            auto *const state = camera_runtime->state();
            auto const table = lua_gettop(state);

            if (lua_istable(state, table)) {
                auto const eye_value = lua_field_vec3(state, table, "eye", {eye.x, eye.y, eye.z});
                auto const target_value = lua_field_vec3(state, table, "target", {target.x, target.y, target.z});

                eye = {eye_value.x, eye_value.y, eye_value.z};
                target = {target_value.x, target_value.y, target_value.z};
                auto const up_value = lua_field_vec3(state, table, "up", {up.x, up.y, up.z});

                up = {up_value.x, up_value.y, up_value.z};
                ortho_height = static_cast<float>(lua_field_number(state, table, "ortho_height", ortho_height));
                fov = static_cast<float>(lua_field_number(state, table, "fov", fov));
                near_clip = static_cast<float>(lua_field_number(state, table, "near", near_clip));
                far_clip = static_cast<float>(lua_field_number(state, table, "far", far_clip));
            }

            camera_runtime->pop(1);
        }
    }

    // A camera looking along `up` has no horizontal axis; a script that does that gets the default up instead of NaNs.
    auto const forward = target - eye;

    if (!(glm::length(glm::cross(forward, up)) > 1e-6F * std::max(glm::length(forward), 1.0F))) {
        up = glm::vec3{0.0F, 1.0F, 0.0F};

        if (!(glm::length(glm::cross(forward, up)) > 1e-6F * std::max(glm::length(forward), 1.0F))) {
            up = glm::vec3{0.0F, 0.0F, 1.0F};
        }
    }

    // An orthographic view shows `ortho_height` world units top to bottom; the renderer's fov-based fits (shadow
    // cascades, light LOD) see a narrow fov, which is harmless for the unlit scenes this is meant for.
    auto const orthographic = ortho_height > 0.0F && std::isfinite(ortho_height);
    auto const half_height = ortho_height * 0.5F;
    auto const half_width = half_height * aspect_ratio;
    auto const fov_radians = orthographic ? 0.1F : glm::radians(fov);

    CameraParams params{
            .view = glm::lookAtLH(eye, target, up),
            .projection = orthographic ? glm::orthoLH_ZO(-half_width, half_width, -half_height, half_height, near_clip, far_clip)
                                       : glm::perspectiveLH_ZO(glm::radians(fov), aspect_ratio, near_clip, far_clip),
            .near_clip = near_clip,
            .far_clip = far_clip,
            .vertical_fov_radians = fov_radians,
    };

    self.last_camera = params;
    self.have_camera = true;
    self.bindings.inverse_view_projection = glm::inverse(params.projection * params.view);

    return params;
}
