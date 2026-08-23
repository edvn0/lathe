#pragma once

#include "context.hxx"
#include "debug_renderer.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "imgui_renderer.hxx"
#include "material.hxx"
#include "model.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"

struct KeyPressedEvent {
    std::int32_t key{};
    std::int32_t modifiers{};
};

struct KeyReleasedEvent {
    std::int32_t key{};
    std::int32_t modifiers{};
};

// Raw cursor delta in pixels since the previous callback -- not an
// absolute position. Application decides whether it counts as a
// "look" based on whether a drag is currently active.
struct MouseMovedEvent {
    double delta_x{};
    double delta_y{};
};

struct MouseScrolledEvent {
    double delta_x{};
    double delta_y{};
};

struct MouseButtonPressedEvent {
    std::int32_t button{};
    std::int32_t modifiers{};
};

struct MouseButtonReleasedEvent {
    std::int32_t button{};
    std::int32_t modifiers{};
};

struct ScrollingBuffer {
    std::int32_t max_size;
    std::int32_t offset = 0;
    std::vector<ImVec2> data;

    explicit ScrollingBuffer(const std::int32_t m = 600U) : max_size(m) {
        data.reserve(static_cast<std::size_t>(max_size));
    }

    [[gnu::always_inline]]
    constexpr auto size() -> decltype(auto) {
        return data.size();
    }

    auto add_point(float x, float y) -> void {

        if (std::cmp_less(size(), max_size)) {
            data.emplace_back(x, y);
        } else {
            data[static_cast<std::size_t>(offset)] = ImVec2(x, y);
            offset = (offset + 1) % max_size;
        }
    }
};

struct Application {
    explicit Application(VulkanContext &ctx) noexcept;
    ~Application();
    void on_ui();

    VulkanContext &context;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<debug_draw::DebugRenderer> debug_renderer;
    std::unique_ptr<gui::ImGuiRenderer> imgui_renderer;
    ShaderHotReloadWatcher shader_watcher_;

    std::unique_ptr<Scene> editor_scene = std::make_unique<Scene>(*renderer);
    std::unique_ptr<Scene> runtime_scene;
    Scene *active_scene = editor_scene.get();
    bool is_playing = false;

    entt::entity player_entity{entt::null};
    PlayerController player_controller;
    PlayerCamera player_camera;


    auto play() -> void;
    auto stop() -> void;

    EngineModels engine_models{};

    // Set by recreate_entities() -- reused by shoot_bullet() so
    // projectiles are built from the same model/collision-shape
    // relationship as the grid cubes and floor.
    ModelHandle cube_model{};
    glm::vec3 cube_half_extents{0.5F};

    ModelHandle house_model{};
    ModelHandle tree_model{};

    // Wind-swaying grass material -- created once in recreate_entities()
    // and shared as a MaterialOverride across every grass clump entity.
    MaterialHandle grass_material{};

    // Seconds since startup -- forwarded to UBO.time each frame so the
    // wind shader (wind.slang) has something to animate against.
    float elapsed_time = 0.0F;

    std::array<ScrollingBuffer, stage_count> timing_buffers;
    float timing_x = 0.0F;

    EditorCamera camera;

    float light_azimuth_degrees = 30.0F;
    float light_elevation_degrees = 55.0F;

    bool mouse_dragging = false;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    auto update(float delta_time) -> void;

    auto on_startup() -> void;

    auto on_event(KeyPressedEvent ev) -> bool;
    auto on_event(KeyReleasedEvent ev) -> bool;
    auto on_event(MouseMovedEvent ev) -> bool;
    auto on_event(MouseScrolledEvent ev) -> bool;
    auto on_event(MouseButtonPressedEvent ev) -> bool;
    auto on_event(MouseButtonReleasedEvent ev) -> bool;

    [[nodiscard]] auto cursor_ray() const -> std::pair<glm::vec3, glm::vec3>;

    auto shoot_bullet(std::size_t n = 1) -> void;
    auto recreate_entities() -> void;
};
