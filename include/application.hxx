#pragma once

#include "context.hxx"
#include "debug_renderer.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "game.hxx"
#include "imgui_renderer.hxx"
#include "input_events.hxx"
#include "render_stage.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "terminal_widget.hxx"

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
    bool is_playing = false;

    // Computed rather than a separately-mutated pointer: play()/stop() only
    // need to flip is_playing, so there's no window where this field and
    // is_playing can desync (e.g. a re-entrant stop() call, or something
    // capturing a scene pointer that outlives the run it was captured for).
    [[nodiscard]] auto active_scene() const noexcept -> Scene * {
        return is_playing ? runtime_scene.get() : editor_scene.get();
    }

    // Supplied by main.cxx (via the game's create_game() factory) before
    // on_startup() runs. Application never constructs game content itself --
    // it hands editor_scene/the active scene to `game` at the points listed
    // on IGame (game.hxx).
    std::unique_ptr<IGame> game;

    auto play() -> void;
    auto stop() -> void;

    EngineModels engine_models{};

    // Seconds since startup -- forwarded to UBO.time each frame so the
    // wind shader (wind.slang) has something to animate against.
    float elapsed_time = 0.0F;
    static constexpr auto stats_record_start_time = 5.0F;
    [[nodiscard]] constexpr auto can_start_recording_statistics() { return elapsed_time > stats_record_start_time; }


    std::array<ScrollingBuffer, stage_count> timing_buffers;
    float timing_x = 0.0F;

    EditorCamera camera;

    float light_azimuth_degrees = 30.0F;
    float light_elevation_degrees = 55.0F;

    bool mouse_dragging = false;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    gui::TerminalWidget terminal_widget;

    auto update(float delta_time) -> void;

    auto on_startup() -> void;

    auto on_event(KeyPressedEvent ev) -> bool;
    auto on_event(KeyReleasedEvent ev) -> bool;
    auto on_event(MouseMovedEvent ev) -> bool;
    auto on_event(MouseScrolledEvent ev) -> bool;
    auto on_event(MouseButtonPressedEvent ev) -> bool;
    auto on_event(MouseButtonReleasedEvent ev) -> bool;
};
