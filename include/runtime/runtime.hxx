#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/vec3.hpp>
#include <imgui.h>

#include "assets/file_change_watcher.hxx"
#include "game/game.hxx"
#include "gpu/context.hxx"
#include "rendering/effect_system.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/particle_system.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/text_renderer.hxx"
#include "serialisation/scene_serialisation.hxx"
#include "terrain/terrain_world.hxx"

// Everything a running game needs, and nothing an editor needs: the renderer, the scene a game is loaded into, the
// copy of it a game plays in, the IGame and the input it receives, terrain, particles and effects. The editor
// (app/application.hxx) is a Runtime with panels, gizmos and file handling on top; an installed game (`lathe-player`)
// is a bare Runtime. The virtual functions are the places the editor hooks in; every default is the player's
// behaviour, so a Runtime on its own is a complete host.
struct Runtime {
    explicit Runtime(VulkanContext &ctx) noexcept;
    virtual ~Runtime();

    Runtime(Runtime const &) = delete;
    auto operator=(Runtime const &) -> Runtime & = delete;
    Runtime(Runtime &&) = delete;
    auto operator=(Runtime &&) -> Runtime & = delete;

    VulkanContext &context;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<debug_draw::DebugRenderer> debug_renderer;
    std::unique_ptr<gui::ImGuiRenderer> imgui_renderer;
    std::unique_ptr<TextRenderer> text_renderer;

    std::vector<OverlayRegistration> overlays;
    FileChangeWatcher shader_watcher_;

    // The scene the game populates or a scene file loads into. Playing clones it into runtime_scene, so stopping
    // returns to it untouched.
    std::unique_ptr<Scene> source_scene = std::make_unique<Scene>(*renderer);
    std::unique_ptr<Scene> runtime_scene;
    bool is_playing = false;

    bool play_fullscreen = false;

    // An installed game: no editor to fall back into, so Escape and losing focus do not leave play.
    bool player_mode = false;

    bool game_mouse_captured = false;

    [[nodiscard]] auto active_scene() const noexcept -> Scene * {
        return is_playing ? runtime_scene.get() : source_scene.get();
    }

    std::unique_ptr<IGame> game;

    std::string game_name;
    std::string script_entry;

    auto play() -> void;
    auto stop() -> void;

    // Loads a scene file into source_scene over the next few frames (stopping play first): models stream in, then
    // update() installs the result. With play_when_loaded the game starts fullscreen once it is in.
    std::filesystem::path scene_path;
    std::shared_ptr<AssetPack const> scene_pack;
    std::optional<SceneLoadJob> scene_load_job;
    std::string scene_status;
    bool play_when_loaded = false;

    auto open_scene(std::filesystem::path path) -> void;
    auto update_scene_load() -> void;

    auto capture_mouse() -> void;
    auto release_mouse() -> void;

    EngineModels engine_models{};

    std::unique_ptr<TerrainWorld> terrain;

    bool terrain_enabled = true;
    bool game_hooks_enabled = true;
    float last_delta_time = 0.0F;
    ParticleSystem particle_system;
    EffectSystem effect_system;

    [[nodiscard]] auto active_terrain() const noexcept -> TerrainWorld * {
        return terrain_enabled ? terrain.get() : nullptr;
    }

    float elapsed_time = 0.0F;
    static constexpr auto stats_record_start_time = 5.0F;
    [[nodiscard]] constexpr auto can_start_recording_statistics() const { return elapsed_time > stats_record_start_time; }

    // Where the game is drawn: the whole window while playing fullscreen, else the editor's viewport panel (the
    // runtime only knows its screen rectangle).
    bool viewport_hovered = false;
    ImVec2 viewport_screen_pos{};
    ImVec2 viewport_content_size{};
    int viewport_seen_frame = -1;
    bool screenshot_viewport_only = true;

    bool mouse_dragging = false;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    [[nodiscard]] auto cursor_over_game() const -> CursorPositionEvent;

    auto on_startup() -> void;

    auto register_overlays() -> void;

    auto request_screenshot() -> void;

    // Per-frame, in order: pre_update (before a benchmark steers the frame), update, then on_ui while the frame's
    // ImGui is open.
    virtual auto pre_update(float delta_time) -> void { (void) delta_time; }
    virtual auto update(float delta_time) -> void;
    virtual auto on_ui(std::uint32_t frame_index) -> void;

    virtual auto on_event(KeyPressedEvent ev) -> bool;
    virtual auto on_event(KeyReleasedEvent ev) -> bool;
    virtual auto on_event(MouseMovedEvent ev) -> bool;
    virtual auto on_event(MouseScrolledEvent ev) -> bool;
    virtual auto on_event(MouseButtonPressedEvent ev) -> bool;
    virtual auto on_event(MouseButtonReleasedEvent ev) -> bool;

    // Called before the renderer is torn down: drop everything that holds GPU resources.
    virtual auto on_shutdown() -> void;

    virtual auto on_files_dropped(std::span<std::filesystem::path const> paths) -> void { (void) paths; }

    virtual auto on_frame_timings(std::span<frame_graph::PassTiming const> passes) -> void { (void) passes; }

    // The camera used while not playing (a game supplies its own while playing), and the point terrain streams
    // around while not playing.
    [[nodiscard]] virtual auto editing_camera(float aspect_ratio) const -> CameraParams {
        (void) aspect_ratio;
        return {};
    }
    [[nodiscard]] virtual auto observer_position() const -> glm::vec3 { return glm::vec3{0.0F}; }

protected:
    // Called at the start of play() and stop(), before the play scene is created or torn down.
    virtual auto on_play_starting() -> void {}
    virtual auto on_play_stopping() -> void {}

    // Called from on_startup() once ImGui is up.
    virtual auto create_ui_resources() -> void {}

    // Called when source_scene holds a fresh scene: populated by the game, or loaded from a file.
    virtual auto on_source_scene_replaced() -> void {}

    // Called by open_scene() before the load starts.
    virtual auto on_scene_opening() -> void {}
};
