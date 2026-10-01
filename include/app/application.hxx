#pragma once

// ImGuizmo.h needs imgui.h included first.
#include "rendering/imgui_renderer.hxx"
#include "rendering/overlay.hxx"

#include <ImGuizmo.h>

#include "rendering/editor_icons.hxx"
#include "rendering/file_browser.hxx"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "assets/material_storage.hxx"
#include "gpu/context.hxx"
#include "rendering/debug_renderer.hxx"
#include "scene/editor_camera.hxx"
#include "rendering/engine_models.hxx"
#include "app/game.hxx"
#include "scene/input_events.hxx"
#include "rendering/render_stage.hxx"
#include "rendering/scene.hxx"
#include "assets/shader_hot_reload_watcher.hxx"
#include "rendering/terminal_widget.hxx"
#include "terrain/terrain_world.hxx"
#include "serialisation/scene_serialisation.hxx"

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

    // Lets the Viewport panel show renderer->viewport_target(frame_index).
    auto on_ui(std::uint32_t frame_index) -> void;

    VulkanContext &context;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<debug_draw::DebugRenderer> debug_renderer;
    std::unique_ptr<gui::ImGuiRenderer> imgui_renderer;
    std::unique_ptr<gui::EditorIcons> editor_icons;

    // Debug lines and the ImGui frame. Declared after the renderers so they unregister first.
    std::vector<OverlayRegistration> overlays;
    ShaderHotReloadWatcher shader_watcher_;

    std::unique_ptr<Scene> editor_scene = std::make_unique<Scene>(*renderer);
    std::unique_ptr<Scene> runtime_scene;
    bool is_playing = false;

    // Fullscreen play covers the whole swapchain and captures the cursor. Embedded play (the default) renders into
    // the Viewport panel and keeps the editor usable.
    bool play_fullscreen = false;

    // Embedded play only: set by a click in the Viewport, cleared by Escape or stop().
    bool game_mouse_captured = false;

    enum class ModelBrowseTarget : std::uint8_t { spawn_entity, inspector };
    ModelBrowseTarget model_browse_target = ModelBrowseTarget::spawn_entity;

    // The rename field takes keyboard focus on its first frame.
    bool rename_needs_focus = false;
    bool inspector_name_dirty = false;

    [[nodiscard]] auto active_scene() const noexcept -> Scene * {
        return is_playing ? runtime_scene.get() : editor_scene.get();
    }

    // Set by main.cxx before on_startup().
    std::unique_ptr<IGame> game;

    auto play() -> void;
    auto stop() -> void;

    // Hides and locks the cursor for mouse-look. ImGui ignores the mouse while it is captured: the disabled cursor's
    // position is an unbounded virtual accumulator, which ImGui's GLFW backend would otherwise keep hit-testing.
    auto capture_mouse() -> void;
    auto release_mouse() -> void;

    EngineModels engine_models{};

    // Null when the game has no streaming terrain.
    std::unique_ptr<TerrainWorld> terrain;

    // Seconds since startup.
    float elapsed_time = 0.0F;
    static constexpr auto stats_record_start_time = 5.0F;
    [[nodiscard]] constexpr auto can_start_recording_statistics() { return elapsed_time > stats_record_start_time; }

    std::array<ScrollingBuffer, stage_count> timing_buffers;
    float timing_x = 0.0F;

    EditorCamera camera;

    // The selection itself lives in selection_context(); it's cleared on play()/stop(), since the active registry
    // changes.

    // Inspector model picker: the entity whose Model the picked file replaces.
    entt::entity model_browse_entity = entt::null;

    // Hierarchy inline rename: the row showing a text field, and that field's text.
    entt::entity renaming_entity = entt::null;
    std::array<char, 128> rename_buffer{};

    // Inspector name field: the entity its text belongs to, and whether it holds an uncommitted edit.
    entt::entity inspector_name_entity = entt::null;
    std::array<char, 128> inspector_name_buffer{};

    // The Save As / unsaved-changes path field.
    std::array<char, 512> save_as_buffer{};
    ImGuizmo::OPERATION gizmo_operation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmo_mode = ImGuizmo::WORLD;

    // Updated each frame by the Viewport panel. Used for input routing, the gizmo rect and the render size.
    bool viewport_hovered = false;
    ImVec2 viewport_screen_pos{};
    ImVec2 viewport_content_size{};
    bool screenshot_viewport_only = true;

    std::string hierarchy_search;

    float light_azimuth_degrees = 30.0F;
    float light_elevation_degrees = 55.0F;

    bool mouse_dragging = false;

    // Scene file modals, opened from outside the ImGui frame (shortcuts, drops) on the next one.
    bool unsaved_changes_popup_requested = false;
    bool save_as_popup_requested = false;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    gui::TerminalWidget terminal_widget;

    // Shared by the "Load Model" panel and the Inspector's model picker; `model_browse_target` says which opened it.
    gui::FileBrowser model_browser;

    // Models spawned from the "Load Model" panel, newest last, so it can report how each load went.
    struct StreamedModelLoad {
        Scene *scene = nullptr;
        entt::entity entity = entt::null;
        ModelHandle model{};
        std::string file_name;
        // Set once the load installed or failed.
        bool settled = false;
        std::string status;
    };
    static constexpr std::size_t max_listed_model_loads = 8;
    std::vector<StreamedModelLoad> model_loads;

    // Streams `path` in and spawns an entity for it in the active scene.
    auto spawn_streamed_model(std::filesystem::path const &path) -> void;

    // Settles finished entries of `model_loads`: records the outcome and gives installed models their collider.
    auto update_model_loads() -> void;

    // Points `entity`'s Model at `model`, taking over one reference the caller holds on `model` and releasing the
    // entity's reference on its previous model, if it owned one.
    auto set_entity_model(entt::registry &registry, entt::entity entity, ModelHandle model) -> void;

    // ---- Scene files (.lbf); see scene_files.cxx.

    // The file the editor scene was last opened from or saved to; empty while untitled.
    std::filesystem::path scene_path;

    // scene_fingerprint() of the editor scene at the last open/save/populate; differs once the scene is edited.
    std::uint64_t scene_clean_fingerprint = 0;

    // The file the scene was opened from, kept so a save copies unchanged cooked assets instead of re-cooking.
    std::shared_ptr<AssetPack const> scene_pack;

    std::optional<SceneSaveJob> scene_save_job;
    std::optional<SceneLoadJob> scene_load_job;

    // Opened once the running save finishes (the "Save and open" choice).
    std::optional<std::filesystem::path> open_after_save;

    // A file waiting on the unsaved-changes prompt.
    std::optional<std::filesystem::path> pending_scene_open;

    gui::FileBrowser scene_browser;
    std::string scene_status;

    [[nodiscard]] auto editor_scene_fingerprint() -> std::uint64_t;
    [[nodiscard]] auto editor_scene_dirty() -> bool;
    auto mark_editor_scene_clean() -> void;

    // Opens `path`, first asking about unsaved changes if there are any.
    auto request_open_scene(std::filesystem::path path) -> void;
    auto start_open_scene(std::filesystem::path path) -> void;
    // An empty path asks for one.
    auto start_save_scene(std::filesystem::path path) -> void;

    // Steps the background save/load. Called from update().
    auto update_scene_jobs() -> void;

    // OS drag-and-drop onto the window: .lbf opens the scene, .gltf/.glb spawns the model.
    auto on_files_dropped(std::span<std::filesystem::path const> paths) -> void;

    // The "Scene" panel, the unsaved-changes and save-as modals, and the open-scene browser.
    auto draw_scene_file_ui() -> void;

    // State of the "New Material" popup, kept across frames.
    MaterialCreateInfo new_material_info{};
    std::string new_material_name;

    std::string save_material_name;

    // Deleting from the Assets panel queues `commit` to run once elapsed_time reaches `delete_at`. Until then the
    // entry can be restored by dropping it from the queue.
    struct PendingDeletion {
        std::string label;
        float delete_at = 0.0F;
        std::move_only_function<void()> commit;
    };
    static constexpr float deletion_grace_seconds = 20.0F;
    std::vector<PendingDeletion> pending_deletions;

    auto update(float delta_time) -> void;

    auto on_startup() -> void;

    // Registers the debug-line and ImGui overlays. Called once both renderers exist.
    auto register_overlays() -> void;

    auto request_screenshot() -> void;
    auto on_event(KeyPressedEvent ev) -> bool;
    auto on_event(KeyReleasedEvent ev) -> bool;
    auto on_event(MouseMovedEvent ev) -> bool;
    auto on_event(MouseScrolledEvent ev) -> bool;
    auto on_event(MouseButtonPressedEvent ev) -> bool;
    auto on_event(MouseButtonReleasedEvent ev) -> bool;
};
