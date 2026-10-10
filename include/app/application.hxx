#pragma once

#include "rendering/imgui_renderer.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "rendering/overlay.hxx"

#include <ImGuizmo.h>

#include "rendering/editor_icons.hxx"
#include "rendering/file_browser.hxx"

#include <array>
#include <span>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/frame_graph_editor.hxx"
#include "app/game.hxx"
#include "app/stage_registry.hxx"
#include "assets/material_storage.hxx"
#include "assets/shader_hot_reload_watcher.hxx"
#include "gpu/context.hxx"
#include "rendering/cluster_grid.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/scene.hxx"
#include "rendering/terminal_widget.hxx"
#include "scene/editor_camera.hxx"
#include "scene/hierarchy_model.hxx"
#include "scene/input_events.hxx"
#include "scripting/script_widget.hxx"
#include "serialisation/scene_serialisation.hxx"
#include "terrain/terrain_world.hxx"

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

    auto on_ui(std::uint32_t frame_index) -> void;

    VulkanContext &context;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<debug_draw::DebugRenderer> debug_renderer;
    std::unique_ptr<gui::ImGuiRenderer> imgui_renderer;
    std::unique_ptr<gui::EditorIcons> editor_icons;

    std::vector<OverlayRegistration> overlays;
    ShaderHotReloadWatcher shader_watcher_;

    std::unique_ptr<Scene> editor_scene = std::make_unique<Scene>(*renderer);
    std::unique_ptr<Scene> runtime_scene;
    bool is_playing = false;

    bool play_fullscreen = false;

    bool player_mode = false;

    bool game_mouse_captured = false;

    enum class ModelBrowseTarget : std::uint8_t { spawn_entity, inspector };
    ModelBrowseTarget model_browse_target = ModelBrowseTarget::spawn_entity;

    bool rename_needs_focus = false;
    bool inspector_name_dirty = false;

    [[nodiscard]] auto active_scene() const noexcept -> Scene * {
        return is_playing ? runtime_scene.get() : editor_scene.get();
    }

    std::unique_ptr<IGame> game;

    auto play() -> void;
    auto stop() -> void;

    auto capture_mouse() -> void;
    auto release_mouse() -> void;

    EngineModels engine_models{};

    std::unique_ptr<TerrainWorld> terrain;

    bool terrain_enabled = true;
    bool game_hooks_enabled = true;
    float last_delta_time = 0.0F;

    [[nodiscard]] auto active_terrain() const noexcept -> TerrainWorld * {
        return terrain_enabled ? terrain.get() : nullptr;
    }

    float elapsed_time = 0.0F;
    static constexpr auto stats_record_start_time = 5.0F;
    [[nodiscard]] constexpr auto can_start_recording_statistics() { return elapsed_time > stats_record_start_time; }

    struct TimingSeries {
        std::string id;
        std::string label;
        ScrollingBuffer buffer;
    };
    std::vector<TimingSeries> timing_series;

    auto add_pass_timings(std::span<frame_graph::PassTiming const> passes) -> void;
    float timing_x = 0.0F;

    EditorCamera camera;

    entt::entity model_browse_entity = entt::null;

    entt::entity renaming_entity = entt::null;
    std::array<char, 128> rename_buffer{};

    entt::entity inspector_name_entity = entt::null;
    std::array<char, 128> inspector_name_buffer{};

    std::array<char, 512> save_as_buffer{};
    ImGuizmo::OPERATION gizmo_operation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmo_mode = ImGuizmo::WORLD;

    bool viewport_hovered = false;
    ImVec2 viewport_screen_pos{};
    ImVec2 viewport_content_size{};
    bool screenshot_viewport_only = true;

    std::string hierarchy_search;

    HierarchyModel hierarchy_model;
    Scene const *hierarchy_model_scene = nullptr;
    std::uint64_t hierarchy_model_revision = 0;

    entt::entity hierarchy_context_entity = entt::null;

    stages::Registry stage_registry;
    gui::FrameGraphEditor frame_graph_editor;

    bool mouse_dragging = false;

    bool unsaved_changes_popup_requested = false;
    bool save_as_popup_requested = false;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    gui::TerminalWidget terminal_widget;

    gui::ScriptWidget script_widget;

    gui::FileBrowser model_browser;

    bool browsing_environment = false;
    std::optional<std::filesystem::path> pending_model_pick;

    struct StreamedModelLoad {
        Scene *scene = nullptr;
        entt::entity entity = entt::null;
        ModelHandle model{};
        std::string file_name;
        bool settled = false;
        std::string status;
    };
    static constexpr std::size_t max_listed_model_loads = 8;
    std::vector<StreamedModelLoad> model_loads;

    auto spawn_streamed_model(std::filesystem::path const &path) -> void;

    auto update_model_loads() -> void;

    auto set_entity_model(entt::registry &registry, entt::entity entity, ModelHandle model) -> void;

    std::filesystem::path scene_path;

    std::uint64_t scene_clean_fingerprint = 0;

    std::shared_ptr<AssetPack const> scene_pack;

    std::optional<SceneSaveJob> scene_save_job;
    std::optional<SceneLoadJob> scene_load_job;

    std::optional<std::filesystem::path> open_after_save;

    std::optional<std::filesystem::path> pending_scene_open;

    gui::FileBrowser scene_browser;
    std::string scene_status;

    [[nodiscard]] auto editor_scene_fingerprint() -> std::uint64_t;
    [[nodiscard]] auto editor_scene_dirty() -> bool;
    auto mark_editor_scene_clean() -> void;

    auto request_open_scene(std::filesystem::path path) -> void;
    auto start_open_scene(std::filesystem::path path) -> void;
    auto start_save_scene(std::filesystem::path path) -> void;

    auto update_scene_jobs() -> void;

    auto on_files_dropped(std::span<std::filesystem::path const> paths) -> void;

    auto draw_scene_file_ui() -> void;

    MaterialCreateInfo new_material_info{};
    std::string new_material_name;

    std::string save_material_name;

    struct PendingDeletion {
        std::string label;
        float delete_at = 0.0F;
        std::move_only_function<void()> commit;
    };
    static constexpr float deletion_grace_seconds = 20.0F;
    std::vector<PendingDeletion> pending_deletions;

    auto update(float delta_time) -> void;

    [[nodiscard]] auto cursor_over_game() const -> CursorPositionEvent;

    auto on_startup() -> void;

    auto register_overlays() -> void;

    auto request_screenshot() -> void;
    auto on_event(KeyPressedEvent ev) -> bool;
    auto on_event(KeyReleasedEvent ev) -> bool;
    auto on_event(MouseMovedEvent ev) -> bool;
    auto on_event(MouseScrolledEvent ev) -> bool;
    auto on_event(MouseButtonPressedEvent ev) -> bool;
    auto on_event(MouseButtonReleasedEvent ev) -> bool;
};
