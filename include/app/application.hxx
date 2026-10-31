#pragma once

#include "scene/selection_context.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "runtime/runtime.hxx"

#include <ImGuizmo.h>

#include "rendering/editor_icons.hxx"
#include "rendering/file_browser.hxx"

#include <array>
#include <chrono>
#include <span>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/cache_menu.hxx"
#include "app/commands.hxx"
#include "app/frame_graph_editor.hxx"
#include "game/game.hxx"
#include "app/game_packager.hxx"
#include "app/scene_inspector.hxx"
#include "app/stage_registry.hxx"
#include "app/text_placement.hxx"
#include "assets/material_storage.hxx"
#include "assets/file_change_watcher.hxx"
#include "gpu/context.hxx"
#include "rendering/effect_system.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/particle_system.hxx"
#include "rendering/scene.hxx"
#include "rendering/terminal_widget.hxx"
#include "app/gpu_timings_view.hxx"
#include "core/gpu_timings.hxx"
#include "scene/editor_camera.hxx"
#include "scene/hierarchy_model.hxx"
#include "scripting/script_widget.hxx"
#include "serialisation/scene_serialisation.hxx"
#include "terrain/terrain_world.hxx"

struct Application : Runtime {
    explicit Application(VulkanContext &ctx) noexcept;
    ~Application() override;

    auto on_ui(std::uint32_t frame_index) -> void override;

    std::unique_ptr<gui::EditorIcons> editor_icons;

    // Which dockable panels are shown; the Window menu toggles them. The viewport is always there.
    struct PanelVisibility {
        bool hierarchy = true;
        bool inspector = true;
        bool assets = false;
        bool console = true;
        bool script = false;
        // The current game's own panels (IGame::on_ui), e.g. the basic game's Grass tuning.
        bool game = false;
        bool project_settings = false;
        bool frame_graph = false;
        bool scene_stats = false;
        bool frame_timings = false;
        bool events = false;
        bool cvars = false;
        bool memory = false;
        bool scene_inspector = false;
    } panels;

    // Opens the named panels (the PanelVisibility field names); unknown names are logged.
    auto show_panels_from(std::string_view names) -> void;

    // True once the OS title bar is gone and the menu bar carries the title, drag area and window buttons.
    int project_settings_page = 0;
    bool reset_layout_requested = false;
    bool custom_title_bar = false;
    auto remove_window_decorations() -> void;
    auto handle_window_resize() -> void;
    auto draw_main_menu_bar() -> void;
    CacheMenu cache_menu;
    auto request_import_model() -> void;

    enum class ModelBrowseTarget : std::uint8_t { spawn_entity, inspector };
    ModelBrowseTarget model_browse_target = ModelBrowseTarget::spawn_entity;

    bool rename_needs_focus = false;
    bool inspector_name_dirty = false;

    GpuTimings gpu_timings;
    float gpu_frame_ms = 0.0F;
    gui::GpuTimingsViewState gpu_timings_view;

    auto on_frame_timings(FrameTimings const &timings) -> void override;

    // One pane of the editor's central area: a view of the scene with its own camera. Pane 0 is the renderer's primary
    // view (Runtime's viewport_* fields mirror it, which the run loop sizes the renderer by); pane 1 is an extra view
    // that exists while the viewport is split.
    struct ViewportPane {
        EditorCamera camera;
        Renderer::ViewId view = Renderer::primary_view;
        ImVec2 screen_pos{};
        ImVec2 content_size{};
        bool hovered = false;
        int seen_frame = -1;
        VkExtent2D pending_extent{};
        std::chrono::steady_clock::time_point pending_since{};
    };

    std::array<ViewportPane, 2> viewports;
    // The pane that takes keyboard camera input, the gizmo and the F-key view modes: the one the pointer last was in.
    std::size_t active_viewport = 0;
    bool second_viewport_open = false;
    // The second viewport window was just opened and still has to be docked beside the first.
    bool split_viewport_requested = false;
    bool second_viewport_camera_seeded = false;

    [[nodiscard]] auto active_camera() noexcept -> EditorCamera & { return viewports[active_viewport].camera; }
    [[nodiscard]] auto active_camera() const noexcept -> EditorCamera const & {
        return viewports[active_viewport].camera;
    }
    auto draw_viewport(std::size_t index, char const *title, std::uint32_t frame_index) -> void;

    // The Add menu (add_menu.cxx): Shift+A over a viewport opens a popup at the cursor listing the "add" commands.
    // New entities go where the cursor pointed when it opened, on the ground plane or in front of the camera.
    editor::CommandRegistry commands;
    bool add_menu_requested = false;
    ImGuiTextFilter add_menu_filter;
    glm::vec3 add_menu_position{0.0F};
    auto register_commands() -> void;
    auto register_editor_commands() -> void;
    // The shortcut bound to a command, as menu text; empty if it has none.
    [[nodiscard]] auto shortcut_label(std::string_view id) const -> std::string;
    auto draw_add_menu() -> void;
    [[nodiscard]] auto cursor_spawn_position(ViewportPane const &pane) const -> glm::vec3;

    auto dock_second_viewport(ImGuiID dockspace_id) -> void;
    auto update_views() -> void override;

    enum class CameraDrag : std::uint8_t { none, orbit, pan };
    CameraDrag camera_drag = CameraDrag::none;
    float camera_pivot_distance = 10.0F;

    [[nodiscard]] auto camera_pivot() const -> glm::vec3;
    auto focus_selection() -> void;

    entt::entity model_browse_entity = entt::null;

    entt::entity renaming_entity = entt::null;
    std::array<char, 128> rename_buffer{};

    entt::entity inspector_name_entity = entt::null;
    std::array<char, 128> inspector_name_buffer{};

    std::array<char, 512> save_as_buffer{};
    ImGuizmo::OPERATION gizmo_operation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmo_mode = ImGuizmo::WORLD;
    TextPlacement text_placement;

    std::string hierarchy_search;

    HierarchyModel hierarchy_model;
    Scene const *hierarchy_model_scene = nullptr;
    std::uint64_t hierarchy_model_revision = 0;

    entt::entity hierarchy_context_entity = entt::null;

    // Ctrl+C in the hierarchy: the entities (roots of the selection) that Ctrl+V duplicates, from the scene they are in.
    std::vector<entt::entity> hierarchy_clipboard;
    Scene const *hierarchy_clipboard_scene = nullptr;

    stages::Registry stage_registry;
    gui::FrameGraphEditor frame_graph_editor;

    bool unsaved_changes_popup_requested = false;
    bool save_as_popup_requested = false;

    gui::TerminalWidget terminal_widget;

    gui::ScriptWidget script_widget;

    gui::FileBrowser model_browser;

    gui::SceneInspector scene_inspector;

    bool browsing_environment = false;
    std::optional<std::filesystem::path> pending_model_pick;
    // The singled-out material slot the inspector last scrolled to, so it scrolls once per pick, not every frame.
    SelectionContext::SubmeshHighlight scrolled_to_submesh;

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

    // The loading card over the viewport while models and textures stream in (load_progress.cxx).
    struct LoadProgress {
        std::size_t peak_textures = 0;
        float busy_seconds = 0.0F;
    } load_progress;
    auto draw_load_progress() -> void;

    auto update_model_loads() -> void;

    auto set_entity_model(entt::registry &registry, entt::entity entity, ModelHandle model) -> void;

    std::uint64_t scene_clean_fingerprint = 0;

    std::optional<SceneSaveJob> scene_save_job;

    std::optional<std::filesystem::path> open_after_save;

    std::optional<std::filesystem::path> pending_scene_open;

    gui::FileBrowser scene_browser;

    [[nodiscard]] auto source_scene_fingerprint() -> std::uint64_t;
    [[nodiscard]] auto source_scene_dirty() -> bool;
    auto mark_source_scene_clean() -> void;

    auto request_open_scene(std::filesystem::path path) -> void;
    auto start_save_scene(std::filesystem::path path) -> void;

    auto update_scene_save() -> void;

    auto on_files_dropped(std::span<std::filesystem::path const> paths) -> void override;

    // Leaving with changes that would be lost (confirm_exit): the scene, edited project settings, a save or package that
    // has not finished. The popup offers to save what can be saved, exit anyway, or stay.
    bool exit_confirmed = false;
    bool exit_popup_requested = false;
    bool exit_after_save = false;
    bool exit_scene_dirty = false;
    std::vector<std::string> exit_changes;
    [[nodiscard]] auto unsaved_changes() -> std::vector<std::string>;
    [[nodiscard]] auto confirm_exit() -> bool override;
    auto draw_exit_ui() -> void;
    [[nodiscard]] auto project_form_dirty() const -> bool;
    auto save_project_form() -> bool;

    auto draw_scene_file_ui() -> void;

    // Projects (project_files.cxx): open or create one and switch to it in process (Runtime::switch_project).
    gui::FileBrowser project_browser;
    std::optional<std::filesystem::path> pending_project_open;
    bool project_switch_popup_requested = false;
    bool new_project_popup_requested = false;
    std::array<char, 512> new_project_directory{};
    std::array<char, 128> new_project_name{};

    // The Project page of the settings window: the open project's manifest being edited, reloaded when another opens.
    struct ProjectForm {
        std::filesystem::path root;
        std::array<char, 128> name{};
        std::array<char, 128> title{};
        std::array<char, 32> version{};
        std::array<char, 128> game{};
        std::array<char, 256> entry{};
        std::array<char, 256> scene{};
        std::string status;
    } project_form;
    std::filesystem::path package_form_project;
    auto draw_project_settings_page() -> void;
    std::string new_project_status;
    auto request_open_project(std::filesystem::path root) -> void;
    auto request_new_project_dialog() -> void;
    auto create_project() -> bool;
    auto draw_project_ui() -> void;

    std::string game_choice;
    std::optional<std::string> pending_game_switch;
    bool game_switch_popup_requested = false;
    auto request_switch_game(std::string name) -> void;

    // Packaging (package_ui.cxx): the editor cooks the scene, then GamePackageJob builds the game on a worker thread
    // while a modal progress popup dims the editor and update() stops stepping the simulation.
    struct PackageForm {
        std::array<char, 128> name{};
        std::array<char, 128> title{};
        std::array<char, 32> version{};
        std::array<char, 512> output{};
        bool archive = true;
    } package_form;

    std::optional<SceneSaveJob> package_scene_job;
    std::optional<GamePackageJob> package_job;
    std::optional<PackageOptions> package_pending;
    std::filesystem::path package_staged_scene;
    std::string package_status;
    bool package_popup_requested = false;
    bool package_progress_popup_open = false;

    [[nodiscard]] auto packaging_active() const noexcept -> bool {
        return package_scene_job.has_value() || package_job.has_value();
    }

    auto request_package_dialog() -> void;
    auto start_package() -> void;
    auto update_package_jobs() -> void;
    auto draw_package_ui() -> void;

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

    auto pre_update(float delta_time) -> void override;
    auto update(float delta_time) -> void override;

    [[nodiscard]] auto editing_camera(float aspect_ratio) const -> CameraParams override;
    [[nodiscard]] auto observer_position() const -> glm::vec3 override { return viewports[0].camera.position(); }


protected:
    auto draw_event_metrics() -> void;
    auto draw_cvar_panel() -> void;
    std::array<char, 128> cvar_filter{};
    std::string cvar_status;
    auto handle_event(KeyPressedEvent ev) -> bool;
    auto handle_event(KeyReleasedEvent ev) -> bool;
    auto handle_event(MouseMovedEvent ev) -> bool;
    auto handle_event(MouseScrolledEvent ev) -> bool;
    auto handle_event(MouseButtonPressedEvent ev) -> bool;
    auto handle_event(MouseButtonReleasedEvent ev) -> bool;
    auto on_play_starting() -> void override;
    auto on_play_stopping() -> void override;
    auto on_scene_opening() -> void override;
    auto on_shutdown() -> void override;
    [[nodiscard]] auto recover_work() -> std::optional<std::filesystem::path> override;
    auto create_ui_resources() -> void override;
    auto on_source_scene_replaced() -> void override;
};
