#pragma once

#include "rendering/imgui_renderer.hxx"
#include "rendering/frame_graph/pass_profiler.hxx"
#include "runtime/runtime.hxx"

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
#include "game/game.hxx"
#include "app/game_packager.hxx"
#include "app/stage_registry.hxx"
#include "assets/material_storage.hxx"
#include "assets/file_change_watcher.hxx"
#include "gpu/context.hxx"
#include "rendering/effect_system.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/particle_system.hxx"
#include "rendering/scene.hxx"
#include "rendering/terminal_widget.hxx"
#include "scene/editor_camera.hxx"
#include "scene/hierarchy_model.hxx"
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

struct Application : Runtime {
    explicit Application(VulkanContext &ctx) noexcept;
    ~Application() override;

    auto on_ui(std::uint32_t frame_index) -> void override;

    std::unique_ptr<gui::EditorIcons> editor_icons;

    enum class ModelBrowseTarget : std::uint8_t { spawn_entity, inspector };
    ModelBrowseTarget model_browse_target = ModelBrowseTarget::spawn_entity;

    bool rename_needs_focus = false;
    bool inspector_name_dirty = false;

    struct TimingSeries {
        std::string id;
        std::string label;
        ScrollingBuffer buffer;
    };
    std::vector<TimingSeries> timing_series;

    auto add_pass_timings(std::span<frame_graph::PassTiming const> passes) -> void;
    auto on_frame_timings(std::span<frame_graph::PassTiming const> passes) -> void override { add_pass_timings(passes); }
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

    std::string hierarchy_search;

    HierarchyModel hierarchy_model;
    Scene const *hierarchy_model_scene = nullptr;
    std::uint64_t hierarchy_model_revision = 0;

    entt::entity hierarchy_context_entity = entt::null;

    stages::Registry stage_registry;
    gui::FrameGraphEditor frame_graph_editor;

    bool unsaved_changes_popup_requested = false;
    bool save_as_popup_requested = false;

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

    auto draw_scene_file_ui() -> void;

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
    [[nodiscard]] auto observer_position() const -> glm::vec3 override { return camera.position(); }

    auto on_event(KeyPressedEvent ev) -> bool override;
    auto on_event(KeyReleasedEvent ev) -> bool override;
    auto on_event(MouseMovedEvent ev) -> bool override;
    auto on_event(MouseScrolledEvent ev) -> bool override;
    auto on_event(MouseButtonPressedEvent ev) -> bool override;
    auto on_event(MouseButtonReleasedEvent ev) -> bool override;

protected:
    auto on_play_starting() -> void override;
    auto on_play_stopping() -> void override;
    auto on_scene_opening() -> void override;
    auto on_shutdown() -> void override;
    auto create_ui_resources() -> void override;
    auto on_source_scene_replaced() -> void override;
};
