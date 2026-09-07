#pragma once

// ImGuizmo.h expects imgui.h to already be included by the includer rather
// than including it itself -- pull it in first via imgui_renderer.hxx's
// transitive <imgui.h>, which must precede the ImGuizmo.h include below.
#include "rendering/imgui_renderer.hxx"

#include <ImGuizmo.h>

#include "rendering/editor_icons.hxx"

#include <functional> // std::move_only_function

#include "assets/material_storage.hxx" // MaterialCreateInfo
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

// Forward-declared rather than pulling in portable-file-dialogs.h here: the
// dialog is only ever touched from application.cxx, which owns the include.
namespace pfd {
    class open_file;
} // namespace pfd

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
    std::unique_ptr<gui::EditorIcons> editor_icons;
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

    // Created in on_startup() if `game->terrain_create_info()` returns a
    // value; null for games with no streaming terrain. See TerrainWorld.
    std::unique_ptr<TerrainWorld> terrain;

    // Seconds since startup -- forwarded to UBO.time each frame so the
    // wind shader (wind.slang) has something to animate against.
    float elapsed_time = 0.0F;
    static constexpr auto stats_record_start_time = 5.0F;
    [[nodiscard]] constexpr auto can_start_recording_statistics() { return elapsed_time > stats_record_start_time; }


    std::array<ScrollingBuffer, stage_count> timing_buffers;
    float timing_x = 0.0F;

    EditorCamera camera;

    // Editor-only selection driven by the "Hierarchy" widget in on_ui() and
    // manipulated in-viewport via ImGuizmo. Cleared on play()/stop() since
    // it names an entity in whichever registry was active_scene() at
    // selection time, and that registry swaps out across the play/stop
    // boundary (see active_scene()).
    entt::entity selected_entity = entt::null;
    ImGuizmo::OPERATION gizmo_operation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE gizmo_mode = ImGuizmo::WORLD;

    // Persists the Hierarchy widget's search filter across frames (see
    // on_ui()'s "Entities" section).
    std::string hierarchy_search;

    float light_azimuth_degrees = 30.0F;
    float light_elevation_degrees = 55.0F;

    bool mouse_dragging = false;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool has_last_mouse_position = false;

    gui::TerminalWidget terminal_widget;

    // Status line for the "Load Model" widget's last browse attempt (success
    // or failure), shown underneath its Browse... button in on_ui().
    std::string model_load_status;

    // Non-null while a file-picker dialog spawned by the "Load Model" widget
    // is open. Polled non-blockingly (ready(0)) each frame in on_ui() rather
    // than calling result() straight after construction, so browsing for a
    // model doesn't stall the render loop for as long as the dialog is open.
    std::unique_ptr<pfd::open_file> model_load_dialog;

    // Same as model_load_dialog, for the Inspector's per-entity "Change
    // Model" browse control (see the Model section in on_ui()) -- kept
    // separate since both can conceivably be mid-browse at once (the
    // standalone "Load Model" widget and the Inspector are different
    // windows the user can interact with independently).
    std::unique_ptr<pfd::open_file> inspector_model_dialog;

    // In-progress state for the Assets panel's "New Material" popup (see
    // the Materials section in on_ui()) -- persisted across frames like
    // model_load_status/hierarchy_search above rather than reset to a local
    // the moment the popup is open, so edits made across several frames
    // (dragging a slider, typing a name) aren't lost between them.
    MaterialCreateInfo new_material_info{};
    std::string new_material_name;

    // A "Delete" click in the Assets panel doesn't destroy the underlying
    // asset immediately -- it queues `commit` to run once elapsed_time
    // reaches `delete_at`, deferred by deletion_grace_seconds (see on_ui()'s
    // Materials section). This gives the Assets panel an undo window (the
    // "Recently deleted" list restores an entry by just erasing its queued
    // PendingDeletion, since nothing was actually touched yet) and, unlike
    // Application's other undo-less destructive actions (Hierarchy's
    // Delete), matters more here because a wrongly-deleted named material
    // can be referenced by an arbitrary number of entities' MaterialOverride
    // that have no record of what handle they used to point at.
    //
    // Kept asset-kind-agnostic (a label + a type-erased commit callback)
    // rather than typed to MaterialHandle specifically, so Models/Textures
    // could feed the same queue later instead of each growing its own copy
    // of this grace-period/restore machinery.
    struct PendingDeletion {
        std::string label;
        float delete_at = 0.0F;
        std::move_only_function<void()> commit;
    };
    static constexpr float deletion_grace_seconds = 20.0F;
    std::vector<PendingDeletion> pending_deletions;

    auto update(float delta_time) -> void;

    auto on_startup() -> void;

    auto on_event(KeyPressedEvent ev) -> bool;
    auto on_event(KeyReleasedEvent ev) -> bool;
    auto on_event(MouseMovedEvent ev) -> bool;
    auto on_event(MouseScrolledEvent ev) -> bool;
    auto on_event(MouseButtonPressedEvent ev) -> bool;
    auto on_event(MouseButtonReleasedEvent ev) -> bool;
};
