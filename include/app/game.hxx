#pragma once

#include <glm/mat4x4.hpp>

#include <optional>

#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx" // Components::Meta/GeneratedMeta
#include "rendering/scene.hxx" // clone_registry<>()
#include "scene/components.hxx"
#include "scene/input_events.hxx"
#include "terrain/terrain_world.hxx"

class Scene;
struct Renderer;

// entt::snapshot_loader asserts its destination registry is empty on
// construction (see basic_snapshot_loader's ctor in entt/entity/snapshot.hpp),
// so editor_scene -> runtime_scene cloning can only ever run as a single
// snapshot/loader pass -- there's no way to clone the engine's base component
// set now and a game's own extra component types later via a second,
// separate clone_registry<>() call. This helper is that one pass: it always
// includes the engine's own base set, plus whatever ExtraComponents a game
// passes in for its own per-entity data (see IGame::clone_into_runtime).
template<typename... ExtraComponents>
auto clone_editor_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_registry<Components::Transform, Components::Model, Components::RigidBody, Components::MaterialOverride,
                   Components::PlayerTag, Components::Lifetime, Components::PointLight, Components::SpotLight,
                   Components::GeneratedMeta, Components::Meta, ExtraComponents...>(editor_scene.get_registry(),
                                                                                    runtime_scene.get_registry());
}

// View/projection/clip-plane values the engine needs to render a frame --
// decouples Application/main.cxx from whatever camera type a game uses
// internally (PlayerCamera, a third-person orbit camera, ...).
struct CameraParams {
    glm::mat4 view{1.0F};
    glm::mat4 projection{1.0F};
    float near_clip = 0.1F;
    float far_clip = 10000.0F;
    float vertical_fov_radians = 1.0F;
};

// The engine drives this interface; it never constructs or owns a Scene
// itself -- Application hands one to each call below (editor_scene at
// startup/on_populate, whichever of editor_scene/runtime_scene is currently
// active for on_update/on_event). Scene transitions (menu -> level, level ->
// level) don't need a separate engine mechanism: a game clears and
// repopulates the Scene it's handed here from inside its own code, the same
// way build_demo_scene() already does for Ctrl+R reloads today.
class IGame {
public:
    virtual ~IGame() = default;

    // Called once from Application::on_startup(), and again whenever the
    // engine is asked to rebuild the editor scene (Ctrl+R). engine_models
    // holds the built-in primitive models (cube/sphere/capsule/grass_clump)
    // the engine creates once at startup -- handed in rather than requiring
    // the game to call create_engine_models() itself.
    virtual auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void = 0;

    // Called every frame while the engine is playing, against the active
    // (runtime) scene.
    virtual auto on_update(Scene &scene, float delta_time) -> void = 0;

    virtual auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void { (void) scene; (void) event; }
    virtual auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void { (void) scene; (void) event; }
    virtual auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void { (void) scene; (void) event; }
    virtual auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }
    virtual auto on_mouse_button_released(Scene &scene, MouseButtonReleasedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }

    // Optional game HUD -- drawn separately from the engine's own debug/editor
    // ImGui panels (Application::on_ui()).
    virtual auto on_ui() -> void {}

    // Opts into engine-managed streaming terrain (see TerrainWorld): called
    // once from Application::on_startup(), after on_populate(), on the
    // render thread with a live one-time command buffer already open (so
    // implementations needing a MaterialHandle can create one against
    // `renderer` from inside on_populate() and pass it through here). A
    // game with no streaming terrain -- or one that builds its own fixed
    // terrain, as the pre-streaming code did -- leaves this at the default
    // nullopt, and Application never creates a TerrainWorld.
    [[nodiscard]] virtual auto terrain_create_info(Renderer &renderer) -> std::optional<TerrainWorldCreateInfo> {
        (void) renderer;
        return std::nullopt;
    }

    [[nodiscard]] virtual auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams = 0;

    // Called once from Application::play() to populate runtime_scene's
    // registry from editor_scene's -- see clone_editor_into_runtime() above
    // for why this can't be split into an engine-side call plus a separate
    // game-side one. Default clones just the engine's own base component
    // set; a game that adds its own component types via on_populate() (e.g.
    // gameplay/script data) must override this and pass them as
    // ExtraComponents, or they silently won't exist on the entities
    // Application actually simulates/renders once playing -- see
    // BasicGame::clone_into_runtime for Components::Script/CircularMotion.
    virtual auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
        clone_editor_into_runtime<>(editor_scene, runtime_scene);
    }
};
