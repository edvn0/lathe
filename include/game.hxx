#pragma once

#include <glm/mat4x4.hpp>

#include "engine_models.hxx"
#include "input_events.hxx"

class Scene;
struct Renderer;

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

    [[nodiscard]] virtual auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams = 0;
};
