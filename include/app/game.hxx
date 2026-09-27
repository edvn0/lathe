#pragma once

#include <glm/mat4x4.hpp>

#include <optional>
#include <vector>

#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/scene.hxx"
#include "scene/camera_path.hxx"
#include "scene/components.hxx"
#include "scene/input_events.hxx"
#include "terrain/terrain_world.hxx"

class Scene;
struct Renderer;

// entt's snapshot loader requires an empty destination, so the engine's base components and a game's extras
// have to be cloned in a single pass.
template<typename... ExtraComponents>
auto clone_editor_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_registry<Components::Transform, Components::Model, Components::InstancedModel, Components::RigidBody,
                   Components::MaterialOverride, Components::PlayerTag, Components::Lifetime, Components::PointLight,
                   Components::SpotLight, Components::GeneratedMeta, Components::Meta, Components::Parent,
                   ExtraComponents...>(editor_scene.get_registry(), runtime_scene.get_registry());
}

// Decouples the engine from the game's camera type.
struct CameraParams {
    glm::mat4 view{1.0F};
    glm::mat4 projection{1.0F};
    float near_clip = 0.1F;
    float far_clip = 10000.0F;
    float vertical_fov_radians = 1.0F;
};

// The interface the engine drives. Application hands in the active Scene on each call; games change levels by
// clearing and repopulating it.
class IGame {
public:
    virtual ~IGame() = default;

    // Called at startup and on Ctrl+R to rebuild the editor scene.
    virtual auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void = 0;

    // Called every frame while playing, with the runtime scene.
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

    // Optional game UI, drawn inside the engine's ImGui frame. `scene` is the active scene.
    virtual auto on_ui(Scene &scene, Renderer &renderer) -> void {
        (void) scene;
        (void) renderer;
    }

    // Opt-in streaming terrain. Called once at startup after on_populate(), on the render thread inside a
    // one-time command buffer. nullopt means no TerrainWorld.
    [[nodiscard]] virtual auto terrain_create_info(Renderer &renderer) -> std::optional<TerrainWorldCreateInfo> {
        (void) renderer;
        return std::nullopt;
    }

    // The loop --benchmark flies the editor camera along. Empty means the game has no benchmark and --benchmark
    // fails at startup.
    [[nodiscard]] virtual auto benchmark_camera_path() const -> std::vector<CameraKeyframe> { return {}; }

    [[nodiscard]] virtual auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams = 0;

    // Populates runtime_scene from editor_scene on play(). Games with their own component types must override
    // this and pass them as ExtraComponents, or those components won't exist while playing.
    virtual auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
        clone_editor_into_runtime<>(editor_scene, runtime_scene);
    }
};
