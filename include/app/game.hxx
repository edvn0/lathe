#pragma once

#include <glm/mat4x4.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/game_graph.hxx"
#include "rendering/scene.hxx"
#include "scene/camera_path.hxx"
#include "scene/components.hxx"
#include "scene/input_events.hxx"
#include "terrain/terrain_world.hxx"

class Scene;
struct Renderer;

template<typename... ExtraComponents>
auto clone_editor_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_registry<Components::Transform, Components::Model, Components::InstancedModel, Components::RigidBody,
                   Components::MaterialOverride, Components::PlayerTag, Components::Lifetime, Components::PointLight,
                   Components::SpotLight, Components::GeneratedMeta, Components::Meta, Components::Parent,
                   ExtraComponents...>(editor_scene.get_registry(), runtime_scene.get_registry());
}

struct CameraParams {
    glm::mat4 view{1.0F};
    glm::mat4 projection{1.0F};
    float near_clip = 0.1F;
    float far_clip = 10000.0F;
    float vertical_fov_radians = 1.0F;
};

struct GameHost {
    bool player_mode = false;

    std::function<void()> request_exit;

    std::string script_entry;
};

class IGame {
public:
    virtual ~IGame() = default;

    virtual auto attach_host(GameHost host) -> void { (void) host; }

    virtual auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void = 0;

    virtual auto on_update(Scene &scene, float delta_time) -> void = 0;

    virtual auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }
    virtual auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }
    virtual auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }
    virtual auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }
    virtual auto on_mouse_button_released(Scene &scene, MouseButtonReleasedEvent const &event) -> void {
        (void) scene;
        (void) event;
    }

    [[nodiscard]] virtual auto wants_cursor() const -> bool { return false; }

    virtual auto on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void {
        (void) scene;
        (void) event;
    }

    virtual auto on_ui(Scene &scene, Renderer &renderer) -> void {
        (void) scene;
        (void) renderer;
    }

    // Called while the renderer declares each frame, once per GameSlot (see rendering/game_graph.hxx). The place to
    // add compute passes and scene draws; GPU resources come from Renderer::game_gpu().
    virtual auto on_frame_graph(GameGraph &graph, float delta_time) -> void {
        (void) graph;
        (void) delta_time;
    }

    [[nodiscard]] virtual auto terrain_create_info(Renderer &renderer) -> std::optional<TerrainWorldCreateInfo> {
        (void) renderer;
        return std::nullopt;
    }

    [[nodiscard]] virtual auto benchmark_camera_path() const -> std::vector<CameraKeyframe> { return {}; }

    [[nodiscard]] virtual auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams = 0;

    virtual auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
        clone_editor_into_runtime<>(editor_scene, runtime_scene);
    }
};
