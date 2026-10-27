#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <functional>
#include <optional>
#include <span>
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
class EffectSystem;
class TextRenderer;
struct Renderer;

template<typename... ExtraComponents>
auto clone_editor_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_registry<Components::Transform, Components::Model, Components::InstancedModel, Components::RigidBody,
                   Components::MaterialOverride, Components::PlayerTag, Components::Lifetime, Components::PointLight,
                   Components::SpotLight, Components::Decal, Components::ParticleEmitter, Components::Water, Components::Planet, Components::Text, Components::GeneratedMeta, Components::Meta,
                   Components::Parent, Components::Static, ExtraComponents...>(editor_scene.get_registry(), runtime_scene.get_registry());
}

struct CameraParams {
    glm::mat4 view{1.0F};
    glm::mat4 projection{1.0F};
    float near_clip = 0.1F;
    float far_clip = 10000.0F;
    float vertical_fov_radians = 1.0F;
    // A floating origin: the world point (in double precision) that `view` treats as its origin. Systems that place
    // content too far out for floats (PlanetSystem) draw relative to it; everything else assumes it is zero.
    glm::dvec3 origin{0.0};
};

struct GameHost {
    bool player_mode = false;

    std::function<void()> request_exit;

    std::string script_entry;

    // What scripts attach compute effects through (rendering/effect_system.hxx); null where there are none.
    EffectSystem *effects = nullptr;

    // What scripts draw text through (rendering/text_renderer.hxx); null where there is none.
    TextRenderer *text = nullptr;
};

// What a game needs to subscribe to the runtime's EventBus: handlers only fire while playing, and look events (mouse
// movement) only while the mouse is captured.
struct InputContext {
    EventBus &bus;
    std::function<Scene *()> scene;
    std::function<bool()> playing;
    std::function<bool()> mouse_look;
};

class IGame {
public:
    virtual ~IGame() = default;

protected:
    template<typename E, typename G>
    auto listen(InputContext const &ctx, void (G::*handler)(Scene &, E const &), bool look = false) -> void {
        input_subscriptions_.push_back(ctx.bus.subscribe<E>(
            [this, ctx, handler, look](E const &event) {
                if (ctx.playing() && (!look || ctx.mouse_look())) {
                    (static_cast<G *>(this)->*handler)(*ctx.scene(), event);
                }

                return false;
            },
            200));
    }

public:

    virtual auto attach_host(GameHost host) -> void { (void) host; }

    virtual auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void = 0;

    virtual auto on_update(Scene &scene, float delta_time) -> void = 0;

    // Games subscribe their input handlers here with listen(); the subscriptions live as long as the game.
    virtual auto subscribe_input(InputContext const &ctx) -> void { (void) ctx; }

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

    // Called when the game stops playing, before the runtime scene is destroyed.
    virtual auto on_play_stop(Scene &scene) -> void { (void) scene; }

    // behaviours the editor can attach to entities (Components::Behaviour), with their declared properties.
    [[nodiscard]] virtual auto behaviour_names() -> std::span<std::string const> { return {}; }
    [[nodiscard]] virtual auto behaviour_schema(std::string const &name) -> ScriptSchema const * {
        (void) name;
        return nullptr;
    }
    virtual auto reload_behaviours() -> void {}

    virtual auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
        clone_editor_into_runtime<>(editor_scene, runtime_scene);
    }

private:
    std::vector<EventBus::Subscription> input_subscriptions_;
};
