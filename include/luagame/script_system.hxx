#pragma once

#include <memory>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>

#include "game/game.hxx"
#include "luagame/lua_behaviours.hxx"
#include "luagame/lua_animation.hxx"
#include "physics/mesh_collider.hxx"

class LuaRuntime;

struct LuaGameHost {
    Scene *scene = nullptr;
    Renderer *renderer = nullptr;
    EngineModels const *engine_models = nullptr;

    GameHost host;

    glm::mat4 inverse_view_projection{1.0F};

    // Input a script polls (key.down, mouse.delta); fed by ScriptSystem from the engine's events.
    std::set<int> keys_down;
    double mouse_delta_x = 0.0;
    double mouse_delta_y = 0.0;

    // Mesh colliders of models loaded with { collider = true }, keyed by the model's slot, and the entities that asked
    // to collide with one. A collider is added to each scene's physics world once its triangles are built, so it
    // reaches the runtime scene whenever that is created.
    struct PendingMeshCollider {
        entt::entity entity = entt::null;
        std::shared_ptr<MeshColliderSlot> slot;
        std::uint64_t added_to_world = 0; // PhysicsWorld::id() of the world it is in, 0 for none
    };

    // Skinned characters scripts have asked for (animation library); advanced after each on_update.
    // behaviours (assets/scripts/behaviours); made once the Lua state exists.
    std::unique_ptr<BehaviourSystem> components;

    std::unique_ptr<LuaAnimationSystem> animation = std::make_unique<LuaAnimationSystem>();

    // The headless calls already reported (see headless_note in lua_game_api.cxx).
    std::set<std::string, std::less<>> headless_notes;

    std::unordered_map<std::uint64_t, std::shared_ptr<MeshColliderSlot>> collider_slots;
    std::vector<PendingMeshCollider> mesh_colliders;
};

// The engine's scripting component: runs the entry script (Lua) against a scene, calling the callbacks it defines
// (`on_populate`, `on_update`, `on_key`, `on_ui`, `on_camera`, ...), and gives it the engine's libraries (scene, entity,
// assets, physics, ui, ...). The Runtime owns one next to its IGame, so a script can drive the scene on its own or beside
// a native game. The renderer is optional: without one (a headless run) the libraries that need the GPU report that
// they are unavailable and the rest works.
class ScriptSystem {
public:
    ScriptSystem();
    ~ScriptSystem();

    ScriptSystem(ScriptSystem const &) = delete;
    auto operator=(ScriptSystem const &) -> ScriptSystem & = delete;

    auto add_native_module(std::string name, lua_CFunction opener) -> void;

    auto attach_host(GameHost host) -> void;
    // The scripts the next on_populate runs (data paths); the project's list changed.
    auto set_scripts(std::vector<std::string> scripts) -> void;

    // Loads the entry script into a fresh Lua state and calls its on_populate. `renderer` and `engine_models` may be
    // null.
    auto on_populate(Scene &scene, Renderer *renderer, EngineModels const *engine_models) -> void;
    auto on_update(Scene &scene, float delta_time) -> void;

    auto on_play_stop(Scene &scene) -> void;
    [[nodiscard]] auto behaviour_names() -> std::span<std::string const>;
    [[nodiscard]] auto behaviour_schema(std::string const &name) -> ScriptSchema const *;
    auto reload_behaviours() -> void;
    auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void;

    // Subscribes the script's input handlers; the subscriptions live as long as this object.
    auto subscribe_input(InputContext const &ctx) -> void;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void;

    [[nodiscard]] auto wants_cursor() const -> bool;
    auto on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void;

    auto on_ui(Scene &scene, Renderer &renderer) -> void;

    // True when a script (or a module it requires) changed on disk since the last call: the caller repopulates, which
    // reloads them. Watches the scripts' directories from on_populate until the next one.
    [[nodiscard]] auto poll_changes() -> bool;

    // True once a script is loaded that defines an on_camera (or the older `camera`) callback.
    [[nodiscard]] auto has_camera() const -> bool;
    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams;

    // True when a script is loaded and no error has disabled a callback.
    [[nodiscard]] auto ok() const -> bool;
    // The load error, or the first callback error, empty if none.
    [[nodiscard]] auto error_text() const -> std::string;

private:
    struct Impl;

    template<typename E>
    auto listen(InputContext const &ctx, void (ScriptSystem::*handler)(Scene &, E const &), bool look = false) -> void {
        input_subscriptions_.push_back(ctx.bus.subscribe<E>(
                [this, ctx, handler, look](E const &event) {
                    if (ctx.playing() && (!look || ctx.mouse_look())) {
                        (this->*handler)(*ctx.scene(), event);
                    }

                    return false;
                },
                200));
    }

    std::unique_ptr<Impl> impl_;
    std::vector<EventBus::Subscription> input_subscriptions_;
};
