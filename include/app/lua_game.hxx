#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>

#include "app/game.hxx"
#include "app/lua_animation.hxx"
#include "physics/mesh_collider.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;

struct LuaGameHost {
    Scene *scene = nullptr;
    Renderer *renderer = nullptr;
    EngineModels const *engine_models = nullptr;

    GameHost host;

    glm::mat4 inverse_view_projection{1.0F};

    // Input a script polls (key.down, mouse.delta); fed by LuaGame from the engine's events.
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
    std::unique_ptr<LuaAnimationSystem> animation = std::make_unique<LuaAnimationSystem>();

    std::unordered_map<std::uint64_t, std::shared_ptr<MeshColliderSlot>> collider_slots;
    std::vector<PendingMeshCollider> mesh_colliders;
};

class LuaGame final : public IGame {
public:
    LuaGame();
    ~LuaGame() override;

    auto add_native_module(std::string name, lua_CFunction opener) -> void;

    auto attach_host(GameHost host) -> void override;

    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    [[nodiscard]] auto wants_cursor() const -> bool override;
    auto on_cursor_position(Scene &scene, CursorPositionEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};
