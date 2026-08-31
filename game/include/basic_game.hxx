#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "game.hxx"
#include "material.hxx"
#include "model.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"

// Sample content exercising the engine's rendering and physics systems end
// to end: a player capsule, a scattering of instanced helmet models, a
// physics-cube grid, a textured floor, a few point/spot lights, a field of
// wind-swaying grass clumps, and left-click bullet shooting.
class BasicGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    auto shoot_bullet(Scene &scene, std::size_t n = 1) -> void;

    entt::entity player_entity_{entt::null};
    PlayerController player_controller_;
    PlayerCamera player_camera_;

    // Set by on_populate() -- reused by shoot_bullet() so projectiles are
    // built from the same model/collision-shape relationship as the grid
    // cubes and floor.
    ModelHandle cube_model_{};
    glm::vec3 cube_half_extents_{0.5F};

    ModelHandle house_model_{};
    ModelHandle tree_model_{};

    // Wind-swaying grass material -- created once in on_populate() and
    // shared as a MaterialOverride across every grass clump entity.
    MaterialHandle grass_material_{};
};
