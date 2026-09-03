#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"
#include "assets/material.hxx"
#include "assets/model.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"
#include "terrain/terrain_mesh.hxx"

// Sample content exercising the engine's rendering and physics systems end
// to end: a player capsule exploring a small village of procedurally built
// houses and trees, a textured floor, a few point/spot lights, patches of
// wind-swaying grass, and left-click bullet shooting. The houses/trees are
// composed from the engine's box/sphere primitives (no external assets),
// which gives GTAO plenty of corners, overhangs, and self-shadowing to work
// with.
class BasicGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

    [[nodiscard]] auto terrain_create_info(Renderer &renderer) -> std::optional<TerrainWorldCreateInfo> override;

    auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void override;

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

    // Wind-swaying grass material -- created once in on_populate() and
    // shared as a MaterialOverride across every grass clump entity.
    MaterialHandle grass_material_{};

    // Set by on_populate() -- reused for the rest of that call so
    // houses/trees/grass can sample the same noise field the streaming
    // terrain (see terrain_create_info()) generates and rest on the actual
    // generated surface. height_range_min/max are set here too, so
    // TerrainWorld's chunks agree with these placement calls about where
    // local Y = 0 sits (see TerrainMeshResult::mid_height / make_terrain_chunk).
    TerrainParams terrain_params_{};

    // Terrain material, created once in on_populate() (needs the texture
    // streamer/image storage, both only reachable via Renderer there) and
    // handed to TerrainWorld via terrain_create_info(), which runs later
    // with no Scene access of its own.
    MaterialHandle terrain_material_{};
    float terrain_ground_y_ = 0.0F;
};
