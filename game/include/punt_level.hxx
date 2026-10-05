#pragma once

#include <string_view>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Scene;
struct Renderer;
struct EngineModels;

// The "Punt!" course: a walled pitch with a shaft cut through the floor at one end, obstacles in the middle and a
// ball to kick down it. Built once in code, then written to assets/scenes/punt.lbf so every later run loads the
// course as data (see PuntGame::on_populate) and the editor can move anything in it.
namespace punt {

    // Entities PuntGame looks up by name after the course loads, so the names are part of the file format.
    inline constexpr std::string_view player_entity_name = "player";
    inline constexpr std::string_view ball_entity_name = "ball";

    // A nameless marker with only a Transform: its position is the centre of the hole's trigger volume and its
    // scale the volume's half extents. The ball is holed once its centre is inside. Drag it in the editor and the
    // target moves with it.
    inline constexpr std::string_view hole_entity_name = "hole";

    inline constexpr float floor_top_y = 0.0F;
    inline constexpr float floor_thickness = 3.0F;

    // Half the playable area, about the origin.
    inline constexpr glm::vec2 course_half_extents{20.0F, 14.0F};

    inline constexpr glm::vec2 hole_centre{14.0F, 0.0F};
    inline constexpr float hole_half_size = 1.5F;

    // The shaft's floor, so a holed ball lands somewhere instead of falling out of the world.
    inline constexpr float pit_floor_y = floor_top_y - floor_thickness;

    inline constexpr float ball_radius = 0.35F;
    inline constexpr float ball_mass = 0.45F;
    inline constexpr glm::vec3 ball_spawn{-11.0F, ball_radius + 0.05F, 0.0F};

    inline constexpr float player_capsule_radius = 0.35F;
    inline constexpr float player_capsule_height = 1.0F;
    inline constexpr glm::vec3 player_spawn{-16.0F, 1.5F, 0.0F};

    // The engine cube and sphere are a unit cube and a unit-diameter sphere, so a half extent is also a scale.
    inline constexpr float primitive_half_extent = 0.5F;

} // namespace punt

// Replaces `scene`'s entities with the course. Render thread, since it creates materials.
auto build_punt_level(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void;
