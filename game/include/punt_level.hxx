#pragma once

#include <string_view>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Scene;
struct Renderer;
struct EngineModels;

namespace punt {

    inline constexpr std::string_view player_entity_name = "player";
    inline constexpr std::string_view ball_entity_name = "ball";

    inline constexpr std::string_view hole_entity_name = "hole";

    inline constexpr float floor_top_y = 0.0F;
    inline constexpr float floor_thickness = 3.0F;

    inline constexpr glm::vec2 course_half_extents{20.0F, 14.0F};

    inline constexpr glm::vec2 hole_centre{14.0F, 0.0F};
    inline constexpr float hole_half_size = 1.5F;

    inline constexpr float pit_floor_y = floor_top_y - floor_thickness;

    inline constexpr float ball_radius = 0.35F;
    inline constexpr float ball_mass = 0.45F;
    inline constexpr glm::vec3 ball_spawn{-11.0F, ball_radius + 0.05F, 0.0F};

    inline constexpr float player_capsule_radius = 0.35F;
    inline constexpr float player_capsule_height = 1.0F;
    inline constexpr glm::vec3 player_spawn{-16.0F, 1.5F, 0.0F};

    inline constexpr float primitive_half_extent = 0.5F;

}

auto build_punt_level(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void;
