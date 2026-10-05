#include "punt_level.hxx"

#include <array>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>

#include "assets/material_storage.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    using namespace punt;

    // An axis-aligned slab given by its extent along each axis, which is how the course is easiest to describe.
    struct Slab {
        glm::vec3 min;
        glm::vec3 max;

        [[nodiscard]] auto centre() const -> glm::vec3 { return (min + max) * 0.5F; }
        [[nodiscard]] auto half_extents() const -> glm::vec3 { return (max - min) * 0.5F; }
    };

    [[nodiscard]] auto floor_slabs() -> std::array<Slab, 4> {
        auto const bottom = pit_floor_y;
        auto const top = floor_top_y;

        auto const west_edge = hole_centre.x - hole_half_size;
        auto const east_edge = hole_centre.x + hole_half_size;
        auto const south_edge = hole_centre.y - hole_half_size;
        auto const north_edge = hole_centre.y + hole_half_size;

        return {
                // Everything before the shaft, full depth.
                Slab{{-course_half_extents.x, bottom, -course_half_extents.y}, {west_edge, top, course_half_extents.y}},
                // Everything after it, full depth.
                Slab{{east_edge, bottom, -course_half_extents.y}, {course_half_extents.x, top, course_half_extents.y}},
                // The two strips either side of the shaft.
                Slab{{west_edge, bottom, -course_half_extents.y}, {east_edge, top, south_edge}},
                Slab{{west_edge, bottom, north_edge}, {east_edge, top, course_half_extents.y}},
        };
    }

    [[nodiscard]] auto perimeter_walls() -> std::array<Slab, 4> {
        constexpr float height = 1.6F;
        constexpr float thickness = 0.5F;

        auto const x = course_half_extents.x;
        auto const z = course_half_extents.y;
        auto const top = floor_top_y + height;

        return {
                Slab{{-x - thickness, floor_top_y, z}, {x + thickness, top, z + thickness}},
                Slab{{-x - thickness, floor_top_y, -z - thickness}, {x + thickness, top, -z}},
                Slab{{x, floor_top_y, -z - thickness}, {x + thickness, top, z + thickness}},
                Slab{{-x - thickness, floor_top_y, -z - thickness}, {-x, top, z + thickness}},
        };
    }

} // namespace

auto build_punt_level(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    scene.get_registry().clear();

    scene.physics_settings = PhysicsWorldSettings{};
    scene.environment = new_scene_environment();
    scene.environment.sun.elevation_degrees = 48.0F;
    scene.environment.sun.azimuth_degrees = 200.0F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    // One reference each, released at the end; by then the entities below hold their own.
    std::vector<MaterialHandle> created_materials;

    auto const flat_material = [&](std::string name, glm::vec3 const &colour, float roughness) -> MaterialHandle {
        auto material = renderer.create_material(
                MaterialCreateInfo{
                        .base_colour_factor = glm::vec4{colour, 1.0F},
                        .metallic_factor = 0.0F,
                        .roughness_factor = roughness,
                        .base_colour_texture = images.white(),
                        .normal_texture = images.flat_normal(),
                        .metallic_roughness_texture = images.metallic_roughness(),
                        .occlusion_texture = images.occlusion(),
                        .emissive_texture = images.emissive(),
                        .sampler = samplers.linear_repeat(),
                },
                std::move(name));

        if (!material) {
            error("[build_punt_level] Could not create a material: {}", describe(material.error()));
            return MaterialHandle{};
        }

        created_materials.push_back(*material);
        return *material;
    };

    auto const turf = flat_material("punt.turf", glm::vec3{0.22F, 0.46F, 0.19F}, 0.95F);
    auto const wall = flat_material("punt.wall", glm::vec3{0.62F, 0.61F, 0.58F}, 0.8F);
    auto const bumper = flat_material("punt.bumper", glm::vec3{0.85F, 0.42F, 0.12F}, 0.6F);
    auto const pit = flat_material("punt.pit", glm::vec3{0.06F, 0.06F, 0.07F}, 1.0F);
    auto const ball_look = flat_material("punt.ball", glm::vec3{0.95F, 0.95F, 0.92F}, 0.35F);
    auto const player_look = flat_material("punt.player", glm::vec3{0.18F, 0.42F, 0.78F}, 0.5F);
    auto const flag_look = flat_material("punt.flag", glm::vec3{0.85F, 0.1F, 0.12F}, 0.7F);
    auto const pole_look = flat_material("punt.pole", glm::vec3{0.15F, 0.15F, 0.17F}, 0.4F);

    // Static scenery. `rotation` turns the collider with the mesh, so angled baffles collide as they look.
    auto const add_static_box = [&](std::string_view name, glm::vec3 const &centre, glm::vec3 const &half_extents,
                                    MaterialHandle material,
                                    glm::quat const &rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F}) {
        auto const entity = GeneratedEntity{&scene, "{}", name};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = centre,
                .rotation = rotation,
                .scale = half_extents / primitive_half_extent,
        });
        entity.emplace<Components::Model>(Components::Model{.model = engine_models.cube});
        entity.emplace<Components::RigidBody>(Components::RigidBody{
                .half_extents = half_extents,
                .is_static = true,
        });

        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
    };

    for (auto &&[index, slab]: floor_slabs() | std::views::enumerate) {
        add_static_box(std::format("floor_{}", index), slab.centre(), slab.half_extents(), turf);
    }

    for (auto &&[index, slab]: perimeter_walls() | std::views::enumerate) {
        add_static_box(std::format("wall_{}", index), slab.centre(), slab.half_extents(), wall);
    }

    // The shaft's floor, half a metre thick so the ball can't tunnel through it at speed.
    add_static_box("pit_floor", glm::vec3{hole_centre.x, pit_floor_y - 0.25F, hole_centre.y},
                   glm::vec3{hole_half_size, 0.25F, hole_half_size}, pit);

    // Obstacles between the tee and the hole.
    struct Obstacle {
        glm::vec3 centre;
        glm::vec3 half_extents;
        float yaw_degrees;
    };

    constexpr std::array<Obstacle, 7> obstacles{{
            {.centre = {-4.0F, 0.6F, 5.5F}, .half_extents = {1.8F, 0.6F, 0.5F}, .yaw_degrees = 0.0F},
            {.centre = {-4.0F, 0.6F, -5.5F}, .half_extents = {1.8F, 0.6F, 0.5F}, .yaw_degrees = 0.0F},
            {.centre = {1.5F, 0.9F, 0.0F}, .half_extents = {0.5F, 0.9F, 3.2F}, .yaw_degrees = 0.0F},
            {.centre = {6.5F, 0.5F, 8.0F}, .half_extents = {2.6F, 0.5F, 0.5F}, .yaw_degrees = 0.0F},
            {.centre = {6.5F, 0.5F, -8.0F}, .half_extents = {2.6F, 0.5F, 0.5F}, .yaw_degrees = 0.0F},
            // Baffles that funnel a well-aimed ball towards the shaft.
            {.centre = {10.5F, 0.5F, 3.6F}, .half_extents = {2.6F, 0.5F, 0.3F}, .yaw_degrees = 25.0F},
            {.centre = {10.5F, 0.5F, -3.6F}, .half_extents = {2.6F, 0.5F, 0.3F}, .yaw_degrees = -25.0F},
    }};

    for (auto &&[index, obstacle]: obstacles | std::views::enumerate) {
        add_static_box(std::format("obstacle_{}", index), obstacle.centre, obstacle.half_extents, bumper,
                       glm::angleAxis(glm::radians(obstacle.yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}));
    }

    // A flag by the hole, so it reads as a target from across the course.
    constexpr float pole_height = 2.6F;
    auto const pole_base = glm::vec3{hole_centre.x, floor_top_y, hole_centre.y + hole_half_size + 1.1F};

    add_static_box("flag_pole", pole_base + glm::vec3{0.0F, pole_height * 0.5F, 0.0F},
                   glm::vec3{0.06F, pole_height * 0.5F, 0.06F}, pole_look);
    add_static_box("flag", pole_base + glm::vec3{0.5F, pole_height - 0.3F, 0.0F}, glm::vec3{0.5F, 0.3F, 0.03F},
                   flag_look);

    {
        auto const light = Entity{&scene, "hole_light"};
        light.emplace<Components::Transform>(
                Components::Transform{.position = glm::vec3{hole_centre.x, floor_top_y + 3.5F, hole_centre.y}});
        light.emplace<Components::PointLight>(Components::PointLight{
                .colour = glm::vec3{1.0F, 0.85F, 0.55F},
                .intensity = 35.0F,
                .range = 14.0F,
        });
    }

    {
        auto const ball = Entity{&scene, ball_entity_name};
        ball.emplace<Components::Transform>(Components::Transform{
                .position = ball_spawn,
                .scale = glm::vec3{ball_radius / primitive_half_extent},
        });
        ball.emplace<Components::Model>(Components::Model{.model = engine_models.sphere});
        ball.emplace<Components::RigidBody>(
                Components::RigidBody::make_sphere(ball_radius, ball_mass, /*restitution=*/0.45F));
        ball.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = ball_look});
    }

    {
        // The capsule mesh is 1 m tall with a 0.5 m radius; scale it onto the collider's dimensions.
        constexpr glm::vec3 capsule_scale{player_capsule_radius / primitive_half_extent, player_capsule_height,
                                          player_capsule_radius / primitive_half_extent};

        auto const player = Entity{&scene, player_entity_name};
        player.emplace<Components::Transform>(Components::Transform{.position = player_spawn, .scale = capsule_scale});
        player.emplace<Components::Model>(Components::Model{.model = engine_models.capsule});
        player.emplace<Components::RigidBody>(
                Components::RigidBody::make_capsule(player_capsule_radius, player_capsule_height, /*mass=*/80.0F));
        player.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = player_look});
        player.emplace<Components::PlayerTag>();
    }

    {
        // Invisible: the shaft is already there to see. See punt::hole_entity_name.
        auto const hole = Entity{&scene, hole_entity_name};
        hole.emplace<Components::Transform>(Components::Transform{
                .position = glm::vec3{hole_centre.x, (floor_top_y + pit_floor_y) * 0.5F, hole_centre.y},
                .scale = glm::vec3{hole_half_size, (floor_top_y - pit_floor_y) * 0.5F, hole_half_size},
        });
    }

    for (auto const material: created_materials) {
        renderer.release_material(material);
    }

    info("[build_punt_level] Built the Punt! course");
}
