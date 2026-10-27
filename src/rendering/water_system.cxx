#include "rendering/water_system.hxx"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <tracy/Tracy.hpp>

#include "assets/primitive_meshes.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/water_waves.hxx"

namespace {

    // The world transform without its scale: a water body is sized by its extent, not by the entity's scale.
    struct Placement {
        glm::vec3 position{0.0F};
        // The body's local +x and +z in the world XZ plane.
        glm::vec2 axis_x{1.0F, 0.0F};
        glm::vec2 axis_z{0.0F, 1.0F};
    };

    auto placement_of(glm::mat4 const &world) -> Placement {
        auto const x = glm::vec2{world[0].x, world[0].z};
        auto const length = glm::length(x);
        auto const axis_x = length > 1e-6F ? x / length : glm::vec2{1.0F, 0.0F};
        // Yaw only: +z is +x turned a quarter of the way round (the engine is left-handed, y up).
        return Placement{
                .position = glm::vec3{world[3]},
                .axis_x = axis_x,
                .axis_z = glm::vec2{-axis_x.y, axis_x.x},
        };
    }

    // The ocean's water zone in world XZ: min x, min z, max x, max z.
    auto ocean_zone(Components::Water const &water, Placement const &placement) -> glm::vec4 {
        auto const half = glm::max(water.extent, glm::vec2{0.0F}) * 0.5F;
        return glm::vec4{placement.position.x - half.x, placement.position.z - half.y, placement.position.x + half.x,
                         placement.position.z + half.y};
    }

    // Whether `xz` is on the body, and the point it is evaluated at.
    auto covers(Components::Water const &water, Placement const &placement, glm::vec2 xz) -> bool {
        if (water.body == Components::WaterBody::ocean) {
            auto const zone = ocean_zone(water, placement);
            return xz.x >= zone.x && xz.y >= zone.y && xz.x <= zone.z && xz.y <= zone.w;
        }

        auto const local = xz - glm::vec2{placement.position.x, placement.position.z};
        auto const half = water.extent * 0.5F;
        return std::abs(glm::dot(local, placement.axis_x)) <= half.x && std::abs(glm::dot(local, placement.axis_z)) <= half.y;
    }

    auto field_at(Components::Water const &water, glm::vec2 xz, float time) -> water_waves::Sample {
        return water_waves::field(xz, time, water.wave_scale, water.wave_speed, water_waves::vertex_footprint_minimum);
    }

    // Visits every visible body with its world transform.
    template<typename Visit>
    auto for_each_body(entt::registry const &registry, Visit &&visit) -> void {
        auto const view = registry.view<Components::Transform const, Components::Water const>();
        for (auto [entity, transform, water]: view.each()) {
            if (water.visible) {
                visit(water, systems::get_world_transform(registry, entity, transform));
            }
        }
    }

    // The body whose surface is at `xz`: a lake over an ocean, and the highest lake.
    template<typename Result>
    auto query(entt::registry const &registry, glm::vec2 xz, float time,
               std::optional<Result> (*evaluate)(Components::Water const &, glm::mat4 const &, glm::vec2, float))
            -> std::optional<Result> {
        auto best = std::optional<Result>{};
        auto best_height = -std::numeric_limits<float>::infinity();
        auto best_is_lake = false;

        for_each_body(registry, [&](Components::Water const &water, glm::mat4 const &world) {
            auto const height = WaterSystem::surface_height(water, world, xz, time);
            if (!height) {
                return;
            }

            auto const lake = water.body == Components::WaterBody::lake;
            if (best && ((best_is_lake && !lake) || (lake == best_is_lake && *height <= best_height))) {
                return;
            }

            if (auto value = evaluate(water, world, xz, time)) {
                best = *value;
                best_height = *height;
                best_is_lake = lake;
            }
        });

        return best;
    }

}

auto WaterSystem::material_info(Renderer &renderer, Components::Water const &water, glm::mat4 const &world)
        -> MaterialCreateInfo {
    auto const &images = renderer.image_storage();
    auto const zone = water.body == Components::WaterBody::ocean ? ocean_zone(water, placement_of(world))
                                                                  : WaterSettings{}.zone;

    return MaterialCreateInfo{
            .base_colour_factor = glm::vec4{water.colour, 1.0F},
            .metallic_factor = 0.0F,
            .roughness_factor = water.roughness,
            .base_colour_texture = images.white(),
            .normal_texture = images.flat_normal(),
            .metallic_roughness_texture = images.white(),
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = renderer.sampler_storage().linear_repeat(),
            .alpha_mode = AlphaMode::blend,
            .max_shadow_cascade = GpuMaterial::no_shadow_cascade,
            .water = true,
            .water_settings =
                    WaterSettings{
                            .absorption = water.absorption,
                            .shore_fade = water.shore_fade,
                            .wave_scale = water.wave_scale,
                            .wave_speed = water.wave_speed,
                            .normal_strength = water.normal_strength,
                            .refraction_strength = water.refraction_strength,
                            .foam = water.foam,
                            .caustics = water.caustics,
                            .wave_height = water.wave_height,
                            .whitecaps = water.whitecaps,
                            .zone = zone,
                    },
    };
}

auto WaterSystem::lake_cells(glm::vec2 extent) noexcept -> glm::uvec2 {
    auto const cells = [](float size) {
        auto const wanted = std::isfinite(size) ? std::round(std::max(size, 0.0F) / lake_spacing) : 0.0F;
        return static_cast<std::uint32_t>(
                std::clamp(wanted, static_cast<float>(lake_min_cells), static_cast<float>(lake_max_cells)));
    };
    return glm::uvec2{cells(extent.x), cells(extent.y)};
}

auto WaterSystem::ocean_centre(glm::vec2 camera_xz) noexcept -> glm::vec2 {
    return glm::floor(camera_xz / ocean_snap + 0.5F) * ocean_snap;
}

auto WaterSystem::surface_height(Components::Water const &water, glm::mat4 const &world, glm::vec2 xz, float time)
        -> std::optional<float> {
    auto const placement = placement_of(world);
    if (!covers(water, placement, xz)) {
        return std::nullopt;
    }
    return placement.position.y + field_at(water, xz, time).offset * water.wave_height;
}

auto WaterSystem::surface_normal(Components::Water const &water, glm::mat4 const &world, glm::vec2 xz, float time)
        -> std::optional<glm::vec3> {
    if (!covers(water, placement_of(world), xz)) {
        return std::nullopt;
    }
    auto const slope = field_at(water, xz, time).offset_gradient * water.wave_height;
    return glm::normalize(glm::vec3{-slope.x, 1.0F, -slope.y});
}

auto WaterSystem::water_height(entt::registry const &registry, glm::vec2 xz, float time) -> std::optional<float> {
    return query<float>(registry, xz, time, &WaterSystem::surface_height);
}

auto WaterSystem::water_normal(entt::registry const &registry, glm::vec2 xz, float time) -> std::optional<glm::vec3> {
    return query<glm::vec3>(registry, xz, time, &WaterSystem::surface_normal);
}

auto WaterSystem::is_underwater(entt::registry const &registry, glm::vec3 point, float time) -> bool {
    auto const height = water_height(registry, glm::vec2{point.x, point.z}, time);
    return height.has_value() && point.y < *height;
}

auto WaterSystem::retire(Body &body) -> void {
    if (body.material.valid() || body.model.valid()) {
        retired_.push_back(Retired{.material = body.material, .model = body.model});
    }
    body.material = {};
    body.model = {};
}

auto WaterSystem::age_retired(Renderer &renderer, bool everything) -> void {
    std::erase_if(retired_, [&](Retired &retired) {
        if (!everything && retired.frames_left-- > 0) {
            return false;
        }
        renderer.release_material(retired.material);
        if (retired.model.valid()) {
            renderer.release_model(retired.model);
        }
        return true;
    });
}

auto WaterSystem::ocean_model(Renderer &renderer) -> ModelHandle {
    if (ocean_model_.valid()) {
        return ocean_model_;
    }

    auto mesh = make_water_grid_mesh(GridAxis{.cells = ocean_cells, .curve = ocean_curve, .focus = 0.5F},
                                     GridAxis{.cells = ocean_cells, .curve = ocean_curve, .focus = 0.5F});
    if (!mesh) {
        error("[Water] Could not build the ocean grid");
        return {};
    }
    auto model = renderer.create_model_from_cpu_data(to_model_cpu_data(std::move(*mesh)));
    if (!model) {
        error("[Water] Could not create the ocean grid: {}", describe(model.error()));
        return {};
    }
    ocean_model_ = *model;
    return ocean_model_;
}

auto WaterSystem::submit(Renderer &renderer, entt::registry const &registry, glm::vec3 camera_position) -> void {
    ZoneScopedNC("Water bodies", tracy::Color::DeepSkyBlue);

    age_retired(renderer, false);

    for (auto &body: bodies_ | std::views::values) {
        body.seen = false;
    }

    auto const view = registry.view<Components::Transform const, Components::Water const>();
    for (auto [entity, transform, water]: view.each()) {
        auto &body = bodies_[entity];
        body.seen = true;

        auto const world = systems::get_world_transform(registry, entity, transform);
        auto const placement = placement_of(world);
        auto const info = material_info(renderer, water, world);

        if (!body.material.valid()) {
            auto material = renderer.create_material(info);
            if (!material) {
                error("[Water] Could not create a water material: {}", describe(material.error()));
                continue;
            }
            body.material = *material;
            body.material_info = info;
        } else if (!(body.material_info.water_settings == info.water_settings) ||
                   body.material_info.base_colour_factor != info.base_colour_factor ||
                   body.material_info.roughness_factor != info.roughness_factor) {
            if (auto const updated = renderer.update_material(body.material, info); !updated) {
                error("[Water] Could not update a water material: {}", describe(updated.error()));
            }
            body.material_info = info;
        }
        body.water = water;

        auto model = ModelHandle{};
        auto model_transform = glm::mat4{1.0F};

        if (water.body == Components::WaterBody::ocean) {
            if (body.model.valid()) {
                // It was a lake: its own grid goes, the ocean's is shared.
                retired_.push_back(Retired{.model = body.model});
                body.model = {};
                body.cells = glm::uvec2{0U};
            }
            model = ocean_model(renderer);
            auto const centre = ocean_centre(glm::vec2{camera_position.x, camera_position.z});
            model_transform = glm::scale(glm::translate(glm::mat4{1.0F}, {centre.x, placement.position.y, centre.y}),
                                         glm::vec3{ocean_size, 1.0F, ocean_size});
        } else {
            auto const cells = lake_cells(water.extent);
            if (!body.model.valid() || body.cells != cells) {
                if (body.model.valid()) {
                    retired_.push_back(Retired{.model = body.model});
                    body.model = {};
                }
                auto mesh = make_water_grid_mesh(GridAxis{.cells = cells.x}, GridAxis{.cells = cells.y});
                if (mesh) {
                    if (auto created = renderer.create_model_from_cpu_data(to_model_cpu_data(std::move(*mesh)))) {
                        body.model = *created;
                        body.cells = cells;
                    } else {
                        error("[Water] Could not create a lake grid: {}", describe(created.error()));
                    }
                }
            }
            model = body.model;

            auto rotation = glm::mat4{1.0F};
            rotation[0] = glm::vec4{placement.axis_x.x, 0.0F, placement.axis_x.y, 0.0F};
            rotation[2] = glm::vec4{placement.axis_z.x, 0.0F, placement.axis_z.y, 0.0F};
            model_transform = glm::translate(glm::mat4{1.0F}, placement.position) * rotation *
                              glm::scale(glm::mat4{1.0F}, glm::vec3{water.extent.x, 1.0F, water.extent.y});
        }

        if (!water.visible || !model.valid() || !body.material.valid()) {
            continue;
        }

        if (auto const submitted = renderer.submit_model(model, model_transform, body.material); !submitted) {
            error("[Water] Could not submit a water body: {}", describe(submitted.error()));
        }
    }

    for (auto it = bodies_.begin(); it != bodies_.end();) {
        if (!it->second.seen) {
            retire(it->second);
            it = bodies_.erase(it);
        } else {
            ++it;
        }
    }
}

auto WaterSystem::clear(Renderer &renderer) -> void {
    for (auto &body: bodies_ | std::views::values) {
        retire(body);
    }
    bodies_.clear();
    age_retired(renderer, true);
    if (ocean_model_.valid()) {
        renderer.release_model(ocean_model_);
        ocean_model_ = {};
    }
}
