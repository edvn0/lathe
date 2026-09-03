#include "basic_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <string>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "scene/components.hxx"
#include "rendering/entity.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "physics/physics_world.hxx"
#include "assets/primitive_meshes.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "terrain/terrain_mesh.hxx"

namespace {

    auto direction_to_rotation(glm::vec3 const &direction) -> glm::quat {
        constexpr auto local_forward = glm::vec3{0.0F, -1.0F, 0.0F};
        auto const dot = glm::dot(local_forward, direction);

        if (dot > 0.9999F) {
            return glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        }
        if (dot < -0.9999F) {
            return glm::angleAxis(glm::pi<float>(), glm::vec3{1.0F, 0.0F, 0.0F});
        }

        auto const axis = glm::normalize(glm::cross(local_forward, direction));
        return glm::angleAxis(std::acos(dot), axis);
    }

} // namespace

auto BasicGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    if (auto could_wait = renderer.wait_idle(); !could_wait.has_value()) {
        info("{}", describe(could_wait.error()));
        return;
    }

    scene.get_registry().clear();

    auto const load_or_fallback = [&renderer, &default_model = engine_models.cube, s = &scene](
                                          std::filesystem::path const &path, entt::entity parent_entity = entt::null,
                                          glm::mat4 const &instance_transform = glm::mat4{1.0F}) -> ModelHandle {
        auto model = renderer.load_model(path);

        if (model) {
            auto const rotation_scale = glm::mat3{instance_transform};

            for (auto &&[index, light]: renderer.model_lights(model.value()) | std::views::enumerate) {
                auto const light_entity = GeneratedEntity{s, "model_light_{}", static_cast<std::uint32_t>(index)};

                if (parent_entity != entt::null) {
                    light_entity.emplace<Components::Parent>(Components::Parent{.entity = parent_entity});
                    light_entity.emplace<Components::Transform>(Components::Transform{
                            .position = light.position,
                            .rotation = direction_to_rotation(light.direction),
                    });
                } else {
                    auto const world_position = glm::vec3{instance_transform * glm::vec4{light.position, 1.0F}};
                    auto const world_direction = glm::normalize(rotation_scale * light.direction);

                    light_entity.emplace<Components::Transform>(Components::Transform{
                            .position = world_position,
                            .rotation = direction_to_rotation(world_direction),
                    });
                }

                if (light.type == ModelLightType::point) {
                    light_entity.emplace<Components::PointLight>(Components::PointLight{
                            .colour = light.colour,
                            .intensity = light.intensity,
                            .range = light.range,
                    });
                } else {
                    light_entity.emplace<Components::SpotLight>(Components::SpotLight{
                            .colour = light.colour,
                            .intensity = light.intensity,
                            .range = light.range,
                            .inner_cone_degrees = light.inner_cone_degrees,
                            .outer_cone_degrees = light.outer_cone_degrees,
                    });
                }
            }

            info("Loaded model '{}', with {} lights", path.string(), renderer.model_lights(model.value()).size());
            return model.value();
        }

        error("[BasicGame::on_populate::load_or_fallback] Could not load model '{}': {}", path.string(),
              describe(model.error()));
        warn("[BasicGame::on_populate::load_or_fallback] Falling back to engine cube for '{}'", path.string());
        return default_model;
    };

    constexpr float physics_radius = 0.35F;
    constexpr float physics_height = 1.0F;

    constexpr float mesh_base_radius = 0.5F;
    constexpr float mesh_base_height = 1.0F;

    constexpr glm::vec3 const capsule_scale{physics_radius / mesh_base_radius, physics_height / mesh_base_height,
                                            physics_radius / mesh_base_radius};

    auto player = Entity{&scene, "player"};
    player.emplace<Components::Transform>(
            Components::Transform{.position = glm::vec3{0.0F, 3.0F, 0.0F}, .scale = capsule_scale});
    player.emplace<Components::RigidBody>(
            Components::RigidBody::make_capsule(physics_radius, physics_height, /*mass=*/80.0F));
    player.emplace<Components::PlayerTag>();
    player.emplace<Components::Model>(Components::Model{.model = engine_models.capsule});
    player_entity_ = player;

    cube_model_ = load_or_fallback("assets/models/test_cube.glb");

    auto const cube_bounds = renderer.model_bounds(cube_model_);
    cube_half_extents_ = cube_bounds.has_value() ? (cube_bounds->second - cube_bounds->first) * 0.5F : glm::vec3{0.5F};

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);

    // Terrain generation parameters, kept around for the rest of on_populate
    // so houses/trees/grass below can sample the same noise field and sit on
    // the actual generated surface instead of a flat plane. The actual mesh
    // is no longer generated here -- see terrain_create_info(), which hands
    // these same params to a streaming TerrainWorld created once on_populate()
    // (and this material) are ready.
    terrain_params_ = TerrainParams{
            .samples_x = 129,
            .samples_z = 129,
            .world_width = 80.0F,
            .world_depth = 80.0F,
            .amplitude = 1.6F,
            .frequency = 0.045F,
            .octaves = 4,
            .lacunarity = 2.0F,
            .persistence = 0.5F,
            .seed = 1337U,
            .uv_scale = 0.08F,
            .height_range_min = -1.6F,
            .height_range_max = 1.6F,
    };
    terrain_ground_y_ = scene.physics_settings.ground_y;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();
    auto &streamer = renderer.texture_streamer();

    // Handles come back immediately, sampling their fallback default
    // texture; the real BC5/BC7 KTX2 texture streams in asynchronously (see
    // texture_pipeline.hxx / TextureStreamer) and swaps in in place once
    // decoded, cached, and uploaded.
    auto const dirt_normal_index = streamer.request(images, "assets/textures/dirt/dirt_nor_gl_1k_zip.exr",
                                                    TextureRole::normal_map, images.flat_normal(), "dirt.normal");

    auto const dirt_albedo_index = streamer.request(images, "assets/textures/dirt/dirt_diff_1k.jpg",
                                                    TextureRole::colour, images.white(), "dirt.albedo");

    auto const dirt_roughness_index =
            streamer.request(images, "assets/textures/dirt/dirt_rough_1k.exr", TextureRole::generic,
                             images.metallic_roughness(), "dirt.roughness");

    auto const terrain_material = renderer.create_material(MaterialCreateInfo{
            .base_colour_factor = glm::vec4{1.0F, 1.0F, 1.0F, 1.0F},
            .base_colour_texture = dirt_albedo_index,
            .normal_texture = dirt_normal_index,
            .metallic_roughness_texture = dirt_roughness_index,
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = samplers.linear_repeat(),
    });

    if (terrain_material) {
        terrain_material_ = *terrain_material;
    } else {
        error("Could not create terrain material: {}", describe(terrain_material.error()));
    }

    constexpr auto village_radius = 14.0F;

    // Houses and trees are built from the engine's box/sphere primitives
    // rather than external assets -- walls, split doorways, overhanging
    // roofs, and chimneys give GTAO real corners and eaves to shade, which a
    // flat floor and scattered cubes did not.
    auto const flat_material = [&](glm::vec3 const &colour) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{colour, 1.0F},
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.metallic_roughness(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = images.emissive(),
                .sampler = samplers.linear_repeat(),
        });

        if (!material) {
            error("Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        return material.value();
    };

    auto const add_static_box = [&](std::string const &name, glm::vec3 const &position, glm::vec3 const &half_extents,
                                    MaterialHandle material) {
        auto entity = GeneratedEntity{&scene, "{}", name};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .scale = half_extents / cube_half_extents_,
        });
        entity.emplace<Components::Model>(Components::Model{.model = cube_model_});
        entity.emplace<Components::RigidBody>(Components::RigidBody{.half_extents = half_extents, .is_static = true});
        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{material});
        }
    };

    struct HouseStyle {
        glm::vec3 position;
        float width;
        float depth;
        float wall_height;
        glm::vec3 wall_colour;
        glm::vec3 roof_colour;
    };

    constexpr std::array<HouseStyle, 4> house_styles{{
            {glm::vec3{-10.0F, 0.0F, -8.0F}, 6.0F, 5.0F, 3.0F, glm::vec3{0.85F, 0.78F, 0.65F},
             glm::vec3{0.45F, 0.2F, 0.18F}},
            {glm::vec3{9.0F, 0.0F, -10.0F}, 5.0F, 5.0F, 2.6F, glm::vec3{0.75F, 0.72F, 0.68F},
             glm::vec3{0.3F, 0.3F, 0.32F}},
            {glm::vec3{10.0F, 0.0F, 9.0F}, 7.0F, 5.5F, 3.4F, glm::vec3{0.88F, 0.6F, 0.45F},
             glm::vec3{0.25F, 0.22F, 0.2F}},
            {glm::vec3{-9.0F, 0.0F, 10.0F}, 5.5F, 5.0F, 3.0F, glm::vec3{0.7F, 0.68F, 0.6F},
             glm::vec3{0.4F, 0.35F, 0.3F}},
    }};

    for (auto house_index = 0; house_index < static_cast<int>(house_styles.size()); ++house_index) {
        auto const &style = house_styles[house_index];
        auto const base_y = scene.physics_settings.ground_y +
                            sample_terrain_height(terrain_params_, style.position.x, style.position.z);
        auto const base = glm::vec3{style.position.x, base_y, style.position.z};

        auto const wall_material = flat_material(style.wall_colour);
        auto const roof_material = flat_material(style.roof_colour);

        constexpr float wall_thickness = 0.3F;
        constexpr float door_width = 1.1F;
        constexpr float roof_overhang = 0.6F;
        constexpr float roof_thickness = 0.25F;

        auto const half_w = style.width * 0.5F;
        auto const half_d = style.depth * 0.5F;
        auto const half_h = style.wall_height * 0.5F;

        auto const part_name = [&](char const *part) { return std::format("house_{}_{}", house_index, part); };

        // Back and side walls span the full footprint; the front wall is
        // split in two to leave a doorway gap between them.
        add_static_box(part_name("wall_back"), base + glm::vec3{0.0F, half_h, -half_d},
                        {half_w, half_h, wall_thickness * 0.5F}, wall_material);
        add_static_box(part_name("wall_left"), base + glm::vec3{-half_w, half_h, 0.0F},
                        {wall_thickness * 0.5F, half_h, half_d}, wall_material);
        add_static_box(part_name("wall_right"), base + glm::vec3{half_w, half_h, 0.0F},
                        {wall_thickness * 0.5F, half_h, half_d}, wall_material);

        auto const front_segment_width = (style.width - door_width) * 0.5F;
        auto const front_segment_offset = (door_width + front_segment_width) * 0.5F;
        add_static_box(part_name("wall_front_left"), base + glm::vec3{-front_segment_offset, half_h, half_d},
                        {front_segment_width * 0.5F, half_h, wall_thickness * 0.5F}, wall_material);
        add_static_box(part_name("wall_front_right"), base + glm::vec3{front_segment_offset, half_h, half_d},
                        {front_segment_width * 0.5F, half_h, wall_thickness * 0.5F}, wall_material);

        add_static_box(part_name("roof"), base + glm::vec3{0.0F, style.wall_height + roof_thickness * 0.5F, 0.0F},
                        {half_w + roof_overhang, roof_thickness * 0.5F, half_d + roof_overhang}, roof_material);

        add_static_box(part_name("chimney"),
                        base + glm::vec3{half_w * 0.5F, style.wall_height + roof_thickness + 0.4F, -half_d * 0.5F},
                        {0.3F, 0.4F, 0.3F}, roof_material);
    }

    {
        auto const trunk_material = flat_material(glm::vec3{0.35F, 0.24F, 0.15F});
        auto const canopy_material = flat_material(glm::vec3{0.22F, 0.45F, 0.2F});
        constexpr float sphere_base_radius = 0.5F;

        struct TreeStyle {
            glm::vec3 position;
            float trunk_height;
            float trunk_radius;
            float canopy_radius;
        };

        constexpr std::array<TreeStyle, 3> tree_styles{{
                {glm::vec3{0.0F, 0.0F, -13.0F}, 2.2F, 0.2F, 1.4F},
                {glm::vec3{-13.0F, 0.0F, 1.0F}, 2.6F, 0.22F, 1.6F},
                {glm::vec3{13.0F, 0.0F, -1.5F}, 2.0F, 0.18F, 1.2F},
        }};

        for (auto tree_index = 0; tree_index < static_cast<int>(tree_styles.size()); ++tree_index) {
            auto const &style = tree_styles[tree_index];
            auto const base_y = scene.physics_settings.ground_y +
                                sample_terrain_height(terrain_params_, style.position.x, style.position.z);
            auto const base = glm::vec3{style.position.x, base_y, style.position.z};

            add_static_box(std::format("tree_{}_trunk", tree_index),
                            base + glm::vec3{0.0F, style.trunk_height * 0.5F, 0.0F},
                            {style.trunk_radius, style.trunk_height * 0.5F, style.trunk_radius}, trunk_material);

            auto canopy_entity = GeneratedEntity{&scene, "tree_{}_canopy", tree_index};
            canopy_entity.emplace<Components::Transform>(Components::Transform{
                    .position = base + glm::vec3{0.0F, style.trunk_height + style.canopy_radius * 0.7F, 0.0F},
                    .scale = glm::vec3{style.canopy_radius / sphere_base_radius},
            });
            canopy_entity.emplace<Components::Model>(Components::Model{.model = engine_models.sphere});
            if (canopy_material.valid()) {
                canopy_entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{canopy_material});
            }
        }
    }

    {
        constexpr std::array<glm::vec3, 4> point_light_colours{
                glm::vec3{1.0F, 0.35F, 0.25F},
                glm::vec3{0.25F, 0.55F, 1.0F},
                glm::vec3{0.35F, 1.0F, 0.4F},
                glm::vec3{1.0F, 0.85F, 0.25F},
        };

        for (std::size_t i = 0; i < point_light_colours.size(); ++i) {
            auto const angle = (static_cast<float>(i) / static_cast<float>(point_light_colours.size())) * 6.2831853F;

            auto const light_entity = GeneratedEntity{&scene, "point_light_{}", i};
            light_entity.emplace<Components::Transform>(Components::Transform{
                    .position = glm::vec3{village_radius * std::cos(angle), 12.0F, village_radius * std::sin(angle)},
            });
            light_entity.emplace<Components::PointLight>(Components::PointLight{
                    .colour = point_light_colours[i],
                    .intensity = 25.0F,
                    .range = 20.0F,
            });
        }

        auto const spot_entity = GeneratedEntity{&scene, "spot_light"};
        spot_entity.emplace<Components::Transform>(Components::Transform{
                .position = glm::vec3{0.0F, 15.0F, 0.0F},
                .rotation = glm::angleAxis(glm::radians(30.0F), glm::vec3{1.0F, 0.0F, 0.0F}),
        });
        spot_entity.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = glm::vec3{0.9F, 0.95F, 1.0F},
                .intensity = 60.0F,
                .range = 30.0F,
                .inner_cone_degrees = 15.0F,
                .outer_cone_degrees = 25.0F,
        });
    }


    auto const grass_material_result = renderer.create_material(MaterialCreateInfo{
            .base_colour_factor = glm::vec4{0.25F, 0.55F, 0.18F, 1.0F},
            .base_colour_texture = images.white(),
            .normal_texture = images.flat_normal(),
            .metallic_roughness_texture = images.metallic_roughness(),
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = samplers.linear_repeat(),
            .wind_strength = 0.28F,
            .max_shadow_cascade = GpuMaterial::no_shadow_cascade,
    });

    if (!grass_material_result) {
        error("Could not create grass material: {}", describe(grass_material_result.error()));
    } else {
        grass_material_ = *grass_material_result;

        constexpr auto grass_field_size = 20.0F;
        constexpr auto grass_spacing = 0.15F;
        constexpr auto grass_cells = static_cast<int>(grass_field_size / grass_spacing);

        std::uniform_real_distribution<float> jitter(-grass_spacing * 0.4F, grass_spacing * 0.4F);
        std::uniform_real_distribution<float> yaw(0.0F, 6.2831853F);
        std::uniform_real_distribution<float> scale{0.85F, 1.15F};


        for (auto cell_x = 0; cell_x < grass_cells; ++cell_x) {
            for (auto cell_z = 0; cell_z < grass_cells; ++cell_z) {
                auto const x =
                        (static_cast<float>(cell_x) + 0.5F) * grass_spacing - grass_field_size * 0.5F + jitter(eng);
                auto const z =
                        (static_cast<float>(cell_z) + 0.5F) * grass_spacing - grass_field_size * 0.5F + jitter(eng);

                auto const grass_entity = GeneratedEntity{&scene, "grass_{}_{}", cell_x, cell_z};
                auto const grass_scale = scale(eng);

                auto const grass_y = scene.physics_settings.ground_y + sample_terrain_height(terrain_params_, x, z);

                grass_entity.emplace<Components::Transform>(Components::Transform{
                        .position = glm::vec3{x, grass_y, z},
                        .rotation = glm::angleAxis(yaw(eng), glm::vec3{0.0F, 1.0F, 0.0F}),
                        .scale = glm::vec3{grass_scale},

                });
                grass_entity.emplace<Components::Model>(Components::Model{.model = engine_models.grass_clump});
                grass_entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{grass_material_});
            }
        }
    }
}

auto BasicGame::on_update(Scene &scene, float delta_time) -> void {
    auto &registry = scene.get_registry();

    if (!registry.valid(player_entity_) ||
        !registry.all_of<Components::Transform, Components::RigidBody>(player_entity_)) {
        return;
    }

    auto &physics_world = *scene.physics_world;

    auto const &transform = registry.get<Components::Transform>(player_entity_);
    auto const &body = registry.get<Components::RigidBody>(player_entity_);

    auto const capsule_half_height = body.capsule_height * 0.5F;
    auto const is_grounded =
            physics_world.is_grounded(registry, player_entity_, capsule_half_height, body.capsule_radius);

    if (player_controller_.consumes_jump() && is_grounded) {
        constexpr float jump_velocity = 6.5F; // Adjust to match gravity
        physics_world.jump(registry, player_entity_, jump_velocity);
    }

    auto const desired_velocity = player_controller_.desired_horizontal_velocity();
    physics_world.set_velocity(registry, player_entity_, desired_velocity);

    auto const speed_factor = player_controller_.move_speed() > 0.0F
                                       ? glm::length(desired_velocity) / player_controller_.move_speed()
                                       : 0.0F;

    auto const occlusion_query = [&physics_world, player = player_entity_](
                                          glm::vec3 const &origin, glm::vec3 const &direction,
                                          float max_distance) -> std::optional<float> {
        auto const hit = physics_world.raycast(origin, direction, max_distance);
        if (hit && hit->entity != player) {
            return hit->distance;
        }
        return std::nullopt;
    };

    player_camera_.update(transform.position, player_controller_.yaw_degrees(), player_controller_.pitch_degrees(),
                          speed_factor, delta_time, occlusion_query);
}

auto BasicGame::on_key_pressed(Scene & /*scene*/, KeyPressedEvent const &event) -> void {
    player_controller_.on_key_pressed(event.key);
}

auto BasicGame::on_key_released(Scene & /*scene*/, KeyReleasedEvent const &event) -> void {
    player_controller_.on_key_released(event.key);
}

auto BasicGame::on_mouse_moved(Scene & /*scene*/, MouseMovedEvent const &event) -> void {
    player_controller_.on_mouse_moved(static_cast<float>(event.delta_x), static_cast<float>(event.delta_y),
                                      /*look_enabled=*/true);
}

auto BasicGame::on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void {
    if (event.button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }

    if (event.modifiers & GLFW_MOD_CONTROL) {
        shoot_bullet(scene, 12);
    } else {
        shoot_bullet(scene);
    }
}

[[nodiscard]] auto BasicGame::camera(Scene const & /*scene*/, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = player_camera_.view(),
            .projection = player_camera_.projection(aspect_ratio),
            .near_clip = player_camera_.near_clip(),
            .far_clip = player_camera_.far_clip(),
            .vertical_fov_radians = glm::radians(player_camera_.field_of_view_degrees()),
    };
}

[[nodiscard]] auto BasicGame::terrain_create_info(Renderer & /*renderer*/) -> std::optional<TerrainWorldCreateInfo> {
    // terrain_params_/terrain_material_/terrain_ground_y_ are set in
    // on_populate(), which always runs first (see Application::on_startup).
    //
    // TerrainLodSettings' own defaults (view_distance = 2048) assume a much
    // larger world than this one; left as-is, the quadtree keeps a steady-state
    // 64 LOD0 chunks split out around the camera (a fixed function of
    // split_factor/split_hysteresis vs. the LOD0 chunk span, independent of
    // view_distance) against a slots_per_lod of 48, so LOD0 permanently runs
    // 16 chunks over budget -- see the "slot pool exhausted" warning spam this
    // fixes. view_distance is cut down to match this small terrain, and
    // slots_per_lod raised past that 64-chunk floor with headroom for the
    // eviction grace period's transient parent/child overlap.
    return TerrainWorldCreateInfo{
            .params = terrain_params_,
            .lod_settings = TerrainLodSettings{.view_distance = 512.0F},
            .slots_per_lod = 96,
            .material = terrain_material_,
            .ground_y = terrain_ground_y_,
    };
}

auto BasicGame::shoot_bullet(Scene &scene, std::size_t n) -> void {
    if (!scene.physics_world) {
        return;
    }

    if (!scene.get_registry().valid(player_entity_) ||
        !scene.get_registry().all_of<Components::Transform>(player_entity_)) {
        return;
    }

    constexpr auto bullet_half_extent = 0.15F;
    constexpr auto bullet_speed = 40.0F;
    constexpr auto bullet_mass = 0.2F;
    constexpr auto bullet_lifetime_seconds = 3.0F;
    constexpr auto max_aim_distance = 1000.0F;
    constexpr auto player_eye_height = 1.5F; // Height offset from player base position

    auto const &position = ReadOnlyEntity{&scene, player_entity_}.get<Components::Transform>().position;
    auto const muzzle_position = position + glm::vec3{0.0F, player_eye_height, 0.0F};

    auto const cam_origin = player_camera_.position();
    auto const cam_forward = player_camera_.forward();

    glm::vec3 target_point = cam_origin + (cam_forward * max_aim_distance);

    if (auto const hit = scene.physics_world->raycast(cam_origin, cam_forward, max_aim_distance)) {
        target_point = hit->point;
    }

    auto const bullet_direction = glm::normalize(target_point - muzzle_position);

    auto const transform = Components::Transform{
            .position = muzzle_position + bullet_direction * (bullet_half_extent + 0.2F),
            .scale = glm::vec3{bullet_half_extent} / cube_half_extents_,
    };
    auto const rigid_body = Components::RigidBody{
            .velocity = bullet_direction * bullet_speed,
            .half_extents = glm::vec3{bullet_half_extent},
            .restitution = 0.3F,
            .mass = bullet_mass,
    };

    for (auto i = 0U; i < n; ++i) {
        auto const entity = GeneratedEntity{&scene, "bullet_{}", static_cast<std::uint32_t>(i)};
        entity.emplace<Components::Transform>(transform);
        entity.emplace<Components::Model>(Components::Model{.model = cube_model_});
        entity.emplace<Components::RigidBody>(rigid_body);
        entity.emplace<Components::Lifetime>(bullet_lifetime_seconds);

        scene.physics_world->add_body(scene.get_registry(), entity, transform, rigid_body);
    }
}
