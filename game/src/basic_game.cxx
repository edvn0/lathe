#include "basic_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numbers>
#include <random>
#include <ranges>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "components.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "logger.hxx"
#include "physics_world.hxx"
#include "renderer.hxx"
#include "scene.hxx"

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

    auto const helmet_model = load_or_fallback("assets/models/damaged_helmet/DamagedHelmet.gltf");
    cube_model_ = load_or_fallback("assets/models/test_cube.glb");

    constexpr auto house_position = glm::vec3{40.0F, 10.0F, 10.0F};
    auto const house_transform = glm::translate(glm::mat4{1.0F}, house_position);

    auto house_entity = Entity{&scene, "house"};
    house_model_ = load_or_fallback("assets/models/scene.glb", house_entity, house_transform);

    house_entity.emplace<Components::Transform>(Components::Transform{.position = house_position});
    house_entity.emplace<Components::Model>(Components::Model{.model = house_model_});
    house_entity.emplace<Components::RigidBody>(
            Components::RigidBody::from_model_bounds(renderer.model_bounds(house_model_).value()));

    tree_model_ = load_or_fallback("assets/models/tree.glb");

    auto tree_entity = Entity{&scene, "tree"};
    tree_entity.emplace<Components::Transform>(Components::Transform{.position = glm::vec3{20.0F, 0.0F, 20.0F}});
    tree_entity.emplace<Components::Model>(Components::Model{.model = tree_model_});

    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 eng(seed);


    std::uniform_real_distribution<float> position_dist(-20.0F, 20.0F);
    std::uniform_real_distribution<float> rotation_dist(0.0F, 1.0F);

    auto random_quat = [&]() -> glm::quat {
        const float u1 = rotation_dist(eng);
        const float u2 = rotation_dist(eng);
        const float u3 = rotation_dist(eng);

        const float a = std::sqrt(1.0F - u1);

        const float b = std::sqrt(u1);

        const float theta1 = 2.0F * std::numbers::pi_v<float> * u2;
        const float theta2 = 2.0F * std::numbers::pi_v<float> * u3;


        return glm::quat{
                b * std::cos(theta2), // w
                a * std::sin(theta1), // x
                a * std::cos(theta1), // y
                b * std::sin(theta2), // z
        };
    };


    const auto count = 10;
    auto const bounds = renderer.model_bounds(helmet_model);

    for (auto i = 0; i < count * count * count; ++i) {
        auto entity = GeneratedEntity{&scene, "helmet_{}_{}_{}", i, i % 3, i / 3};

        entity.emplace<Components::Transform>(Components::Transform{
                .position =
                        glm::vec3{
                                5.0F * position_dist(eng),
                                5.0F * position_dist(eng),
                                5.0F * position_dist(eng),
                        },
                .rotation = random_quat(),
        });

        entity.emplace<Components::Model>(Components::Model{.model = helmet_model});

        if (bounds.has_value()) {
            auto const [min, max] = *bounds;
            auto const half_extents = (max - min) * 0.5F;

            entity.emplace<Components::RigidBody>(Components::RigidBody{.half_extents = half_extents});
        }
    }

    constexpr auto physics_grid = 5;

    auto const cube_bounds = renderer.model_bounds(cube_model_);
    cube_half_extents_ = cube_bounds.has_value() ? (cube_bounds->second - cube_bounds->first) * 0.5F : glm::vec3{0.5F};
    auto const cube_diameter = std::max({cube_half_extents_.x, cube_half_extents_.y, cube_half_extents_.z}) * 2.0F;
    auto const spacing = cube_diameter * 1.3F;

    for (auto i = 0; i < physics_grid; i++) {
        for (auto j = 0; j < physics_grid; j++) {
            for (auto k = 0; k < physics_grid; k++) {
                auto const entity = GeneratedEntity{&scene, "physics_cube_{}_{}_{}", i, j, k};
                auto const position = glm::vec3{
                        static_cast<float>(i - physics_grid / 2) * spacing,
                        5.0F + static_cast<float>(j) * spacing,
                        static_cast<float>(k - physics_grid / 2) * spacing,
                };
                entity.emplace<Components::Transform>(Components::Transform{.position = position});
                entity.emplace<Components::Model>(Components::Model{.model = cube_model_});
                entity.emplace<Components::RigidBody>(Components::RigidBody{.half_extents = cube_half_extents_});
            }
        }
    }


    constexpr auto floor_half_extents = glm::vec3{40.0F, 0.5F, 40.0F};

    auto const floor_entity = GeneratedEntity{&scene, "floor"};
    floor_entity.emplace<Components::Transform>(Components::Transform{
            .position = glm::vec3{0.0F, scene.physics_settings.ground_y - floor_half_extents.y, 0.0F},
            .scale = floor_half_extents / cube_half_extents_,
    });
    floor_entity.emplace<Components::Model>(Components::Model{.model = cube_model_});
    floor_entity.emplace<Components::RigidBody>(
            Components::RigidBody{.half_extents = floor_half_extents, .is_static = true});

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

    auto const floor_material = renderer.create_material(MaterialCreateInfo{
            .base_colour_factor = glm::vec4{1.0F, 1.0F, 1.0F, 1.0F},
            .base_colour_texture = dirt_albedo_index,
            .normal_texture = dirt_normal_index,
            .metallic_roughness_texture = dirt_roughness_index,
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = samplers.linear_repeat(),
    });

    if (floor_material) {
        floor_entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{*floor_material});
    } else {
        error("Could not create floor material override: {}", describe(floor_material.error()));
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
            auto const radius = spacing * static_cast<float>(physics_grid) * 0.5F;

            auto const light_entity = GeneratedEntity{&scene, "point_light_{}", i};
            light_entity.emplace<Components::Transform>(Components::Transform{
                    .position = glm::vec3{radius * std::cos(angle), 12.0F, radius * std::sin(angle)},
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

        constexpr auto grass_field_size = 30.0F;
        constexpr auto grass_spacing = 0.1F;
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

                grass_entity.emplace<Components::Transform>(Components::Transform{
                        .position = glm::vec3{x, scene.physics_settings.ground_y, z},
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

    player_camera_.update(transform.position, player_controller_.yaw_degrees(), player_controller_.pitch_degrees(),
                          speed_factor, delta_time);
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
