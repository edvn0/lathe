#include "basic_game.hxx"
#include "core/paths.hxx"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <numbers>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

#include "assets/primitive_meshes.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/random.hxx"
#include "enemy_ai_script.hxx"
#include "net/http.hxx"
#include "physics/physics_world.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/script_storage.hxx"
#include "scene/components.hxx"
#include "terrain/terrain_mesh.hxx"

namespace {

    [[nodiscard]] auto hash_to_unit_float(std::uint32_t x, std::uint32_t z, std::uint32_t seed) -> float {
        std::uint32_t h = x * 374761393U + z * 668265263U + seed * 2654435761U;
        h = (h ^ (h >> 13)) * 1274126177U;
        h ^= h >> 16;
        return static_cast<float>(h) / static_cast<float>(std::numeric_limits<std::uint32_t>::max());
    }

    [[nodiscard]] auto value_noise_2d(glm::vec2 pos, std::uint32_t seed) -> float {
        auto const cell = glm::floor(pos);
        auto const frac = pos - cell;

        auto const cell_x = static_cast<std::uint32_t>(static_cast<std::int32_t>(cell.x));
        auto const cell_z = static_cast<std::uint32_t>(static_cast<std::int32_t>(cell.y));

        auto const corner00 = hash_to_unit_float(cell_x, cell_z, seed);
        auto const corner10 = hash_to_unit_float(cell_x + 1U, cell_z, seed);
        auto const corner01 = hash_to_unit_float(cell_x, cell_z + 1U, seed);
        auto const corner11 = hash_to_unit_float(cell_x + 1U, cell_z + 1U, seed);

        auto const smooth = frac * frac * (glm::vec2{3.0F} - 2.0F * frac);
        auto const top = glm::mix(corner00, corner10, smooth.x);
        auto const bottom = glm::mix(corner01, corner11, smooth.x);
        return glm::mix(top, bottom, smooth.y);
    }

    [[nodiscard]] auto blotch_density(glm::vec2 world_pos, GrassParams const &params) -> float {
        constexpr auto octaves = 3;
        constexpr auto lacunarity = 2.0F;
        constexpr auto persistence = 0.5F;

        auto sample_pos = world_pos / params.blotch_scale;
        auto amplitude = 1.0F;
        auto amplitude_sum = 0.0F;
        auto density = 0.0F;

        for (auto octave = 0; octave < octaves; ++octave) {
            density += value_noise_2d(sample_pos, params.blotch_seed + static_cast<std::uint32_t>(octave)) * amplitude;
            amplitude_sum += amplitude;
            sample_pos *= lacunarity;
            amplitude *= persistence;
        }

        return density / amplitude_sum;
    }

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

    auto sample_spline(std::span<glm::vec2 const> control, bool closed, float step) -> std::vector<glm::vec2> {
        auto const count = static_cast<int>(control.size());
        auto const point = [&](int index) {
            return closed ? control[static_cast<std::size_t>((index % count + count) % count)]
                          : control[static_cast<std::size_t>(std::clamp(index, 0, count - 1))];
        };

        std::vector<glm::vec2> samples;

        for (int segment = 0; segment < (closed ? count : count - 1); ++segment) {
            auto const p0 = point(segment - 1);
            auto const p1 = point(segment);
            auto const p2 = point(segment + 1);
            auto const p3 = point(segment + 2);

            auto const steps = std::max(1, static_cast<int>(std::ceil(glm::distance(p1, p2) / step)));

            for (int k = 0; k < steps; ++k) {
                auto const t = static_cast<float>(k) / static_cast<float>(steps);
                samples.push_back(0.5F *
                                  ((2.0F * p1) + (p2 - p0) * t + (2.0F * p0 - 5.0F * p1 + 4.0F * p2 - p3) * t * t +
                                   (3.0F * p1 - p0 - 3.0F * p2 + p3) * t * t * t));
            }
        }

        if (!closed) {
            samples.push_back(control.back());
        }

        return samples;
    }

}

namespace {

    constexpr std::string_view helmet_url =
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
            "5bad5aaa0bbb5d0f9cdc934e626f27d0df1e79b8/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb";
    constexpr std::string_view helmet_sha256 = "a1e3b04de97b11de564ce6e53b95f02954a297f0008183ac63a4f5974f6b32d8";
    auto helmet_path() -> CachePath { return cache_path("downloads/damaged_helmet.glb"); }

}

auto BasicGame::request_helmet() -> void {
    if (helmet_download_.valid() || helmet_model_.valid()) {
        return;
    }

    std::error_code directory_error;
    std::filesystem::create_directories(helmet_path().absolute().parent_path(), directory_error);

    helmet_download_ = http_client_.get_file_async({
            .uri = std::string{helmet_url},
            .destination = helmet_path().absolute(),
            .sha256 = std::string{helmet_sha256},
    });
}

auto BasicGame::spawn_helmet(Scene &scene, Renderer &renderer) -> void {
    if (!helmet_model_.valid()) {
        auto model = renderer.load_model(AssetPath::external(helmet_path().absolute()).value_or(AssetPath::missing()));

        if (!model) {
            error("Could not load the DamagedHelmet: {}", describe(model.error()));
            return;
        }

        helmet_model_ = model.value();
    }

    auto helmet = Entity{&scene, "damaged_helmet"};
    helmet.emplace<Components::Transform>(Components::Transform{
            .position = glm::vec3{4.0F, 2.0F, -8.5F},
            .scale = glm::vec3{2.0F},
    });
    helmet.emplace<Components::Model>(Components::Model{.model = helmet_model_});
}

auto BasicGame::poll_helmet(Scene &scene, Renderer &renderer) -> void {
    if (!helmet_download_.valid() ||
        helmet_download_.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
        return;
    }

    auto const result = helmet_download_.get();

    if (!result) {
        warn("Could not fetch the DamagedHelmet: {}", describe(result.error()));
        return;
    }

    spawn_helmet(scene, renderer);
}

auto BasicGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    if (auto could_wait = renderer.wait_idle(); !could_wait.has_value()) {
        info("{}", describe(could_wait.error()));
        return;
    }

    scene.get_registry().clear();
    engine_models_ = engine_models;

    if (auto const created = particles_.create(renderer.game_gpu()); !created) {
        warn("Could not set up the particle field: {}", describe(created.error()));
    }

    if (helmet_model_.valid()) {
        spawn_helmet(scene, renderer);
    } else {
        request_helmet();
    }

    auto const load_or_fallback = [&renderer, &default_model = engine_models.cube, s = &scene](
                                          AssetPath const &path, entt::entity parent_entity = entt::null,
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

            info("Loaded model '{}', with {} lights", path.key(), renderer.model_lights(model.value()).size());
            return model.value();
        }

        error("[BasicGame::on_populate::load_or_fallback] Could not load model '{}': {}", path.key(),
              describe(model.error()));
        warn("[BasicGame::on_populate::load_or_fallback] Falling back to engine cube for '{}'", path.key());
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
            Components::RigidBody::make_capsule(physics_radius, physics_height, 80.0F));
    player.emplace<Components::PlayerTag>();
    player.emplace<Components::Model>(Components::Model{.model = engine_models.capsule});
    player_entity_ = player;

    constexpr auto enemy_count = 10U;
    constexpr float enemy_orbit_radius = 6.0F;
    constexpr float enemy_capsule_radius = 0.3F;
    constexpr float enemy_capsule_height = 0.8F;

    if (!enemy_ai_script_) {
        if (auto const created = scene.get_scripts().emplace<EnemyAIScript>()) {
            enemy_ai_script_ = ScriptHolder{scene.get_scripts(), *created};
        } else {
            error("[BasicGame::on_populate] Could not create EnemyAIScript instance");
        }
    }

    if (enemy_ai_script_) {
        auto const enemy_scale =
                glm::vec3{enemy_capsule_radius / mesh_base_radius, enemy_capsule_height / mesh_base_height,
                          enemy_capsule_radius / mesh_base_radius};

        for (auto i = 0U; i < enemy_count; ++i) {
            auto const angle =
                    (static_cast<float>(i) / static_cast<float>(enemy_count)) * 2.0F * std::numbers::pi_v<float>;
            auto const center = glm::vec3{0.0F, 3.0F, 0.0F};

            auto enemy = GeneratedEntity{&scene, "enemy_{}", i};
            enemy.emplace<Components::Transform>(Components::Transform{
                    .position = center + glm::vec3{enemy_orbit_radius * std::cos(angle), 0.0F,
                                                   enemy_orbit_radius * std::sin(angle)},
                    .scale = enemy_scale,
            });
            enemy.emplace<Components::CircularMotion>(Components::CircularMotion{
                    .center = center,
                    .radius = enemy_orbit_radius,
                    .angular_speed = 0.6F,
                    .angle = angle,
            });
            enemy.emplace<Components::Model>(Components::Model{.model = engine_models.capsule});
            enemy.emplace<Components::Script>(Components::Script{.script = enemy_ai_script_.handle()});
        }
    }

    auto const release_previous = [&](ModelHandle previous) -> void {
        if (previous.valid() && previous != engine_models.cube) {
            renderer.release_model(previous);
        }
    };

    auto const previous_cube_model = std::exchange(cube_model_, load_or_fallback(data_path("assets/models/test_cube.glb")));
    release_previous(previous_cube_model);

    auto const cube_bounds = renderer.model_bounds(cube_model_);
    cube_half_extents_ = cube_bounds.has_value() ? (cube_bounds->second - cube_bounds->first) * 0.5F : glm::vec3{0.5F};

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
            .hills =
                    {
                            {.world_x = 48.0F, .world_z = -30.0F, .height = 16.0F, .radius = 14.0F},
                            {.world_x = -55.0F, .world_z = 28.0F, .height = 20.0F, .radius = 16.0F},
                            {.world_x = 22.0F, .world_z = 58.0F, .height = 11.0F, .radius = 12.0F},
                            {.world_x = -32.0F, .world_z = -58.0F, .height = 14.0F, .radius = 13.0F},
                            {.world_x = 72.0F, .world_z = 28.0F, .height = 18.0F, .radius = 15.0F},
                            {.world_x = -78.0F, .world_z = -22.0F, .height = 24.0F, .radius = 18.0F},
                            {.world_x = 5.0F, .world_z = -88.0F, .height = 15.0F, .radius = 14.0F},
                            {.world_x = -12.0F, .world_z = 84.0F, .height = 17.0F, .radius = 15.0F},
                            {.world_x = 88.0F, .world_z = -62.0F, .height = 22.0F, .radius = 17.0F},
                            {.world_x = -92.0F, .world_z = 72.0F, .height = 13.0F, .radius = 12.0F},
                            {.world_x = 58.0F, .world_z = 88.0F, .height = 12.0F, .radius = 12.0F},
                            {.world_x = -62.0F, .world_z = -88.0F, .height = 19.0F, .radius = 16.0F},
                            {.world_x = 125.0F, .world_z = 70.0F, .height = 40.0F, .radius = 28.0F},
                    },
            .height_range_min = -1.6F,
            .height_range_max = 44.0F,
    };
    terrain_ground_y_ = scene.physics_settings.ground_y;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();
    auto &streamer = renderer.texture_streamer();

    auto const terrain_normal_index =
            streamer.request(images, data_path("assets/textures/terrain/terrain_normal.exr"), TextureRole::normal_map,
                             images.flat_normal(), FlyString{"terrain.normal"});

    auto const terrain_albedo_index = streamer.request(images, data_path("assets/textures/terrain/terrain_albedo.png"),
                                                    TextureRole::colour, images.white(), FlyString{"terrain.albedo"});

    auto const terrain_roughness_index =
            streamer.request(images, data_path("assets/textures/terrain/terrain_roughness.png"), TextureRole::generic,
                             images.metallic_roughness(), FlyString{"terrain.roughness"});

    if (!terrain_material_.valid()) {
        auto const terrain_material = renderer.create_material(
                MaterialCreateInfo{
                        .base_colour_factor = glm::vec4{1.0F, 1.0F, 1.0F, 1.0F},
                        .base_colour_texture = terrain_albedo_index,
                        .normal_texture = terrain_normal_index,
                        .metallic_roughness_texture = terrain_roughness_index,
                        .occlusion_texture = images.occlusion(),
                        .emissive_texture = images.emissive(),
                        .sampler = samplers.linear_repeat(),
                },
                "terrain");

        if (terrain_material) {
            terrain_material_ = *terrain_material;
        } else {
            error("Could not create terrain material: {}", describe(terrain_material.error()));
        }
    }

    constexpr auto village_radius = 14.0F;

    std::vector<MaterialHandle> scene_materials;

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
        scene_materials.push_back(*material);
        return material.value();
    };

    auto const add_static_box = [&](std::string const &name, glm::vec3 const &position, glm::vec3 const &half_extents,
                                    MaterialHandle material, entt::entity parent = entt::null) {
        auto entity = GeneratedEntity{&scene, "{}", name};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .scale = half_extents / cube_half_extents_,
        });
        entity.emplace<Components::Model>(Components::Model{.model = cube_model_});
        entity.emplace<Components::RigidBody>(Components::RigidBody{.half_extents = half_extents, .is_static = true});
        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
        if (parent != entt::null) {
            entity.emplace<Components::Parent>(Components::Parent{.entity = parent});
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
            {.position = glm::vec3{-10.0F, 0.0F, -8.0F},
             .width = 6.0F,
             .depth = 5.0F,
             .wall_height = 3.0F,
             .wall_colour = glm::vec3{0.85F, 0.78F, 0.65F},
             .roof_colour = glm::vec3{0.45F, 0.2F, 0.18F}},
            {.position = glm::vec3{9.0F, 0.0F, -10.0F},
             .width = 5.0F,
             .depth = 5.0F,
             .wall_height = 2.6F,
             .wall_colour = glm::vec3{0.75F, 0.72F, 0.68F},
             .roof_colour = glm::vec3{0.3F, 0.3F, 0.32F}},
            {.position = glm::vec3{10.0F, 0.0F, 9.0F},
             .width = 7.0F,
             .depth = 5.5F,
             .wall_height = 3.4F,
             .wall_colour = glm::vec3{0.88F, 0.6F, 0.45F},
             .roof_colour = glm::vec3{0.25F, 0.22F, 0.2F}},
            {.position = glm::vec3{-9.0F, 0.0F, 10.0F},
             .width = 5.5F,
             .depth = 5.0F,
             .wall_height = 3.0F,
             .wall_colour = glm::vec3{0.7F, 0.68F, 0.6F},
             .roof_colour = glm::vec3{0.4F, 0.35F, 0.3F}},
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

        auto house_entity = GeneratedEntity{&scene, "house_{}", house_index};
        house_entity.emplace<Components::Transform>();

        add_static_box(part_name("wall_back"), base + glm::vec3{0.0F, half_h, -half_d},
                       {half_w, half_h, wall_thickness * 0.5F}, wall_material, house_entity);
        add_static_box(part_name("wall_left"), base + glm::vec3{-half_w, half_h, 0.0F},
                       {wall_thickness * 0.5F, half_h, half_d}, wall_material, house_entity);
        add_static_box(part_name("wall_right"), base + glm::vec3{half_w, half_h, 0.0F},
                       {wall_thickness * 0.5F, half_h, half_d}, wall_material, house_entity);

        auto const front_segment_width = (style.width - door_width) * 0.5F;
        auto const front_segment_offset = (door_width + front_segment_width) * 0.5F;
        add_static_box(part_name("wall_front_left"), base + glm::vec3{-front_segment_offset, half_h, half_d},
                       {front_segment_width * 0.5F, half_h, wall_thickness * 0.5F}, wall_material, house_entity);
        add_static_box(part_name("wall_front_right"), base + glm::vec3{front_segment_offset, half_h, half_d},
                       {front_segment_width * 0.5F, half_h, wall_thickness * 0.5F}, wall_material, house_entity);

        add_static_box(part_name("roof"), base + glm::vec3{0.0F, style.wall_height + roof_thickness * 0.5F, 0.0F},
                       {half_w + roof_overhang, roof_thickness * 0.5F, half_d + roof_overhang}, roof_material,
                       house_entity);

        add_static_box(part_name("chimney"),
                       base + glm::vec3{half_w * 0.5F, style.wall_height + roof_thickness + 0.4F, -half_d * 0.5F},
                       {0.3F, 0.4F, 0.3F}, roof_material, house_entity);
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
                {.position = glm::vec3{0.0F, 0.0F, -13.0F},
                 .trunk_height = 2.2F,
                 .trunk_radius = 0.2F,
                 .canopy_radius = 1.4F},
                {.position = glm::vec3{-13.0F, 0.0F, 1.0F},
                 .trunk_height = 2.6F,
                 .trunk_radius = 0.22F,
                 .canopy_radius = 1.6F},
                {.position = glm::vec3{13.0F, 0.0F, -1.5F},
                 .trunk_height = 2.0F,
                 .trunk_radius = 0.18F,
                 .canopy_radius = 1.2F},
        }};

        for (auto tree_index = 0; tree_index < static_cast<int>(tree_styles.size()); ++tree_index) {
            auto const &style = tree_styles[tree_index];
            auto const base_y = scene.physics_settings.ground_y +
                                sample_terrain_height(terrain_params_, style.position.x, style.position.z);
            auto const base = glm::vec3{style.position.x, base_y, style.position.z};

            auto tree_entity = GeneratedEntity{&scene, "tree_{}", tree_index};
            tree_entity.emplace<Components::Transform>();

            add_static_box(
                    std::format("tree_{}_trunk", tree_index), base + glm::vec3{0.0F, style.trunk_height * 0.5F, 0.0F},
                    {style.trunk_radius, style.trunk_height * 0.5F, style.trunk_radius}, trunk_material, tree_entity);

            auto canopy_entity = GeneratedEntity{&scene, "tree_{}_canopy", tree_index};
            canopy_entity.emplace<Components::Transform>(Components::Transform{
                    .position = base + glm::vec3{0.0F, style.trunk_height + style.canopy_radius * 0.7F, 0.0F},
                    .scale = glm::vec3{style.canopy_radius / sphere_base_radius},
            });
            canopy_entity.emplace<Components::Model>(Components::Model{.model = engine_models.sphere});
            if (canopy_material.valid()) {
                canopy_entity.emplace<Components::MaterialOverride>(
                        Components::MaterialOverride{.material = canopy_material});
            }
            canopy_entity.emplace<Components::Parent>(Components::Parent{.entity = tree_entity});
        }
    }

    {
        constexpr auto skull_position = glm::vec3{0.0F, 0.0F, -8.5F};
        constexpr float skull_scale = 8.0F;

        auto const previous_skull_model =
                std::exchange(skull_model_, load_or_fallback(data_path("assets/models/scattering_skull.glb")));
        release_previous(previous_skull_model);
        auto const skull_model = skull_model_;

        if (skull_model != engine_models.cube) {
            for (auto const material: renderer.model_materials(skull_model)) {
                auto const *source = renderer.material_storage().create_info(material);

                if (source == nullptr) {
                    continue;
                }

                auto debug_info = *source;
                debug_info.debug_meshlet_colours = true;

                if (auto const updated = renderer.update_material(material, debug_info); !updated) {
                    error("Could not enable meshlet colours on the skull material: {}", describe(updated.error()));
                }

                auto &named_materials = renderer.assets().materials();
                if (named_materials.name_of(material).empty()) {
                    static_cast<void>(named_materials.register_asset("scattering_skull", material));
                }
            }
        }

        auto const base_y = scene.physics_settings.ground_y +
                            sample_terrain_height(terrain_params_, skull_position.x, skull_position.z) - 0.1F;

        auto skull = Entity{&scene, "meshlet_debug_skull"};
        skull.emplace<Components::Transform>(Components::Transform{
                .position = glm::vec3{skull_position.x, base_y, skull_position.z},
                .scale = glm::vec3{skull_scale},
        });
        skull.emplace<Components::Model>(Components::Model{.model = skull_model});
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

    {
        struct RoadSpec {
            std::vector<glm::vec2> control;
            bool closed;
            float width;
            glm::vec3 lamp_colour;
        };

        std::vector<RoadSpec> roads;

        {
            constexpr std::array<float, 9> outer_radii{40.0F, 46.0F, 38.0F, 44.0F, 40.0F, 36.0F, 46.0F, 42.0F, 38.0F};
            RoadSpec outer{.control = {}, .closed = true, .width = 6.0F, .lamp_colour = {0.75F, 0.88F, 1.0F}};
            for (std::size_t i = 0; i < outer_radii.size(); ++i) {
                auto const angle = static_cast<float>(i) / static_cast<float>(outer_radii.size()) * 6.2831853F;
                outer.control.emplace_back(outer_radii[i] * std::cos(angle), outer_radii[i] * std::sin(angle));
            }
            roads.push_back(std::move(outer));

            RoadSpec inner{.control = {}, .closed = true, .width = 5.0F, .lamp_colour = {1.0F, 0.72F, 0.38F}};
            for (std::size_t i = 0; i < 6; ++i) {
                auto const angle = (static_cast<float>(i) / 6.0F) * 6.2831853F + 0.3F;
                auto const radius = 27.0F + ((i % 2 == 0) ? 1.5F : -1.5F);
                inner.control.emplace_back(radius * std::cos(angle), radius * std::sin(angle));
            }
            roads.push_back(std::move(inner));
        }

        roads.push_back(
                {.control = {{-38.0F, 14.0F}, {-43.0F, 29.0F}, {-55.0F, 37.0F}, {-66.0F, 30.0F}, {-60.0F, 20.0F}},
                 .closed = false,
                 .width = 4.0F,
                 .lamp_colour = {1.0F, 0.45F, 0.75F}});
        roads.push_back({.control = {{40.0F, 0.0F}, {54.0F, 10.0F}, {66.0F, 16.0F}, {74.0F, 28.0F}},
                         .closed = false,
                         .width = 4.0F,
                         .lamp_colour = {0.4F, 1.0F, 0.75F}});
        roads.push_back({.control = {{7.0F, -40.0F}, {3.0F, -58.0F}, {11.0F, -72.0F}, {5.0F, -84.0F}},
                         .closed = false,
                         .width = 4.0F,
                         .lamp_colour = {0.75F, 0.5F, 1.0F}});

        constexpr float road_lift = 0.15F;
        constexpr float lamp_spacing = 7.0F;
        constexpr float pole_height = 4.5F;
        constexpr float lamp_radius = 0.22F;
        constexpr float sphere_radius_unscaled = 0.5F;
        constexpr std::size_t max_lamp_lights = 190;

        for (auto const model: road_models_) {
            renderer.release_model(model);
        }
        road_models_.clear();
        road_samples_.clear();

        auto const surface_y = [&](glm::vec2 const &xz) {
            return scene.physics_settings.ground_y + sample_terrain_height(terrain_params_, xz.x, xz.y);
        };

        auto const material_for = [&](MaterialCreateInfo info) -> MaterialHandle {
            info.base_colour_texture = images.white();
            info.normal_texture = images.flat_normal();
            info.metallic_roughness_texture = images.metallic_roughness();
            info.occlusion_texture = images.occlusion();
            info.emissive_texture = images.emissive();
            info.sampler = samplers.linear_repeat();
            info.metallic_factor = 0.0F;

            auto material = renderer.create_material(info);
            if (!material) {
                error("Could not create road material: {}", describe(material.error()));
                return MaterialHandle{};
            }
            scene_materials.push_back(*material);
            return *material;
        };

        auto const road_material = material_for(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{0.06F, 0.06F, 0.07F, 1.0F}, .roughness_factor = 0.85F});
        auto const pole_material = material_for(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{0.12F, 0.12F, 0.13F, 1.0F}, .roughness_factor = 0.6F});

        std::size_t lamp_count = 0;

        for (auto const &[road_index, road]: roads | std::views::enumerate) {
            auto const samples = sample_spline(road.control, road.closed, 1.5F);
            auto const count = static_cast<int>(samples.size());
            auto const half_width = road.width * 0.5F;

            auto const forward_at = [&](int i) {
                auto const next = road.closed ? samples[static_cast<std::size_t>((i + 1) % count)]
                                              : samples[static_cast<std::size_t>(std::min(i + 1, count - 1))];
                auto const previous = road.closed ? samples[static_cast<std::size_t>((i + count - 1) % count)]
                                                  : samples[static_cast<std::size_t>(std::max(i - 1, 0))];
                return glm::normalize(next - previous);
            };
            auto const across_at = [&](int i) {
                auto const forward = forward_at(i);
                return glm::vec2{-forward.y, forward.x};
            };

            auto const lift = road_lift + 0.01F * static_cast<float>(road_index);
            std::vector<glm::vec3> grid;
            grid.reserve(static_cast<std::size_t>(count + 1) * 3U);

            for (int i = 0; i < count + (road.closed ? 1 : 0); ++i) {
                auto const index = i % count;
                auto const centre = samples[static_cast<std::size_t>(index)];
                auto const across = across_at(index);

                for (int column = -1; column <= 1; ++column) {
                    auto const xz = centre + across * (static_cast<float>(column) * half_width);
                    grid.emplace_back(xz.x, surface_y(xz) + lift, xz.y);
                }
            }

            for (auto const &centre: samples) {
                road_samples_.emplace_back(centre.x, centre.y, half_width);
            }

            auto mesh = make_ribbon_mesh(grid, 3U);
            if (!mesh) {
                error("Could not build road mesh {}: {}", road_index, describe(mesh.error()));
                continue;
            }
            auto model = renderer.create_model_from_cpu_data(to_model_cpu_data(std::move(*mesh)));
            if (!model) {
                error("Could not upload road model {}: {}", road_index, describe(model.error()));
                continue;
            }
            road_models_.push_back(*model);

            auto const road_entity = GeneratedEntity{&scene, "road_{}", road_index};
            road_entity.emplace<Components::Transform>();
            road_entity.emplace<Components::Model>(Components::Model{.model = *model});
            if (road_material.valid()) {
                road_entity.emplace<Components::MaterialOverride>(
                        Components::MaterialOverride{.material = road_material});
            }

            auto const lamp_group = GeneratedEntity{&scene, "road_{}_lamps", road_index};
            lamp_group.emplace<Components::Transform>();

            auto const lamp_material = material_for(MaterialCreateInfo{
                    .base_colour_factor = glm::vec4{road.lamp_colour, 1.0F},
                    .emissive_factor = road.lamp_colour,
                    .emissive_strength = 6.0F,
                    .roughness_factor = 0.4F,
            });

            float distance_until_lamp = 0.0F;
            int side = 1;
            std::size_t lamp_index = 0;

            for (int i = 0; i < count; ++i) {
                auto const centre = samples[static_cast<std::size_t>(i)];

                if (i > 0) {
                    distance_until_lamp -= glm::distance(centre, samples[static_cast<std::size_t>(i - 1)]);
                }
                if (distance_until_lamp > 0.0F) {
                    continue;
                }
                distance_until_lamp += lamp_spacing;

                if (lamp_count >= max_lamp_lights) {
                    warn("Road lamp limit ({}) reached; remaining lamps skipped", max_lamp_lights);
                    break;
                }

                auto const xz = centre + across_at(i) * (static_cast<float>(side) * (half_width + 0.7F));
                side = -side;
                auto const base = glm::vec3{xz.x, surface_y(xz), xz.y};

                add_static_box(std::format("road_{}_pole_{}", road_index, lamp_index),
                               base + glm::vec3{0.0F, pole_height * 0.5F, 0.0F}, {0.07F, pole_height * 0.5F, 0.07F},
                               pole_material, lamp_group);

                auto const lamp = GeneratedEntity{&scene, "road_{}_lamp_{}", road_index, lamp_index};
                lamp.emplace<Components::Transform>(Components::Transform{
                        .position = base + glm::vec3{0.0F, pole_height + lamp_radius, 0.0F},
                        .scale = glm::vec3{lamp_radius / sphere_radius_unscaled},
                });
                lamp.emplace<Components::Model>(Components::Model{.model = engine_models.sphere});
                if (lamp_material.valid()) {
                    lamp.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = lamp_material});
                }
                lamp.emplace<Components::PointLight>(Components::PointLight{
                        .colour = road.lamp_colour,
                        .intensity = 40.0F,
                        .range = 14.0F,
                });
                lamp.emplace<Components::Parent>(Components::Parent{.entity = lamp_group});

                ++lamp_index;
                ++lamp_count;
            }
        }

        info("Built {} roads with {} lamps", road_models_.size(), lamp_count);
    }

    grass_material_info_ = MaterialCreateInfo{
            .base_colour_factor = glm::vec4{0.25F, 0.55F, 0.18F, 1.0F},
            .base_colour_texture = images.white(),
            .normal_texture = images.flat_normal(),
            .metallic_roughness_texture = images.metallic_roughness(),
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = samplers.linear_repeat(),
            .wind_strength = 0.28F,
            .max_shadow_cascade = GpuMaterial::no_shadow_cascade,
    };
    if (auto const materials = grass_materials(renderer, engine_models, grass_material_info_, grass_materials_);
        materials) {
        grass_materials_ = *materials;
    } else {
        error("Could not create the grass materials: {}", describe(materials.error()));
    }

    if (grass_materials_.blades.valid()) {
        auto const grass_field_entity = GeneratedEntity{&scene, "grass_field"};
        grass_field_entity.emplace<Components::InstancedModel>(Components::InstancedModel{
                .model = engine_models.grass_clump,
                .material_override = grass_materials_.blades,
        });
        grass_field_entity_ = grass_field_entity;

        rebuild_grass_field(scene);
    }

    for (auto const material: scene_materials) {
        renderer.release_material(material);
    }
}

auto BasicGame::rebuild_grass_field(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    if (!registry.valid(grass_field_entity_) || !registry.all_of<Components::InstancedModel>(grass_field_entity_)) {
        return;
    }

    auto const grass_cells = static_cast<int>(grass_field_params_.field_size / grass_field_params_.spacing);

    auto grass_eng = make_random_engine(1);

    std::uniform_real_distribution<float> jitter(-grass_field_params_.spacing * 0.4F,
                                                 grass_field_params_.spacing * 0.4F);
    std::uniform_real_distribution<float> yaw(0.0F, 6.2831853F);
    std::uniform_real_distribution<float> scale{0.85F, 1.15F};
    std::uniform_real_distribution<float> spawn_roll(0.0F, 1.0F);

    std::vector<glm::mat4> grass_transforms;
    grass_transforms.reserve(static_cast<std::size_t>(grass_cells) * static_cast<std::size_t>(grass_cells));

    for (auto cell_x = 0; cell_x < grass_cells; ++cell_x) {
        for (auto cell_z = 0; cell_z < grass_cells; ++cell_z) {
            auto const x = (static_cast<float>(cell_x) + 0.5F) * grass_field_params_.spacing -
                           grass_field_params_.field_size * 0.5F + jitter(grass_eng);
            auto const z = (static_cast<float>(cell_z) + 0.5F) * grass_field_params_.spacing -
                           grass_field_params_.field_size * 0.5F + jitter(grass_eng);

            constexpr auto road_margin = 0.6F;
            auto const on_road = std::ranges::any_of(road_samples_, [&](glm::vec3 const &road) {
                auto const reach = road.z + road_margin;
                return glm::dot(glm::vec2{x, z} - glm::vec2{road}, glm::vec2{x, z} - glm::vec2{road}) < reach * reach;
            });
            if (on_road) {
                continue;
            }

            auto const density = blotch_density(glm::vec2{x, z}, grass_field_params_);
            auto const spawn_chance = glm::smoothstep(
                    grass_field_params_.blotch_threshold - grass_field_params_.blotch_softness,
                    grass_field_params_.blotch_threshold + grass_field_params_.blotch_softness, density);

            if (spawn_roll(grass_eng) >= spawn_chance) {
                continue;
            }

            auto const grass_scale = scale(grass_eng);
            auto const grass_y = scene.physics_settings.ground_y + sample_terrain_height(terrain_params_, x, z);

            grass_transforms.push_back(Components::Transform{
                    .position = glm::vec3{x, grass_y, z},
                    .rotation = glm::angleAxis(yaw(grass_eng), glm::vec3{0.0F, 1.0F, 0.0F}),
                    .scale = glm::vec3{grass_scale},
            }
                                               .matrix());
        }
    }

    grass_field_blade_count_ = static_cast<std::uint32_t>(grass_transforms.size());
    auto &field = registry.get<Components::InstancedModel>(grass_field_entity_);
    field.transforms = std::move(grass_transforms);
    field.touch();
}

auto BasicGame::clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void {
    clone_editor_into_runtime<Components::Script, Components::CircularMotion>(editor_scene, runtime_scene);
}

auto BasicGame::on_ui(Scene &scene, Renderer &renderer) -> void {
    poll_helmet(scene, renderer);

    if (!grass_materials_.blades.valid()) {
        return;
    }

    gui::widget("Grass", [&] {
        bool material_changed = false;
        material_changed |= ImGui::ColorEdit3("Colour", &grass_material_info_.base_colour_factor.x);
        material_changed |= ImGui::SliderFloat("Wind strength", &grass_material_info_.wind_strength, 0.0F, 2.0F);

        if (material_changed) {
            auto const result = grass_materials(renderer, engine_models_, grass_material_info_, grass_materials_);

            if (!result) {
                error("Could not update the grass materials: {}", describe(result.error()));
            }
        }

        ImGui::SeparatorText("Field");
        ImGui::SliderFloat("Field size (m)", &grass_field_params_.field_size, 5.0F, 300.0F, "%.1f");
        bool const field_size_committed = ImGui::IsItemDeactivatedAfterEdit();

        ImGui::SliderFloat("Spacing (m)", &grass_field_params_.spacing, 0.2F, 2.5F, "%.2f");
        bool const spacing_committed = ImGui::IsItemDeactivatedAfterEdit();

        ImGui::SeparatorText("Blotchiness");
        ImGui::SliderFloat("Blotch scale (m)", &grass_field_params_.blotch_scale, 2.0F, 80.0F, "%.1f");
        bool const blotch_scale_committed = ImGui::IsItemDeactivatedAfterEdit();

        ImGui::SliderFloat("Blotch threshold", &grass_field_params_.blotch_threshold, 0.0F, 1.0F, "%.2f");
        bool const blotch_threshold_committed = ImGui::IsItemDeactivatedAfterEdit();

        ImGui::SliderFloat("Blotch softness", &grass_field_params_.blotch_softness, 0.0F, 0.5F, "%.2f");
        bool const blotch_softness_committed = ImGui::IsItemDeactivatedAfterEdit();

        bool const any_committed = field_size_committed || spacing_committed || blotch_scale_committed ||
                                   blotch_threshold_committed || blotch_softness_committed;

        static constexpr std::uint32_t max_blade_count = 250'000;
        float const min_spacing_for_size =
                grass_field_params_.field_size / std::sqrt(static_cast<float>(max_blade_count));

        if (field_size_committed) {
            grass_field_params_.spacing = std::max(grass_field_params_.spacing, min_spacing_for_size);
        }
        if (spacing_committed) {
            float const max_size_for_spacing =
                    grass_field_params_.spacing * std::sqrt(static_cast<float>(max_blade_count));
            grass_field_params_.field_size = std::min(grass_field_params_.field_size, max_size_for_spacing);
        }

        auto const grid_candidates =
                static_cast<std::uint32_t>(grass_field_params_.field_size / grass_field_params_.spacing) *
                static_cast<std::uint32_t>(grass_field_params_.field_size / grass_field_params_.spacing);
        ImGui::Text("Blades: %u (of %u candidates)", grass_field_blade_count_, grid_candidates);

        if (any_committed) {
            rebuild_grass_field(scene);
        }
    });
}

auto BasicGame::on_frame_graph(GameGraph &graph, float delta_time) -> void { particles_.declare(graph, delta_time); }

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
        constexpr float jump_velocity = 6.5F;
        physics_world.jump(registry, player_entity_, jump_velocity);
    }

    auto const desired_velocity = player_controller_.desired_horizontal_velocity();
    physics_world.set_velocity(registry, player_entity_, desired_velocity);

    auto const speed_factor = player_controller_.move_speed() > 0.0F
                                      ? glm::length(desired_velocity) / player_controller_.move_speed()
                                      : 0.0F;

    auto const occlusion_query = [&physics_world, player = player_entity_](glm::vec3 const &origin,
                                                                           glm::vec3 const &direction,
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

auto BasicGame::on_key_pressed(Scene & , KeyPressedEvent const &event) -> void {
    player_controller_.on_key_pressed(event.key);
}

auto BasicGame::on_key_released(Scene & , KeyReleasedEvent const &event) -> void {
    player_controller_.on_key_released(event.key);
}

auto BasicGame::on_mouse_moved(Scene & , MouseMovedEvent const &event) -> void {
    player_controller_.on_mouse_moved(static_cast<float>(event.delta_x), static_cast<float>(event.delta_y),
                                      true);
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

[[nodiscard]] auto BasicGame::camera(Scene const & , float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = player_camera_.view(),
            .projection = player_camera_.projection(aspect_ratio),
            .near_clip = player_camera_.near_clip(),
            .far_clip = player_camera_.far_clip(),
            .vertical_fov_radians = glm::radians(player_camera_.field_of_view_degrees()),
    };
}

[[nodiscard]] auto BasicGame::terrain_create_info(Renderer & ) -> std::optional<TerrainWorldCreateInfo> {
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
    constexpr auto player_eye_height = 1.5F;

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
        entity.emplace<Components::BulletTag>();

        scene.physics_world->add_body(scene.get_registry(), entity, transform, rigid_body);
    }
}

auto BasicGame::benchmark_camera_path() const -> std::vector<CameraKeyframe> {
    return {
            {.position = {0.0F, 2.2F, 6.0F}, .target = {0.0F, 2.0F, -10.0F}},
            {.position = {-6.0F, 2.0F, 2.0F}, .target = {-10.0F, 1.5F, -8.0F}},
            {.position = {-16.0F, 3.0F, -4.0F}, .target = {-10.0F, 2.0F, -8.0F}},
            {.position = {-18.0F, 8.0F, -18.0F}, .target = {0.0F, 0.0F, 0.0F}},
            {.position = {0.0F, 2.2F, -19.0F}, .target = {0.0F, 2.0F, -13.0F}},
            {.position = {12.0F, 2.5F, -16.0F}, .target = {9.0F, 2.0F, -10.0F}},
            {.position = {22.0F, 5.0F, -2.0F}, .target = {-20.0F, 0.0F, 0.0F}},
            {.position = {17.0F, 2.2F, 4.0F}, .target = {13.0F, 2.0F, -1.5F}},
            {.position = {15.0F, 3.0F, 16.0F}, .target = {10.0F, 2.0F, 9.0F}},
            {.position = {0.0F, 25.0F, 20.0F}, .target = {0.0F, 0.0F, 0.0F}},
            {.position = {-4.0F, 2.2F, 16.0F}, .target = {-9.0F, 2.0F, 10.0F}},
            {.position = {-18.0F, 2.5F, 6.0F}, .target = {-40.0F, 1.0F, 30.0F}},
            {.position = {-15.0F, 2.2F, -1.0F}, .target = {-13.0F, 2.5F, 1.0F}},
            {.position = {-6.0F, 6.0F, -6.0F}, .target = {6.0F, 0.0F, 6.0F}},
            {.position = {0.0F, 3.0F, 12.0F}, .target = {0.0F, 3.0F, 0.0F}},
    };
}
