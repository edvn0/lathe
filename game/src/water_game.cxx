#include "water_game.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <string>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    // The pool, cut into a terrace whose top is at terrace_top; the water stands a little below the rim.
    constexpr float terrace_top = 1.0F;
    constexpr float terrace_bottom = -3.0F;
    constexpr glm::vec2 pool_min{-5.0F, -14.0F};
    constexpr glm::vec2 pool_max{5.0F, -4.0F};
    constexpr float pool_floor = -1.6F;
    constexpr float pool_water = 0.7F;

    // The sea starts at the terrace's front edge and runs out to the horizon.
    constexpr float sea_level = 0.0F;
    constexpr float sea_floor = -6.0F;
    constexpr float coast_z = 6.0F;

}

auto WaterGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    // Ctrl-R populates the same scene again: drop the last one and the materials it made first.
    scene.get_registry().clear();
    if (!owned_materials_.empty()) {
        if (auto const idle = renderer.wait_idle(); !idle) {
            warn("{}", describe(idle.error()));
        }
        for (auto const material: owned_materials_) {
            renderer.release_material(material);
        }
        owned_materials_.clear();
    }

    scene.environment.source = EnvironmentSource::procedural_sky;
    scene.environment.draw_skybox = true;
    // Low and out over the sea (+z), so shot 3 looks into the sun's glint and the pool is lit from behind the camera.
    scene.environment.sun.elevation_degrees = 18.0F;
    scene.environment.sun.azimuth_degrees = 80.0F;
    scene.environment.sun.intensity = 0.15F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    struct Look {
        glm::vec4 colour;
        float metallic = 0.0F;
        float roughness = 0.5F;
        AlphaMode alpha_mode = AlphaMode::opaque;
        bool double_sided = false;
    };

    auto const make_material = [&](Look const &look) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = look.colour,
                .metallic_factor = look.metallic,
                .roughness_factor = look.roughness,
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = images.emissive(),
                .sampler = samplers.linear_repeat(),
                .alpha_mode = look.alpha_mode,
                .double_sided = look.double_sided,
        });
        if (!material) {
            error("[Water] Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        owned_materials_.push_back(*material);
        return *material;
    };

    auto const opaque = [&](glm::vec3 const &colour, float roughness = 0.7F, float metallic = 0.0F) {
        return make_material(Look{.colour = {colour, 1.0F}, .metallic = metallic, .roughness = roughness});
    };
    auto const glass = [&](glm::vec4 const &colour) {
        return make_material(
                Look{.colour = colour, .roughness = 0.03F, .alpha_mode = AlphaMode::blend, .double_sided = true});
    };

    auto const add_prop = [&](std::string const &name, ModelHandle model, glm::vec3 const &position,
                              glm::vec3 const &scale, MaterialHandle material,
                              glm::quat const &rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F}) -> Entity {
        auto const prop = Entity{&scene, "{}", name};
        prop.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .rotation = rotation,
                .scale = scale,
        });
        prop.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            prop.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
        return prop;
    };

    // An axis-aligned box between two corners (the engine cube is a unit cube about its centre).
    auto const add_box = [&](std::string const &name, glm::vec3 const &low, glm::vec3 const &high,
                             MaterialHandle material) {
        return add_prop(name, engine_models.cube, (low + high) * 0.5F, high - low, material);
    };

    // The terrace around the pool: four slabs leave the pool's hole open.
    auto const stone = opaque({0.62F, 0.6F, 0.56F}, 0.85F);
    add_box("terrace_south", {-20.0F, terrace_bottom, pool_max.y}, {20.0F, terrace_top, coast_z}, stone);
    add_box("terrace_north", {-20.0F, terrace_bottom, -26.0F}, {20.0F, terrace_top, pool_min.y}, stone);
    add_box("terrace_west", {-20.0F, terrace_bottom, pool_min.y}, {pool_min.x, terrace_top, pool_max.y}, stone);
    add_box("terrace_east", {pool_max.x, terrace_bottom, pool_min.y}, {20.0F, terrace_top, pool_max.y}, stone);

    // Grassland around the terrace, kept clear of the pool's hole and of the sea.
    auto const grass = opaque({0.22F, 0.36F, 0.14F}, 0.95F);
    add_box("land_west", {-400.0F, -3.5F, -400.0F}, {-20.0F, 0.8F, coast_z}, grass);
    add_box("land_east", {20.0F, -3.5F, -400.0F}, {400.0F, 0.8F, coast_z}, grass);
    add_box("land_north", {-20.0F, -3.5F, -400.0F}, {20.0F, 0.8F, -26.0F}, grass);

    // Tiles: a grout bed on the pool floor and tiles on it and on the four walls, in a two-colour mosaic with a dark
    // lane line down the middle so the refraction shows.
    auto const grout = opaque({0.18F, 0.2F, 0.22F}, 0.9F);
    add_box("pool_bed", {pool_min.x, pool_floor - 0.6F, pool_min.y}, {pool_max.x, pool_floor, pool_max.y}, grout);

    std::array const tile_looks{
            opaque({0.9F, 0.93F, 0.95F}, 0.25F),
            opaque({0.45F, 0.75F, 0.85F}, 0.25F),
            opaque({0.05F, 0.12F, 0.35F}, 0.25F),
    };
    constexpr float tile_gap = 0.06F;
    constexpr float tile_thickness = 0.05F;

    for (int row = 0; row < 10; ++row) {
        for (int column = 0; column < 10; ++column) {
            auto const lane = column == 4 && row > 0 && row < 9;
            auto const look = lane ? 2U : static_cast<std::size_t>((row + column) % 2);
            auto const x = pool_min.x + 0.5F + static_cast<float>(column);
            auto const z = pool_min.y + 0.5F + static_cast<float>(row);
            add_prop(std::format("floor_tile_{}_{}", row, column), engine_models.cube,
                     {x, pool_floor + tile_thickness * 0.5F, z}, {1.0F - tile_gap, tile_thickness, 1.0F - tile_gap},
                     tile_looks[look]);
        }
    }

    constexpr int wall_rows = 3;
    auto const row_pitch = (terrace_top - pool_floor) / static_cast<float>(wall_rows);
    for (int row = 0; row < wall_rows; ++row) {
        auto const y = pool_floor + row_pitch * (static_cast<float>(row) + 0.5F);
        auto const height = row_pitch - tile_gap;
        for (int i = 0; i < 10; ++i) {
            auto const along = static_cast<float>(i) + 0.5F;
            auto const look = tile_looks[static_cast<std::size_t>((row + i) % 2)];
            auto const name = [&](char const *wall) { return std::format("wall_tile_{}_{}_{}", wall, row, i); };

            add_prop(name("s"), engine_models.cube, {pool_min.x + along, y, pool_max.y - tile_thickness * 0.5F},
                     {1.0F - tile_gap, height, tile_thickness}, look);
            add_prop(name("n"), engine_models.cube, {pool_min.x + along, y, pool_min.y + tile_thickness * 0.5F},
                     {1.0F - tile_gap, height, tile_thickness}, look);
            add_prop(name("w"), engine_models.cube, {pool_min.x + tile_thickness * 0.5F, y, pool_min.y + along},
                     {tile_thickness, height, 1.0F - tile_gap}, look);
            add_prop(name("e"), engine_models.cube, {pool_max.x - tile_thickness * 0.5F, y, pool_min.y + along},
                     {tile_thickness, height, 1.0F - tile_gap}, look);
        }
    }

    // In the pool: a pillar through the surface, a ramp that climbs out of the water at the west end (a shoreline to
    // fade and foam along), spheres half in the water and one on the floor, and a glass ball floating in it.
    add_box("pillar", {1.8F, pool_floor, -8.2F}, {2.6F, 2.6F, -7.4F}, opaque({0.85F, 0.82F, 0.76F}, 0.4F));
    add_prop("ramp", engine_models.cube, {-3.4F, 0.0F, -10.5F}, {2.8F, 0.3F, 6.5F},
             opaque({0.8F, 0.72F, 0.55F}, 0.8F), glm::angleAxis(glm::radians(-24.0F), glm::vec3{1.0F, 0.0F, 0.0F}));
    add_prop("ball_orange", engine_models.sphere, {-0.6F, pool_water - 0.1F, -6.4F}, glm::vec3{1.3F},
             opaque({0.95F, 0.45F, 0.1F}, 0.35F));
    add_prop("ball_green", engine_models.sphere, {3.4F, pool_water + 0.15F, -11.5F}, glm::vec3{0.9F},
             opaque({0.2F, 0.75F, 0.3F}, 0.35F));
    add_prop("ball_floor", engine_models.sphere, {0.5F, pool_floor + 0.6F, -10.5F}, glm::vec3{1.2F},
             opaque({0.95F, 0.8F, 0.25F}, 0.25F, 1.0F));
    add_prop("glass_float", engine_models.sphere, {-1.2F, pool_water + 0.1F, -11.8F}, glm::vec3{0.9F},
             glass({0.6F, 0.9F, 1.0F, 0.25F}));

    // Glass on the terrace by the pool.
    add_prop("glass_ball", engine_models.sphere, {6.8F, terrace_top + 0.6F, -3.0F}, glm::vec3{1.2F},
             glass({0.3F, 0.5F, 0.95F, 0.35F}));
    add_prop("glass_block", engine_models.cube, {-7.0F, terrace_top + 0.5F, -6.0F}, glm::vec3{1.0F},
             glass({0.9F, 0.95F, 0.85F, 0.25F}),
             glm::angleAxis(glm::radians(20.0F), glm::vec3{0.0F, 1.0F, 0.0F}));
    add_prop("glass_pane", engine_models.cube, {-7.5F, terrace_top + 0.9F, -10.0F}, {0.05F, 1.8F, 3.0F},
             glass({0.95F, 0.3F, 0.25F, 0.35F}));

    // The water bodies are components: the engine's WaterSystem makes their meshes and materials. The pool is a calm
    // lake filling the pool's hole a little below the rim.
    {
        auto const pool = Entity{&scene, "pool_water"};
        pool.emplace<Components::Transform>(Components::Transform{
                .position = {(pool_min.x + pool_max.x) * 0.5F, pool_water, (pool_min.y + pool_max.y) * 0.5F}});
        pool.emplace<Components::Water>(Components::Water::pool(pool_max - pool_min));
    }

    // The sea: a sloping beach to the east, rocks and a buoy, and the sea floor below.
    add_box("sea_floor", {-400.0F, sea_floor - 1.0F, coast_z}, {400.0F, sea_floor, 600.0F}, opaque({0.55F, 0.5F, 0.38F}, 0.9F));
    add_prop("beach", engine_models.cube, {50.0F, -2.0F, 25.0F}, {60.0F, 1.0F, 42.0F},
             opaque({0.86F, 0.78F, 0.58F}, 0.9F), glm::angleAxis(glm::radians(11.0F), glm::vec3{1.0F, 0.0F, 0.0F}));
    auto const rock = opaque({0.35F, 0.34F, 0.33F}, 0.8F);
    add_prop("rock_0", engine_models.sphere, {-10.0F, -0.4F, 22.0F}, {6.0F, 3.0F, 5.0F}, rock);
    add_prop("rock_1", engine_models.sphere, {-26.0F, -0.8F, 45.0F}, {9.0F, 4.0F, 7.0F}, rock);
    add_prop("rock_2", engine_models.sphere, {14.0F, -0.3F, 38.0F}, {4.0F, 2.5F, 4.0F}, rock);
    add_prop("buoy", engine_models.sphere, {4.0F, sea_level + 0.1F, 16.0F}, glm::vec3{1.0F},
             opaque({0.9F, 0.12F, 0.08F}, 0.4F));

    // The sea: an ocean out to the horizon, whose water zone keeps it off the land (it starts at the coast).
    {
        constexpr float sea_near = coast_z + 0.5F;
        constexpr float sea_far = 600.0F;
        auto const sea = Entity{&scene, "sea"};
        sea.emplace<Components::Transform>(
                Components::Transform{.position = {8.0F, sea_level, (sea_near + sea_far) * 0.5F}});
        sea.emplace<Components::Water>(Components::Water::ocean({800.0F, sea_far - sea_near}));
    }

    constexpr std::array<glm::vec3, 2> light_positions{{{-7.0F, 8.0F, -1.0F}, {7.0F, 8.0F, -16.0F}}};
    constexpr std::array<glm::vec3, 2> light_targets{{{-1.0F, 0.0F, -9.0F}, {1.0F, 0.0F, -8.0F}}};
    for (std::size_t i = 0; i < light_positions.size(); ++i) {
        auto const light = Entity{&scene, "pool_lamp_{}", i};
        light.emplace<Components::Transform>(Components::Transform{
                .position = light_positions[i],
                .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F},
                                          glm::normalize(light_targets[i] - light_positions[i])),
        });
        light.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = {1.0F, 0.95F, 0.85F},
                .intensity = 500.0F,
                .range = 40.0F,
                .inner_cone_degrees = 25.0F,
                .outer_cone_degrees = 45.0F,
                .casts_shadows = true,
        });
    }

    info("[Water] Keys: 1-3 shots, P pause orbit, M toggle MSAA");
}

auto WaterGame::on_update(Scene &, float) -> void {
    if (orbiting_) {
        ++frame_;
    }
}

auto WaterGame::subscribe_input(InputContext const &ctx) -> void {
    listen(ctx, &WaterGame::on_key_pressed);
}

auto WaterGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    switch (event.key) {
        case GLFW_KEY_1:
            shot_ = 0;
            break;
        case GLFW_KEY_2:
            shot_ = 1;
            break;
        case GLFW_KEY_3:
            shot_ = 2;
            break;
        case GLFW_KEY_P:
            orbiting_ = !orbiting_;
            break;
        case GLFW_KEY_M:
            // MSAA off and back on: the scene copy takes a different route without it (docs/water.md).
            if (renderer_ != nullptr) {
                if (renderer_->samples() != VK_SAMPLE_COUNT_1_BIT) {
                    msaa_samples_ = renderer_->samples();
                    renderer_->set_samples(VK_SAMPLE_COUNT_1_BIT);
                } else {
                    renderer_->set_samples(msaa_samples_);
                }
            }
            break;
        default:
            break;
    }
}

auto WaterGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    // Shot 1 looks down into the pool, shot 2 across it at a grazing angle, shot 3 out to sea.
    auto const angle = static_cast<float>(frame_) * 0.006F;
    glm::vec3 eye{};
    glm::vec3 target{};

    switch (shot_) {
        case 1:
            eye = {-1.5F + std::sin(angle) * 2.0F, 2.0F, -2.2F};
            target = {1.0F, 0.4F, -11.0F};
            break;
        case 2:
            eye = {6.0F + std::sin(angle) * 4.0F, 4.0F, 2.0F};
            target = {16.0F, 0.0F, 40.0F};
            break;
        default:
            eye = {std::sin(angle) * 3.0F, 8.0F, 2.0F};
            target = {0.0F, -1.0F, -9.0F};
            break;
    }

    return CameraParams{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(55.0F), aspect_ratio, 0.1F, 1000.0F),
            .near_clip = 0.1F,
            .far_clip = 1000.0F,
            .vertical_fov_radians = glm::radians(55.0F),
    };
}
