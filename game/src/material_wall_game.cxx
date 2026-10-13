#include "material_wall_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {
    constexpr std::size_t shot_count = 4;
    constexpr float spacing = 2.2F;
    constexpr float sphere_scale = 1.0F;

    constexpr std::array<glm::vec3, 8> palette{{
            {0.90F, 0.12F, 0.10F},
            {0.95F, 0.55F, 0.10F},
            {0.95F, 0.85F, 0.15F},
            {0.15F, 0.75F, 0.25F},
            {0.10F, 0.70F, 0.80F},
            {0.15F, 0.30F, 0.90F},
            {0.60F, 0.20F, 0.85F},
            {0.90F, 0.90F, 0.90F},
    }};
}

auto MaterialWallGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    scene.environment.source = EnvironmentSource::procedural_sky;
    scene.environment.draw_skybox = true;
    scene.environment.sun.elevation_degrees = 38.0F;
    scene.environment.sun.intensity = 4.0F;
    scene.environment.exposure_ev = 0.5F;
    scene.environment.specular_intensity = 1.5F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    auto const make_material = [&](glm::vec3 const &colour, float metallic, float roughness,
                                   glm::vec3 const &emissive = glm::vec3{0.0F},
                                   float emissive_strength = 1.0F) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{colour, 1.0F},
                .emissive_factor = emissive,
                .emissive_strength = emissive_strength,
                .metallic_factor = metallic,
                .roughness_factor = roughness,
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = glm::length(emissive) > 0.0F ? images.white() : images.emissive(),
                .sampler = samplers.linear_repeat(),
                .double_sided = true,
        });
        if (!material) {
            error("[MaterialWall] Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        return *material;
    };

    auto const add_prop = [&](std::string const &name, ModelHandle model, glm::vec3 const &position,
                              glm::vec3 const &scale, MaterialHandle material) {
        auto const prop = Entity{&scene, "{}", name};
        prop.emplace<Components::Transform>(Components::Transform{.position = position, .scale = scale});
        prop.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            prop.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
    };

    // Shared backdrop and floor.
    auto const backdrop = make_material({0.45F, 0.45F, 0.47F}, 0.0F, 0.9F);
    add_prop("floor", engine_models.cube, {0.0F, -0.25F, 4.0F}, {40.0F, 0.25F, 20.0F}, backdrop);
    add_prop("backdrop", engine_models.cube, {0.0F, 7.5F, -1.5F}, {40.0F, 16.0F, 0.5F}, backdrop);

    // Key lights in front of the wall so highlights (and so roughness) read on every sphere.
    constexpr std::array<glm::vec3, 3> light_positions{{{-9.0F, 9.0F, 9.0F}, {9.0F, 9.0F, 9.0F}, {0.0F, 14.0F, 7.0F}}};
    for (std::size_t i = 0; i < light_positions.size(); ++i) {
        auto const light = Entity{&scene, "key_light_{}", i};
        light.emplace<Components::Transform>(Components::Transform{.position = light_positions[i]});
        light.emplace<Components::PointLight>(
                Components::PointLight{.colour = {1.0F, 0.95F, 0.9F}, .intensity = 400.0F, .range = 60.0F});
    }

    constexpr int columns = 9;
    constexpr int rows = 3;
    constexpr float left = -(columns - 1) * spacing * 0.5F;

    // Wall 1 (top row): roughness along x, metallic along y rows (0, 0.5, 1). Gold base.
    for (int row = 0; row < rows; ++row) {
        auto const metallic = static_cast<float>(row) / (rows - 1);
        for (int col = 0; col < columns; ++col) {
            auto const roughness = std::max(0.04F, static_cast<float>(col) / (columns - 1));
            add_prop(std::format("rm_{}_{}", row, col), engine_models.sphere,
                     {left + static_cast<float>(col) * spacing, 12.6F - static_cast<float>(row) * spacing, 0.0F},
                     glm::vec3{sphere_scale}, make_material({1.0F, 0.77F, 0.34F}, metallic, roughness));
        }
    }

    // Wall 2: tinted dielectrics (columns) at three roughness levels (rows).
    constexpr std::array<float, 3> dielectric_roughness{0.1F, 0.45F, 0.9F};
    for (std::size_t row = 0; row < dielectric_roughness.size(); ++row) {
        for (int col = 0; col < columns; ++col) {
            auto const &colour = palette[static_cast<std::size_t>(col) % palette.size()];
            add_prop(std::format("dielectric_{}_{}", row, col), engine_models.sphere,
                     {left + static_cast<float>(col) * spacing, 6.0F - static_cast<float>(row) * spacing, 0.0F},
                     glm::vec3{sphere_scale}, make_material(colour, 0.0F, dielectric_roughness[row]));
        }
    }

    // Wall 3 (bottom): emissive colours, strength rising along x. Dark base so only the emission shows.
    for (int col = 0; col < columns; ++col) {
        auto const &colour = palette[static_cast<std::size_t>(col) % palette.size()];
        auto const strength = std::pow(2.0F, static_cast<float>(col) * 0.5F);
        add_prop(std::format("emissive_{}", col), engine_models.cube,
                 {left + static_cast<float>(col) * spacing, 0.4F, 3.0F}, glm::vec3{0.8F},
                 make_material({0.02F, 0.02F, 0.02F}, 0.0F, 0.6F, colour, strength));
    }

    info("[MaterialWall] Keys: 1-4 shot, B bloom");
}

auto MaterialWallGame::on_update(Scene &, float delta_time) -> void {
    time_ += delta_time;

    switch (shot_) {
        case 0: // everything
            eye_ = {std::sin(time_ * 0.1F) * 3.0F, 6.5F, 26.0F};
            look_at_ = {0.0F, 6.0F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        case 1: // roughness x metallic
            eye_ = {0.0F, 11.0F, 14.0F};
            look_at_ = {0.0F, 10.5F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        case 2: // dielectrics
            eye_ = {0.0F, 5.0F, 14.0F};
            look_at_ = {0.0F, 4.5F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        default: // emissive
            eye_ = {0.0F, 2.5F, 14.0F};
            look_at_ = {0.0F, 0.8F, 3.0F};
            fov_degrees_ = 45.0F;
            break;
    }
}

auto MaterialWallGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (event.key >= GLFW_KEY_1 && event.key < GLFW_KEY_1 + static_cast<std::int32_t>(shot_count)) {
        shot_ = static_cast<std::size_t>(event.key - GLFW_KEY_1);
        return;
    }

    if (event.key == GLFW_KEY_B && renderer_ != nullptr) {
        auto bloom = renderer_->bloom_settings();
        bloom.enabled = !bloom.enabled;
        renderer_->set_bloom_settings(bloom);
    }
}

auto MaterialWallGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = glm::lookAtLH(eye_, look_at_, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov_degrees_), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(fov_degrees_),
    };
}
