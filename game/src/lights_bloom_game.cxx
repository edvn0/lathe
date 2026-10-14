#include "lights_bloom_game.hxx"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    // Spot lights point along local -Y.
    auto aim(glm::vec3 const &from, glm::vec3 const &to) -> glm::quat {
        constexpr auto local_forward = glm::vec3{0.0F, -1.0F, 0.0F};
        auto const direction = glm::normalize(to - from);
        auto const dot = glm::dot(local_forward, direction);

        if (dot > 0.9999F) {
            return glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        }
        if (dot < -0.9999F) {
            return glm::angleAxis(std::numbers::pi_v<float>, glm::vec3{1.0F, 0.0F, 0.0F});
        }
        return glm::angleAxis(std::acos(dot), glm::normalize(glm::cross(local_forward, direction)));
    }

    auto hue(float h) -> glm::vec3 {
        auto const channel = [h](float offset) {
            return std::clamp(std::abs(std::fmod(h * 6.0F + offset, 6.0F) - 3.0F) - 1.0F, 0.0F, 1.0F);
        };
        return {channel(0.0F), channel(4.0F), channel(2.0F)};
    }

    constexpr std::size_t shot_count = 3;
}

auto LightsBloomGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;
    orbiters_.clear();

    scene.environment.source = EnvironmentSource::flat_ambient;
    scene.environment.draw_skybox = false;
    scene.environment.ambient_intensity = 0.02F;
    scene.environment.exposure_ev = 0.0F;
    scene.environment.sun.intensity = 0.0F;

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
                .metallic_roughness_texture = images.metallic_roughness(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = glm::length(emissive) > 0.0F ? images.white() : images.emissive(),
                .sampler = samplers.linear_repeat(),
                .double_sided = true,
        });
        if (!material) {
            error("[LightsBloom] Could not create material: {}", describe(material.error()));
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

    // Matte grey floor and back wall: low-contrast gradients under the spot cones are where banding shows.
    auto const matte = make_material({0.55F, 0.55F, 0.58F}, 0.0F, 0.9F);
    add_prop("floor", engine_models.cube, {0.0F, -0.25F, 0.0F}, {24.0F, 0.25F, 24.0F}, matte);
    add_prop("back_wall", engine_models.cube, {0.0F, 6.0F, -9.0F}, {24.0F, 6.0F, 0.25F}, matte);

    // Sphere row spanning roughness (bottom) and metal/dielectric: specular glints are the firefly candidates.
    constexpr int row_count = 7;
    for (int i = 0; i < row_count; ++i) {
        auto const t = static_cast<float>(i) / (row_count - 1);
        auto const x = (t - 0.5F) * 12.0F;
        add_prop(std::format("metal_{}", i), engine_models.sphere, {x, 0.8F, -3.0F}, glm::vec3{0.8F},
                 make_material({0.95F, 0.78F, 0.45F}, 1.0F, std::max(0.04F, t)));
        add_prop(std::format("plastic_{}", i), engine_models.sphere, {x, 0.8F, -5.5F}, glm::vec3{0.8F},
                 make_material(hue(t) * 0.8F + 0.1F, 0.0F, std::max(0.04F, t)));
    }

    add_prop("pillar_left", engine_models.cube, {-4.5F, 1.5F, 1.5F}, {0.4F, 1.5F, 0.4F},
             make_material({0.8F, 0.8F, 0.8F}, 0.0F, 0.5F));
    add_prop("pillar_right", engine_models.cube, {4.5F, 1.5F, 1.5F}, {0.4F, 1.5F, 0.4F},
             make_material({0.8F, 0.8F, 0.8F}, 0.0F, 0.5F));
    add_prop("capsule", engine_models.capsule, {0.0F, 1.0F, 1.5F}, glm::vec3{1.0F},
             make_material({0.9F, 0.9F, 0.9F}, 0.0F, 0.3F));

    // Orbiting point lights, each with a small emissive bead; the bead's HDR value is what drives bloom.
    constexpr int orbiter_count = 14;
    for (int i = 0; i < orbiter_count; ++i) {
        auto const t = static_cast<float>(i) / orbiter_count;
        auto const colour = hue(t);
        auto const name = std::format("orbiter_{}", i);

        auto const bead = Entity{&scene, "{}", name};
        bead.emplace<Components::Transform>(Components::Transform{.position = {0.0F, 1.0F, 0.0F}, .scale = glm::vec3{0.12F}});
        bead.emplace<Components::Model>(Components::Model{.model = engine_models.sphere});
        bead.emplace<Components::MaterialOverride>(Components::MaterialOverride{
                .material = make_material({0.02F, 0.02F, 0.02F}, 0.0F, 1.0F, colour, 6.0F + 6.0F * static_cast<float>(i % 3))});
        bead.emplace<Components::PointLight>(Components::PointLight{
                .colour = colour,
                .intensity = 5.0F,
                .range = 7.0F,
        });

        orbiters_.push_back(Orbiter{
                .name = name,
                .radius = 3.0F + 0.45F * static_cast<float>(i % 7),
                .height = 0.6F + 0.35F * static_cast<float>(i % 5),
                .speed = (i % 2 == 0 ? 0.35F : -0.25F) * (1.0F + 0.1F * static_cast<float>(i % 4)),
                .phase = t * 2.0F * std::numbers::pi_v<float>,
        });
    }

    // Spot lights from the ceiling: wide soft penumbras on the wall and floor.
    constexpr int spot_count = 5;
    for (int i = 0; i < spot_count; ++i) {
        auto const t = static_cast<float>(i) / (spot_count - 1);
        auto const x = (t - 0.5F) * 14.0F;
        auto const from = glm::vec3{x, 9.0F, 3.0F};
        auto const to = glm::vec3{x * 0.8F, 3.0F, -9.0F};

        auto const spot = Entity{&scene, "spot_{}", i};
        spot.emplace<Components::Transform>(Components::Transform{.position = from, .rotation = aim(from, to)});
        spot.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = i % 2 == 0 ? glm::vec3{1.0F, 0.85F, 0.65F} : glm::vec3{0.6F, 0.75F, 1.0F},
                .intensity = 100.0F,
                .range = 30.0F,
                .inner_cone_degrees = 4.0F,
                .outer_cone_degrees = 22.0F,
        });
    }

    // Two dim floor-level spots grazing the floor: long smooth gradients.
    for (int i = 0; i < 2; ++i) {
        auto const side = i == 0 ? -1.0F : 1.0F;
        auto const from = glm::vec3{side * 10.0F, 0.6F, 6.0F};
        auto const to = glm::vec3{-side * 2.0F, 0.0F, -4.0F};

        auto const spot = Entity{&scene, "graze_{}", i};
        spot.emplace<Components::Transform>(Components::Transform{.position = from, .rotation = aim(from, to)});
        spot.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = {1.0F, 0.9F, 0.8F},
                .intensity = 25.0F,
                .range = 30.0F,
                .inner_cone_degrees = 2.0F,
                .outer_cone_degrees = 35.0F,
        });
    }

    info("[LightsBloom] Keys: 1-3 shot, B bloom, -/= intensity, T/G threshold");
}

auto LightsBloomGame::on_update(Scene &scene, float delta_time) -> void {
    time_ += delta_time;

    for (auto const &orbiter: orbiters_) {
        auto const entity = scene.find_entity(orbiter.name);
        if (auto *transform = scene.get_registry().try_get<Components::Transform>(entity)) {
            auto const angle = orbiter.phase + time_ * orbiter.speed;
            transform->position = {std::cos(angle) * orbiter.radius, orbiter.height + 0.2F * std::sin(time_ + orbiter.phase),
                                   std::sin(angle) * orbiter.radius - 2.0F};
        }
    }

    switch (shot_) {
        case 0: {
            auto const yaw = std::sin(time_ * 0.15F) * 0.35F;
            eye_ = {std::sin(yaw) * 12.0F, 3.0F, std::cos(yaw) * 12.0F};
            look_at_ = {0.0F, 1.2F, -3.0F};
            fov_degrees_ = 45.0F;
            break;
        }
        case 1:
            eye_ = {-2.5F, 1.4F, 4.0F};
            look_at_ = {1.0F, 1.0F, -4.0F};
            fov_degrees_ = 40.0F;
            break;
        default: // wall gradient close-up for banding
            eye_ = {0.0F, 4.0F, 3.0F};
            look_at_ = {0.0F, 4.0F, -9.0F};
            fov_degrees_ = 55.0F;
            break;
    }
}

auto LightsBloomGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (event.key >= GLFW_KEY_1 && event.key < GLFW_KEY_1 + static_cast<std::int32_t>(shot_count)) {
        shot_ = static_cast<std::size_t>(event.key - GLFW_KEY_1);
        return;
    }

    if (renderer_ == nullptr) {
        return;
    }

    auto bloom = renderer_->bloom_settings();
    switch (event.key) {
        case GLFW_KEY_B: bloom.enabled = !bloom.enabled; break;
        case GLFW_KEY_MINUS: bloom.intensity = std::max(0.0F, bloom.intensity - 0.02F); break;
        case GLFW_KEY_EQUAL: bloom.intensity = std::min(1.0F, bloom.intensity + 0.02F); break;
        case GLFW_KEY_T: bloom.threshold += 0.25F; break;
        case GLFW_KEY_G: bloom.threshold = std::max(0.0F, bloom.threshold - 0.25F); break;
        default: return;
    }
    renderer_->set_bloom_settings(bloom);
    info("[LightsBloom] bloom {} intensity {:.2f} threshold {:.2f}", bloom.enabled ? "on" : "off", bloom.intensity,
         bloom.threshold);
}

auto LightsBloomGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = glm::lookAtLH(eye_, look_at_, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov_degrees_), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(fov_degrees_),
    };
}
