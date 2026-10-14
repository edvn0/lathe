#include "shadow_caster_game.hxx"

#include <array>
#include <cmath>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {
    constexpr std::string_view human_model_path = "assets/models/animated_human.glb";
    // Measured from the model: its height and the height of its feet, as the moving game uses them.
    constexpr float human_height = 5.535F;
    constexpr float human_feet_y = -0.015F;
    constexpr float human_target_height = 1.8F;

    constexpr std::array<glm::vec3, 3> human_positions{{{5.5F, 0.0F, 1.0F}, {7.5F, 0.0F, 0.0F}, {6.5F, 0.0F, -2.2F}}};

    constexpr glm::vec3 orbit_centre{0.5F, 0.4F, -3.0F};
}

auto ShadowCasterGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    scene.environment.source = EnvironmentSource::procedural_sky;
    scene.environment.draw_skybox = true;
    scene.environment.sun.elevation_degrees = 20.0F;
    scene.environment.sun.intensity = 0.15F;
    scene.environment.exposure_ev = 0.0F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    auto const make_material = [&](glm::vec3 const &colour, float roughness, bool double_sided = false,
                                   bool cutout = false) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{colour, 1.0F},
                .metallic_factor = 0.0F,
                .roughness_factor = roughness,
                .alpha_cutoff = 0.5F,
                .base_colour_texture = cutout ? engine_models.grass_card_texture : images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = images.emissive(),
                .sampler = cutout ? samplers.linear_clamp() : samplers.linear_repeat(),
                .alpha_mode = cutout ? AlphaMode::mask : AlphaMode::opaque,
                .double_sided = double_sided,
        });
        if (!material) {
            error("[ShadowCaster] Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        return *material;
    };

    auto const add_prop = [&](std::string const &name, ModelHandle model, glm::vec3 const &position,
                              float yaw_degrees, glm::vec3 const &scale, MaterialHandle material) -> Entity {
        auto const prop = Entity{&scene, "{}", name};
        prop.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .rotation = glm::angleAxis(glm::radians(yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}),
                .scale = scale,
        });
        prop.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            prop.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
        return prop;
    };

    add_prop("floor", engine_models.cube, {0.0F, -0.25F, 0.0F}, 0.0F, {30.0F, 0.25F, 30.0F},
             make_material({0.6F, 0.6F, 0.62F}, 0.8F));

    // Opaque casters.
    auto const stone = make_material({0.75F, 0.75F, 0.78F}, 0.6F);
    add_prop("pillar_a", engine_models.cube, {-6.0F, 1.2F, 0.0F}, 0.0F, {0.4F, 1.2F, 0.4F}, stone);
    add_prop("pillar_b", engine_models.cube, {-4.6F, 0.7F, 2.2F}, 20.0F, {0.4F, 0.7F, 0.4F}, stone);

    // A thin panel drawn double sided, so its shadow comes from the double-sided batch.
    add_prop("panel_double_sided", engine_models.cube, {-2.2F, 1.4F, 0.0F}, 30.0F, {1.1F, 1.1F, 0.02F},
             make_material({0.85F, 0.45F, 0.3F}, 0.5F, true));

    // An alpha-masked cutout (grass blades on a flat card), so its shadow is the blades, not the card.
    add_prop("panel_cutout", engine_models.cube, {1.4F, 1.4F, 0.0F}, -20.0F, {1.3F, 1.3F, 0.02F},
             make_material({0.4F, 0.8F, 0.35F}, 0.7F, true, true));

    struct Lamp {
        glm::vec3 position;
        glm::vec3 target;
        glm::vec3 colour;
        bool spot;
    };

    constexpr std::array<Lamp, 3> lamps{{
            {{-9.0F, 8.0F, 9.0F}, {-1.0F, 0.0F, 0.0F}, {1.0F, 0.75F, 0.5F}, true},
            {{9.0F, 7.0F, 8.0F}, {3.0F, 0.0F, -1.0F}, {0.5F, 0.7F, 1.0F}, true},
            {{1.0F, 3.2F, 3.5F}, {0.0F, 0.0F, 0.0F}, {1.0F, 0.9F, 0.7F}, false},
    }};
    for (std::size_t i = 0; i < lamps.size(); ++i) {
        auto const light = Entity{&scene, "shadow_lamp_{}", i};
        auto const direction = glm::normalize(lamps[i].target - lamps[i].position);
        light.emplace<Components::Transform>(Components::Transform{
                .position = lamps[i].position,
                .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F}, direction),
        });
        if (lamps[i].spot) {
            light.emplace<Components::SpotLight>(Components::SpotLight{
                    .colour = lamps[i].colour,
                    .intensity = 2600.0F,
                    .range = 40.0F,
                    .inner_cone_degrees = 18.0F,
                    .outer_cone_degrees = 32.0F,
                    .casts_shadows = true,
            });
        } else {
            light.emplace<Components::PointLight>(Components::PointLight{
                    .colour = lamps[i].colour, .intensity = 150.0F, .range = 16.0F, .casts_shadows = true});
        }
    }

    orbiter_ = static_cast<entt::entity>(add_prop("orbiter", engine_models.cube, orbit_centre, 0.0F,
                                                  glm::vec3{0.5F}, make_material({0.9F, 0.9F, 0.9F}, 0.5F)));

    // Skinned characters in the bind pose: an identity palette leaves every joint where the model rests, and still
    // sends the batch down the GPU skinning path with its own scratch vertices.
    if (auto loaded = renderer.load_model(data_path(human_model_path)); loaded) {
        if (auto const skin = renderer.model_animation(*loaded)) {
            joint_count_ = static_cast<std::uint32_t>(skin->skeleton.joint_count());
            auto const humans = GeneratedEntity{&scene, "{}", "skinned_humans"};
            humans.emplace<Components::InstancedModel>(Components::InstancedModel{.model = *loaded});
            humans_ = static_cast<entt::entity>(humans);
            human_count_ = std::min<std::uint32_t>(static_cast<std::uint32_t>(human_positions.size()),
                                                   std::max(renderer.max_skinned_instances(*loaded), 1U));
        } else {
            warn("[ShadowCaster] '{}' has no skin; the scene has no skinned casters", human_model_path);
        }
    } else {
        warn("[ShadowCaster] Could not load '{}': {}", human_model_path, describe(loaded.error()));
    }

    info("[ShadowCaster] Keys: C GPU culling, V caching, O orbiter, K skinned characters");
}

auto ShadowCasterGame::on_update(Scene &scene, float) -> void {
    auto &registry = scene.get_registry();

    if (orbiting_) {
        ++frame_;
    }

    if (auto *transform = registry.try_get<Components::Transform>(orbiter_)) {
        auto const angle = static_cast<float>(frame_) * 0.05F;
        transform->position = orbit_centre + glm::vec3{std::cos(angle) * 1.6F, 0.0F, std::sin(angle) * 1.6F};
    }

    if (renderer_ != nullptr && ++updates_ % 10 == 0) {
        auto const &stats = renderer_->local_shadow_stats();
        info("[ShadowCaster] stats lights={} cached={} tiles={} culled={}", stats.shadowed_lights, stats.cached_lights,
             stats.rendered_tiles, stats.culled ? 1 : 0);
    }

    if (renderer_ == nullptr || !registry.valid(humans_)) {
        return;
    }

    auto &humans = registry.get<Components::InstancedModel>(humans_);
    humans.transforms.clear();
    humans.palette_offsets.clear();
    static_cast<void>(renderer_->set_skin_palette({}));

    if (show_humans_) {
        std::vector<glm::mat4> const rest_pose(joint_count_, glm::mat4{1.0F});
        auto const fit = glm::translate(glm::scale(glm::mat4{1.0F}, glm::vec3{human_target_height / human_height}),
                                        {0.0F, -human_feet_y, 0.0F});

        for (std::uint32_t i = 0; i < human_count_; ++i) {
            auto const offset = renderer_->append_skin_palette(rest_pose);
            if (!offset) {
                break;
            }

            auto const yaw = glm::rotate(glm::mat4{1.0F}, glm::radians(-60.0F + 25.0F * static_cast<float>(i)),
                                         glm::vec3{0.0F, 1.0F, 0.0F});
            humans.transforms.push_back(glm::translate(glm::mat4{1.0F}, human_positions[i]) * yaw * fit);
            humans.palette_offsets.push_back(*offset);
        }
    }

    humans.touch();
}

auto ShadowCasterGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (renderer_ == nullptr) {
        return;
    }

    auto settings = renderer_->local_shadow_settings();

    switch (event.key) {
        case GLFW_KEY_C:
            settings.gpu_culling = !settings.gpu_culling;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCaster] GPU culling {}", settings.gpu_culling ? "on" : "off");
            break;
        case GLFW_KEY_V:
            settings.caching = !settings.caching;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCaster] Caching {}", settings.caching ? "on" : "off");
            break;
        case GLFW_KEY_O:
            orbiting_ = !orbiting_;
            info("[ShadowCaster] Orbiter {}", orbiting_ ? "moving" : "stopped");
            break;
        case GLFW_KEY_K:
            show_humans_ = !show_humans_;
            info("[ShadowCaster] Skinned characters {}", show_humans_ ? "shown" : "hidden");
            break;
        default:
            break;
    }
}

auto ShadowCasterGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    constexpr glm::vec3 eye{0.5F, 7.5F, 15.0F};
    constexpr glm::vec3 target{0.5F, 0.4F, -0.5F};

    return CameraParams{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(48.0F), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(48.0F),
    };
}
