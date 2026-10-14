#include "shadow_city_game.hxx"

#include <cmath>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "app/shadow_city.hxx"
#include "core/logger.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"

auto ShadowCityGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;
    populate_shadow_city(scene, renderer, engine_models, ShadowCityOptions{});
    info("[ShadowCity] Keys: 1-3 shots, P pause orbit, C GPU culling, V caching, B bloom");
}

auto ShadowCityGame::on_update(Scene &, float) -> void {
    if (orbiting_) {
        ++frame_;
    }

    if (renderer_ != nullptr && ++updates_ % 60 == 0) {
        auto const &stats = renderer_->local_shadow_stats();
        info("[ShadowCity] lights requested={} shadowed={} tiles={} cached={} cells={} culled={}",
             stats.requested_lights, stats.shadowed_lights, stats.rendered_tiles, stats.cached_lights,
             stats.atlas_cells_used, stats.culled ? 1 : 0);
    }
}

auto ShadowCityGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (renderer_ == nullptr) {
        return;
    }

    auto settings = renderer_->local_shadow_settings();

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
        case GLFW_KEY_C:
            settings.gpu_culling = !settings.gpu_culling;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCity] GPU culling {}", settings.gpu_culling ? "on" : "off");
            break;
        case GLFW_KEY_V:
            settings.caching = !settings.caching;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCity] Caching {}", settings.caching ? "on" : "off");
            break;
        case GLFW_KEY_B: {
            bloom_ = !bloom_;
            auto bloom = renderer_->bloom_settings();
            bloom.enabled = bloom_;
            renderer_->set_bloom_settings(bloom);
            info("[ShadowCity] Bloom {}", bloom_ ? "on" : "off");
            break;
        }
        default:
            break;
    }
}

auto ShadowCityGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    // Shot 1 circles the grid high up, shot 2 walks between the objects, shot 3 looks down the long axis.
    auto const angle = static_cast<float>(frame_) * 0.004F;
    glm::vec3 eye{};
    glm::vec3 target{};

    switch (shot_) {
        case 1:
            eye = {std::cos(angle * 2.0F) * 18.0F, 3.0F, std::sin(angle * 2.0F) * 10.0F};
            target = {eye.x + std::sin(angle * 2.0F) * 6.0F, 1.0F, eye.z - std::cos(angle * 2.0F) * 4.0F};
            break;
        case 2:
            eye = {-70.0F, 9.0F, 0.0F};
            target = {0.0F, 0.5F, 0.0F};
            break;
        default:
            eye = {std::cos(angle) * 48.0F, 24.0F, std::sin(angle) * 34.0F};
            target = {0.0F, 0.0F, 0.0F};
            break;
    }

    return CameraParams{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(55.0F), aspect_ratio, 0.1F, 400.0F),
            .near_clip = 0.1F,
            .far_clip = 400.0F,
            .vertical_fov_radians = glm::radians(55.0F),
    };
}
