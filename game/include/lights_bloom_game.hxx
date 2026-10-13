#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"

// A dark stage full of point and spot lights over assorted meshes, for judging bloom and colour banding:
// emissive beads and glints feed the bloom, soft spot cones on a matte wall and floor expose 8-bit banding.
// Keys: 1-3 shot, B bloom on/off, -/= bloom intensity, T/G threshold.
class LightsBloomGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Orbiter {
        std::string name;
        float radius = 4.0F;
        float height = 1.0F;
        float speed = 0.5F;
        float phase = 0.0F;
    };

    Renderer *renderer_ = nullptr;
    std::vector<Orbiter> orbiters_;

    std::size_t shot_ = 0;
    float time_ = 0.0F;
    glm::vec3 eye_{0.0F, 3.0F, 11.0F};
    glm::vec3 look_at_{0.0F, 1.0F, 0.0F};
    float fov_degrees_ = 45.0F;
};
