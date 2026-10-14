#pragma once

#include <cstddef>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"

// A fixed, deterministic scene for checking local light shadows against every kind of caster: opaque pillars, a
// thin double-sided panel, an alpha-masked cutout, skinned characters and a cube circling inside the lights. Frames,
// not time, drive the motion, so two runs screenshot the same image and can be compared pixel for pixel.
// Keys: C GPU culling, V caching, O stop/start the orbiter, K skinned characters on/off.
class ShadowCasterGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    Renderer *renderer_ = nullptr;

    entt::entity orbiter_ = entt::null;
    entt::entity humans_ = entt::null;
    std::uint32_t joint_count_ = 0;
    std::uint32_t human_count_ = 0;

    std::uint32_t frame_ = 0;
    std::uint32_t updates_ = 0;
    bool orbiting_ = true;
    bool show_humans_ = true;
};
