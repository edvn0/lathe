#pragma once

#include <cstdint>
#include <vector>

#include <volk.h>

#include "game/game.hxx"

// Water bodies (Components::Water, docs/water.md): a tiled pool cut into a stone terrace, with a pillar and spheres standing in the
// water and glass on and in it, next to an open sea that runs out to the horizon over a sloping beach and rocks.
// Shots: 1 the pool from above (refraction, absorption, caustics), 2 across the pool at a grazing angle (reflection,
// foam where things break the surface, shoreline fade), 3 the sea and the beach towards the sun (horizon, sun glint,
// shore foam). Keys: 1-3 shots, P pause the camera orbit, M toggle MSAA.
class WaterGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto subscribe_input(InputContext const &ctx) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    Renderer *renderer_ = nullptr;
    std::vector<MaterialHandle> owned_materials_;

    std::uint32_t frame_ = 0;
    std::uint32_t shot_ = 0;
    bool orbiting_ = true;
    VkSampleCountFlagBits msaa_samples_ = VK_SAMPLE_COUNT_4_BIT;
};
