#pragma once

#include <cstdint>
#include <vector>

#include "game/game.hxx"

// A dense stress scene for local light shadows: the twenty gallery meshes, fifty copies of each (a thousand shadow
// casters) packed on a grid, lit by more shadow casting spot and point lights than the atlas can hold, with bloom
// tuned down so the shadows read. Needs the gallery downloads (run --game=gallery once); missing ones are skipped.
// Keys: 1-3 camera shots, P pause the camera orbit, C GPU culling, V caching, B bloom.
class ShadowCityGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    Renderer *renderer_ = nullptr;
    std::vector<MaterialHandle> owned_materials_;

    std::uint32_t frame_ = 0;
    std::uint32_t updates_ = 0;
    std::uint32_t shot_ = 0;
    bool orbiting_ = true;
    bool bloom_ = false;
};
