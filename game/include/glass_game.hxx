#pragma once

#include <cstdint>
#include <vector>

#include "game/game.hxx"

// A gallery of alpha-blended materials: clear, tinted and frosted glass, coloured gels, a glowing bulb inside a
// bottle, and layered panes that only sort correctly back to front. Everything sits on an opaque checker floor
// with opaque props behind the glass so the blending is easy to see.
// Keys: 1-3 camera shots, P pause the camera orbit.
class GlassGame final : public IGame {
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
};
