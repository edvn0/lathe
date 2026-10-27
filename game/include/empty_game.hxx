#pragma once

#include "game/game.hxx"

// An empty project: no entities and nothing running, only the editor's grid and axes. What the editor opens on when no
// game or scene is asked for.
class EmptyGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;
};
