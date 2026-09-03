#pragma once

#include "rendering/script.hxx"

#include <glm/vec3.hpp>

namespace Components {
    // Per-entity orbit state for EnemyAIScript. `angle` is mutated in place
    // each frame via ScriptEntity::get<CircularMotion>() (never replaced), so
    // this stays safe to update concurrently across every entity sharing one
    // EnemyAIScript instance -- see EnemyAIScript::parallelizable().
    struct CircularMotion {
        glm::vec3 center{0.0F};
        float radius = 3.0F;
        float angular_speed = 1.0F; // radians/second
        float angle = 0.0F; // current phase
    };
} // namespace Components

// One shared instance drives every entity whose Components::Script points at
// it (see BasicGame::on_populate) -- it holds no per-entity state itself,
// only Components::CircularMotion does, which is what makes on_update safe to
// run concurrently across entities.
class EnemyAIScript final : public IScript {
public:
    auto on_update(ScriptEntity entity, float delta_time) -> void override;

    [[nodiscard]] auto parallelizable() const noexcept -> bool override { return true; }
};
