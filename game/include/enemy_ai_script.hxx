#pragma once

#include "rendering/script.hxx"

#include <glm/vec3.hpp>

namespace Components {
    struct CircularMotion {
        glm::vec3 center{0.0F};
        float radius = 3.0F;
        float angular_speed = 1.0F;
        float angle = 0.0F;
    };
}

class EnemyAIScript final : public IScript {
public:
    auto on_update(ScriptEntity entity, float delta_time) -> void override;

    [[nodiscard]] auto parallelizable() const noexcept -> bool override { return true; }
};
