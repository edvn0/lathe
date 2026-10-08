#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include <optional>

struct SweepHit {
    entt::entity entity{entt::null};
    float fraction{0.0F};
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
};

class CapsuleSweep {
public:
    CapsuleSweep() = default;
    CapsuleSweep(CapsuleSweep const &) = default;
    auto operator=(CapsuleSweep const &) -> CapsuleSweep & = default;
    CapsuleSweep(CapsuleSweep &&) = default;
    auto operator=(CapsuleSweep &&) -> CapsuleSweep & = default;
    virtual ~CapsuleSweep() = default;

    [[nodiscard]] virtual auto sweep_capsule(glm::vec3 const &from, glm::vec3 const &to, float radius, float height,
                                             entt::entity ignore) const -> std::optional<SweepHit> = 0;
};
