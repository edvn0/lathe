#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include <optional>

struct SweepHit {
    entt::entity entity{entt::null};
    float fraction{0.0F}; // 0..1 along the sweep
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F, 1.0F, 0.0F}; // world-space surface normal, facing the capsule
};

// What CharacterBody needs of a collision world. A seam so the movement maths can be tested without Bullet.
class CapsuleSweep {
public:
    CapsuleSweep() = default;
    CapsuleSweep(CapsuleSweep const &) = default;
    auto operator=(CapsuleSweep const &) -> CapsuleSweep & = default;
    CapsuleSweep(CapsuleSweep &&) = default;
    auto operator=(CapsuleSweep &&) -> CapsuleSweep & = default;
    virtual ~CapsuleSweep() = default;

    // Sweeps an upright capsule (total `height` including both caps) whose feet are at `from` to `to`. Returns the
    // first hit, ignoring `ignore`'s body.
    [[nodiscard]] virtual auto sweep_capsule(glm::vec3 const &from, glm::vec3 const &to, float radius, float height,
                                             entt::entity ignore) const -> std::optional<SweepHit> = 0;
};
