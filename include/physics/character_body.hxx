#pragma once

#include "physics/capsule_sweep.hxx"

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

struct MovementParams {
    float walk_speed = 4.0F;
    float run_speed = 7.0F;
    float ground_accel = 50.0F;
    float ground_decel = 60.0F;
    float air_accel = 20.0F;
    float gravity = 25.0F;
    float jump_height = 1.2F;
    float fall_gravity_multiplier = 1.6F;
    float jump_cut_multiplier = 3.0F;
    float coyote_time = 0.1F;
    float jump_buffer_time = 0.1F;
    float max_slope_degrees = 50.0F;
    float step_height = 0.3F;
    float max_fall_speed = 40.0F;

    [[nodiscard]] auto jump_speed() const noexcept -> float;
};

struct CharacterInput {
    glm::vec3 desired_velocity{0.0F};
    bool jump_pressed = false;
    bool jump_held = false;
};

class CharacterBody {
public:
    CharacterBody(glm::vec3 const &position, float radius, float height, MovementParams const &params = {},
                  entt::entity self = entt::null);

    auto step(CapsuleSweep const &world, CharacterInput const &input, float dt) -> void;

    auto teleport(glm::vec3 const &position) -> void;

    [[nodiscard]] auto interpolated_position(float alpha) const -> glm::vec3;

    [[nodiscard]] auto position() const noexcept -> glm::vec3 const & { return position_; }
    [[nodiscard]] auto previous_position() const noexcept -> glm::vec3 const & { return previous_position_; }
    [[nodiscard]] auto velocity() const noexcept -> glm::vec3 const & { return velocity_; }
    [[nodiscard]] auto grounded() const noexcept -> bool { return grounded_; }
    [[nodiscard]] auto ground_normal() const noexcept -> glm::vec3 const & { return ground_normal_; }
    [[nodiscard]] auto time_since_grounded() const noexcept -> float { return time_since_grounded_; }
    [[nodiscard]] auto radius() const noexcept -> float { return radius_; }
    [[nodiscard]] auto height() const noexcept -> float { return height_; }

    MovementParams params;

private:
    auto move_and_slide(CapsuleSweep const &world, glm::vec3 delta, bool keep_velocity_on_walkable)
            -> std::optional<SweepHit>;
    auto try_step_up(CapsuleSweep const &world, glm::vec3 const &horizontal) -> bool;
    [[nodiscard]] auto walkable(glm::vec3 const &normal) const -> bool;
    [[nodiscard]] auto sweep(CapsuleSweep const &world, glm::vec3 const &from, glm::vec3 const &to) const
            -> std::optional<SweepHit>;

    glm::vec3 position_;
    glm::vec3 previous_position_;
    glm::vec3 velocity_{0.0F};
    glm::vec3 ground_normal_{0.0F, 1.0F, 0.0F};
    float radius_;
    float height_;
    entt::entity self_;
    bool grounded_ = false;
    bool jumped_ = false;
    float time_since_grounded_ = 1000.0F;
    float jump_buffer_ = 0.0F;
};
