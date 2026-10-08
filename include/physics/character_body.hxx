#pragma once

#include "physics/capsule_sweep.hxx"

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

struct MovementParams {
    float walk_speed = 4.0F; // m/s
    float run_speed = 7.0F; // m/s; callers pick which to put in CharacterInput::desired_velocity
    float ground_accel = 50.0F; // m/s^2 towards a non-zero desired velocity
    float ground_decel = 60.0F; // m/s^2 towards zero desired velocity
    float air_accel = 20.0F; // m/s^2 air control, used for both speeding up and slowing down
    float gravity = 25.0F; // m/s^2, positive
    float jump_height = 1.2F; // m; the apex of a full-held jump
    float fall_gravity_multiplier = 1.6F;
    float jump_cut_multiplier = 3.0F; // extra gravity while rising with jump released
    float coyote_time = 0.1F; // s a jump is still allowed after leaving the ground
    float jump_buffer_time = 0.1F; // s a jump press is remembered before landing
    float max_slope_degrees = 50.0F;
    float step_height = 0.3F;
    float max_fall_speed = 40.0F; // m/s

    [[nodiscard]] auto jump_speed() const noexcept -> float;
};

struct CharacterInput {
    glm::vec3 desired_velocity{0.0F}; // world space, y ignored
    bool jump_pressed = false; // true on the frame the button went down
    bool jump_held = false;
};

// Kinematic capsule controller (collide-and-slide). `position` is the feet.
class CharacterBody {
public:
    CharacterBody(glm::vec3 const &position, float radius, float height, MovementParams const &params = {},
                  entt::entity self = entt::null);

    // Advances one fixed step. Call at a constant dt (see FixedStepper).
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
    // Moves by `delta`, sliding along hits. Returns the last hit, if any.
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
    bool jumped_ = false; // since last on the ground; stops coyote time giving a double jump
    float time_since_grounded_ = 1000.0F;
    float jump_buffer_ = 0.0F;
};
