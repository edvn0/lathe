#include "physics/character_body.hxx"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <algorithm>
#include <cmath>

namespace {
    constexpr float skin = 0.01F;
    constexpr int max_slide_iterations = 4;
    constexpr float epsilon = 1e-5F;

    auto move_towards(glm::vec3 const &current, glm::vec3 const &target, float max_delta) -> glm::vec3 {
        glm::vec3 const diff = target - current;
        float const length = glm::length(diff);
        if (length <= max_delta || length < epsilon) {
            return target;
        }
        return current + diff * (max_delta / length);
    }
}

auto MovementParams::jump_speed() const noexcept -> float { return std::sqrt(2.0F * gravity * jump_height); }

CharacterBody::CharacterBody(glm::vec3 const &position, float radius, float height, MovementParams const &params_in,
                             entt::entity self) :
    params{params_in}, position_{position}, previous_position_{position}, radius_{radius}, height_{height},
    self_{self} {}

auto CharacterBody::teleport(glm::vec3 const &position) -> void {
    position_ = position;
    previous_position_ = position;
    velocity_ = glm::vec3{0.0F};
    grounded_ = false;
    time_since_grounded_ = 1000.0F;
}

auto CharacterBody::interpolated_position(float alpha) const -> glm::vec3 {
    return glm::mix(previous_position_, position_, glm::clamp(alpha, 0.0F, 1.0F));
}

auto CharacterBody::walkable(glm::vec3 const &normal) const -> bool {
    return normal.y >= std::cos(glm::radians(params.max_slope_degrees)) - epsilon;
}

auto CharacterBody::sweep(CapsuleSweep const &world, glm::vec3 const &from, glm::vec3 const &to) const
        -> std::optional<SweepHit> {
    return world.sweep_capsule(from, to, radius_, height_, self_);
}

auto CharacterBody::move_and_slide(CapsuleSweep const &world, glm::vec3 delta, bool keep_velocity_on_walkable)
        -> std::optional<SweepHit> {
    std::optional<SweepHit> last_hit;

    for (int iteration = 0; iteration < max_slide_iterations; ++iteration) {
        float const length = glm::length(delta);
        if (length < epsilon) {
            break;
        }

        auto const hit = sweep(world, position_, position_ + delta);
        if (!hit) {
            position_ += delta;
            break;
        }
        last_hit = hit;

        glm::vec3 const direction = delta / length;
        float const travel = std::max(hit->fraction * length - skin, 0.0F);
        position_ += direction * travel;

        glm::vec3 const remaining = delta * (1.0F - hit->fraction);
        delta = remaining - hit->normal * glm::dot(remaining, hit->normal);

        if (!(keep_velocity_on_walkable && walkable(hit->normal))) {
            float const into = glm::dot(velocity_, hit->normal);
            if (into < 0.0F) {
                velocity_ -= hit->normal * into;
            }
        }
    }
    return last_hit;
}

auto CharacterBody::try_step_up(CapsuleSweep const &world, glm::vec3 const &horizontal) -> bool {
    glm::vec3 const up{0.0F, params.step_height, 0.0F};

    glm::vec3 raised = position_;
    if (auto const hit = sweep(world, raised, raised + up)) {
        raised.y += std::max(hit->fraction * params.step_height - skin, 0.0F);
    } else {
        raised += up;
    }

    glm::vec3 reach = horizontal;
    float const reach_length = glm::length(horizontal);
    float const min_reach = radius_ * 0.5F;
    if (reach_length > epsilon && reach_length < min_reach) {
        reach *= min_reach / reach_length;
    }

    glm::vec3 forward = raised + reach;
    if (auto const hit = sweep(world, raised, forward)) {
        forward = raised + reach * hit->fraction;
    }
    if (glm::length(glm::vec2{forward.x - position_.x, forward.z - position_.z}) < skin * 2.0F) {
        return false;
    }

    auto const down = sweep(world, forward, forward - glm::vec3{0.0F, raised.y - position_.y + skin, 0.0F});
    if (!down || !walkable(down->normal)) {
        return false;
    }
    forward.y -= std::max(down->fraction * (raised.y - position_.y + skin) - skin, 0.0F);
    position_ = forward;
    return true;
}

auto CharacterBody::step(CapsuleSweep const &world, CharacterInput const &input, float dt) -> void {
    previous_position_ = position_;

    jump_buffer_ = input.jump_pressed ? params.jump_buffer_time : std::max(jump_buffer_ - dt, 0.0F);

    glm::vec3 const wish{input.desired_velocity.x, 0.0F, input.desired_velocity.z};
    bool const has_input = glm::length(wish) > epsilon;
    float const accel = grounded_ ? (has_input ? params.ground_accel : params.ground_decel) : params.air_accel;
    glm::vec3 const horizontal_velocity = move_towards({velocity_.x, 0.0F, velocity_.z}, wish, accel * dt);
    velocity_.x = horizontal_velocity.x;
    velocity_.z = horizontal_velocity.z;

    bool const can_jump = !jumped_ && (grounded_ || time_since_grounded_ <= params.coyote_time);
    if (jump_buffer_ > 0.0F && can_jump) {
        velocity_.y = params.jump_speed();
        jump_buffer_ = 0.0F;
        jumped_ = true;
        grounded_ = false;
        time_since_grounded_ = 1000.0F;
    }

    float const vy_before = velocity_.y;
    if (grounded_ && velocity_.y <= 0.0F) {
        velocity_.y = 0.0F;
    } else {
        float scale = 1.0F;
        if (velocity_.y < 0.0F) {
            scale = params.fall_gravity_multiplier;
        } else if (!input.jump_held) {
            scale = params.jump_cut_multiplier;
        }
        velocity_.y = std::max(velocity_.y - params.gravity * scale * dt, -params.max_fall_speed);
    }

    bool const was_grounded = grounded_;
    float const vertical_move = 0.5F * (vy_before + velocity_.y) * dt;

    glm::vec3 const horizontal_delta{velocity_.x * dt, 0.0F, velocity_.z * dt};
    glm::vec3 const before_horizontal = position_;
    auto const wall = move_and_slide(world, horizontal_delta, true);
    if (wall && !walkable(wall->normal) && wall->normal.y > -0.5F && was_grounded) {
        glm::vec3 const slid = position_;
        position_ = before_horizontal;
        if (!try_step_up(world, horizontal_delta)) {
            position_ = slid;
        }
    }

    auto const vertical_hit = move_and_slide(world, {0.0F, vertical_move, 0.0F}, false);
    if (vertical_hit && vertical_hit->normal.y < -0.5F && velocity_.y > 0.0F) {
        velocity_.y = 0.0F;
    }

    bool const rising = velocity_.y > 0.0F;
    float const probe = (was_grounded && !rising) ? params.step_height + skin : skin * 2.0F;
    auto const ground = rising ? std::nullopt : sweep(world, position_, position_ - glm::vec3{0.0F, probe, 0.0F});

    if (ground && walkable(ground->normal)) {
        if (was_grounded || velocity_.y <= 0.0F) {
            position_.y -= std::max(ground->fraction * probe - skin, 0.0F);
        }
        grounded_ = true;
        ground_normal_ = ground->normal;
        jumped_ = false;
        velocity_.y = 0.0F;
        time_since_grounded_ = 0.0F;
    } else {
        grounded_ = false;
        ground_normal_ = glm::vec3{0.0F, 1.0F, 0.0F};
        time_since_grounded_ += dt;
    }
}
