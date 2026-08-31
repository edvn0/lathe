#pragma once

#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>


struct PlayerControllerCreateInfo {
    float yaw_degrees = -90.0F;
    float pitch_degrees = 0.0F;

    float move_speed = 5.0F;
    float sprint_multiplier = 1.8F;
    float look_sensitivity = 0.12F;
    float min_pitch_degrees = -80.0F;
    float max_pitch_degrees = 80.0F;
};

class PlayerController {
public:
    PlayerController() noexcept = default;
    explicit PlayerController(PlayerControllerCreateInfo const &create_info) noexcept;

    auto on_key_pressed(std::int32_t key) noexcept -> void;
    auto on_key_released(std::int32_t key) noexcept -> void;
    auto on_mouse_moved(float delta_x, float delta_y, bool look_enabled) noexcept -> void;
    auto set_sprinting(bool sprinting) noexcept -> void;

    // Horizontal velocity (y = 0) in world space, derived from current
    // move-intent flags and yaw -- feed straight into PhysicsWorld::set_velocity.
    [[nodiscard]] auto desired_horizontal_velocity() const noexcept -> glm::vec3;

    [[nodiscard]] auto yaw_degrees() const noexcept -> float { return yaw_degrees_; }
    [[nodiscard]] auto pitch_degrees() const noexcept -> float { return pitch_degrees_; }
    [[nodiscard]] auto forward() const noexcept -> glm::vec3;
    [[nodiscard]] auto is_moving() const noexcept -> bool;

    [[nodiscard]] auto move_speed() const noexcept -> float { return move_speed_; }
    [[nodiscard]] auto consumes_jump() noexcept -> bool;

private:
    float yaw_degrees_ = -90.0F;
    float pitch_degrees_ = 0.0F;

    float move_speed_ = 5.0F;
    float sprint_multiplier_ = 1.8F;
    float look_sensitivity_ = 0.12F;
    float min_pitch_degrees_ = -80.0F;
    float max_pitch_degrees_ = 80.0F;

    bool moving_forward_ = false;
    bool moving_backward_ = false;
    bool moving_left_ = false;
    bool moving_right_ = false;
    bool sprinting_ = false;

    bool jump_requested_{false};
};
