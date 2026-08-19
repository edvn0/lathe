#include "player_controller.hxx"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <utility>

PlayerController::PlayerController(PlayerControllerCreateInfo const &create_info) noexcept :
    yaw_degrees_(create_info.yaw_degrees), pitch_degrees_(create_info.pitch_degrees),
    move_speed_(create_info.move_speed), sprint_multiplier_(create_info.sprint_multiplier),
    look_sensitivity_(create_info.look_sensitivity), min_pitch_degrees_(create_info.min_pitch_degrees),
    max_pitch_degrees_(create_info.max_pitch_degrees) {}

auto PlayerController::on_key_pressed(std::int32_t key) noexcept -> void {
    switch (key) {
        case GLFW_KEY_W:
            moving_forward_ = true;
            break;
        case GLFW_KEY_S:
            moving_backward_ = true;
            break;
        case GLFW_KEY_A:
            moving_left_ = true;
            break;
        case GLFW_KEY_D:
            moving_right_ = true;
            break;
        case GLFW_KEY_SPACE:
            jump_requested_ = true;
            break;
        default:
            break;
    }
}

auto PlayerController::consumes_jump() noexcept -> bool { return std::exchange(jump_requested_, false); }

auto PlayerController::on_key_released(std::int32_t key) noexcept -> void {
    switch (key) {
        case GLFW_KEY_W:
            moving_forward_ = false;
            break;
        case GLFW_KEY_S:
            moving_backward_ = false;
            break;
        case GLFW_KEY_A:
            moving_left_ = false;
            break;
        case GLFW_KEY_D:
            moving_right_ = false;
            break;
        default:
            break;
    }
}

auto PlayerController::on_mouse_moved(float delta_x, float delta_y, bool look_enabled) noexcept -> void {
    if (!look_enabled) {
        return;
    }

    // FIX 1: Subtract delta_x to match EditorCamera look behavior
    yaw_degrees_ -= delta_x * look_sensitivity_;
    pitch_degrees_ = std::clamp(pitch_degrees_ - delta_y * look_sensitivity_, min_pitch_degrees_, max_pitch_degrees_);
}

auto PlayerController::set_sprinting(bool sprinting) noexcept -> void { sprinting_ = sprinting; }

auto PlayerController::forward() const noexcept -> glm::vec3 {
    // Horizontal-only forward -- pitch affects the camera, not movement.
    auto const yaw = glm::radians(yaw_degrees_);
    return glm::normalize(glm::vec3{std::cos(yaw), 0.0F, std::sin(yaw)});
}

auto PlayerController::is_moving() const noexcept -> bool {
    return moving_forward_ || moving_backward_ || moving_left_ || moving_right_;
}

auto PlayerController::desired_horizontal_velocity() const noexcept -> glm::vec3 {
    if (!is_moving()) {
        return glm::vec3{0.0F};
    }

    auto const fwd = forward();

    // FIX 2: Compute right vector as cross(world_up, fwd) for Left-Handed alignment
    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};
    auto const right = glm::normalize(glm::cross(world_up, fwd));

    glm::vec3 direction{0.0F};
    if (moving_forward_)
        direction += fwd;
    if (moving_backward_)
        direction -= fwd;
    if (moving_right_)
        direction += right;
    if (moving_left_)
        direction -= right;

    if (glm::dot(direction, direction) < 1e-6F) {
        return glm::vec3{0.0F};
    }

    auto const speed = move_speed_ * (sprinting_ ? sprint_multiplier_ : 1.0F);
    return glm::normalize(direction) * speed;
}
