#include "scene/editor_camera.hxx"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace {
    constexpr float pitch_limit_degrees = 89.0F;
    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    // Core coordinate transformation: standard LH Spherical -> Cartesian Basis
    struct CameraBasis {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 up;
    };

    [[nodiscard]] auto compute_basis(float yaw_degrees, float pitch_degrees) noexcept -> CameraBasis {
        auto const yaw = glm::radians(yaw_degrees);
        auto const pitch = glm::radians(pitch_degrees);

        glm::vec3 const forward = glm::normalize(
                glm::vec3{std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)});

        // Left-Handed convention matching glm::lookAtLH
        glm::vec3 const right = glm::normalize(glm::cross(world_up, forward));
        glm::vec3 const up = glm::normalize(glm::cross(forward, right));

        return {forward, right, up};
    }

    auto is_forward_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_W; }
    auto is_backward_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_S; }
    auto is_left_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_A; }
    auto is_right_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_D; }
    auto is_up_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_E; }
    auto is_down_key(std::int32_t key) noexcept -> bool { return key == GLFW_KEY_Q; }
} // namespace

auto EditorCamera::on_key_pressed(std::int32_t key) noexcept -> void {
    if (is_forward_key(key)) {
        moving_forward_ = true;
    } else if (is_backward_key(key)) {
        moving_backward_ = true;
    } else if (is_left_key(key)) {
        moving_left_ = true;
    } else if (is_right_key(key)) {
        moving_right_ = true;
    } else if (is_up_key(key)) {
        moving_up_ = true;
    } else if (is_down_key(key)) {
        moving_down_ = true;
    }
}

auto EditorCamera::on_key_released(std::int32_t key) noexcept -> void {
    if (is_forward_key(key)) {
        moving_forward_ = false;
    } else if (is_backward_key(key)) {
        moving_backward_ = false;
    } else if (is_left_key(key)) {
        moving_left_ = false;
    } else if (is_right_key(key)) {
        moving_right_ = false;
    } else if (is_up_key(key)) {
        moving_up_ = false;
    } else if (is_down_key(key)) {
        moving_down_ = false;
    }
}

auto EditorCamera::set_sprinting(bool sprinting) noexcept -> void { sprinting_ = sprinting; }

EditorCamera::EditorCamera(EditorCameraCreateInfo const &create_info) noexcept :
    position_(create_info.position), yaw_degrees_(create_info.yaw_degrees), pitch_degrees_(create_info.pitch_degrees),
    field_of_view_degrees_(create_info.field_of_view_degrees), near_clip_(create_info.near_clip),
    far_clip_(create_info.far_clip), move_speed_(create_info.move_speed),
    sprint_multiplier_(create_info.sprint_multiplier), look_sensitivity_(create_info.look_sensitivity),
    min_move_speed_(create_info.min_move_speed), max_move_speed_(create_info.max_move_speed) {
    rebuild_basis();
}

auto EditorCamera::on_mouse_moved(float delta_x, float delta_y, bool dragging) noexcept -> void {
    if (!dragging)
        return;

    // Subtracting delta_x ensures turning left rotates camera left
    yaw_degrees_ -= delta_x * look_sensitivity_;
    pitch_degrees_ =
            std::clamp(pitch_degrees_ - (delta_y * look_sensitivity_), -pitch_limit_degrees, pitch_limit_degrees);

    rebuild_basis();
}

auto EditorCamera::on_mouse_scrolled(float delta_y) noexcept -> void {
    move_speed_ = std::clamp(move_speed_ * (1.0F + delta_y * 0.1F), min_move_speed_, max_move_speed_);
}

auto EditorCamera::rebuild_basis() noexcept -> void {
    auto const basis = compute_basis(yaw_degrees_, pitch_degrees_);
    forward_ = basis.forward;
    right_ = basis.right;
    up_ = basis.up;
}

auto EditorCamera::update(float delta_time_seconds) noexcept -> void {
    auto const speed = move_speed_ * (sprinting_ ? sprint_multiplier_ : 1.0F) * delta_time_seconds;

    if (moving_forward_)
        position_ += forward_ * speed;
    if (moving_backward_)
        position_ -= forward_ * speed;
    if (moving_right_)
        position_ += right_ * speed; // D key -> Positive right_
    if (moving_left_)
        position_ -= right_ * speed; // A key -> Negative right_
    if (moving_up_)
        position_ += up_ * speed;
    if (moving_down_)
        position_ -= up_ * speed;
}

auto EditorCamera::view() const noexcept -> glm::mat4 { return glm::lookAtLH(position_, position_ + forward_, up_); }

auto EditorCamera::projection(float aspect_ratio) const noexcept -> glm::mat4 {
    return glm::perspectiveLH_ZO(glm::radians(field_of_view_degrees_), aspect_ratio, near_clip_, far_clip_);
}

auto EditorCamera::view_projection(float aspect_ratio) const noexcept -> glm::mat4 {
    return projection(aspect_ratio) * view();
}
