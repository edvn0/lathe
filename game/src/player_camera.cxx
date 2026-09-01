#include "player_camera.hxx"

#include <algorithm>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

namespace {
    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    [[nodiscard]] auto compute_forward(float yaw_degrees, float pitch_degrees) noexcept -> glm::vec3 {
        auto const yaw = glm::radians(yaw_degrees);
        auto const pitch = glm::radians(pitch_degrees);

        return glm::normalize(
                glm::vec3{std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)});
    }
} // namespace

PlayerCamera::PlayerCamera(PlayerCameraCreateInfo const &create_info) noexcept :
    follow_distance_(create_info.follow_distance), follow_height_(create_info.follow_height),
    look_ahead_height_(create_info.look_ahead_height), follow_stiffness_(create_info.follow_stiffness),
    bob_amplitude_(create_info.bob_amplitude), bob_frequency_(create_info.bob_frequency),
    wall_margin_(create_info.wall_margin), field_of_view_degrees_(create_info.field_of_view_degrees),
    near_clip_(create_info.near_clip), far_clip_(create_info.far_clip) {}

auto PlayerCamera::update(glm::vec3 const &capsule_position, float yaw_degrees, float pitch_degrees, float speed_factor,
                          float delta_time_seconds, CameraOcclusionQuery const &occlusion_query) -> void {
    // Shared forward calculation to maintain LH view-space alignment with EditorCamera
    forward_ = compute_forward(yaw_degrees, pitch_degrees);

    // Horizontal flat forward vector for character relative calculations
    auto const yaw = glm::radians(yaw_degrees);
    auto const flat_forward = glm::normalize(glm::vec3{std::cos(yaw), 0.0F, std::sin(yaw)});

    // Head bob offset
    bob_phase_ += bob_frequency_ * speed_factor * delta_time_seconds;
    auto const bob_offset = (speed_factor > 0.01F) ? (std::sin(bob_phase_) * bob_amplitude_ * speed_factor) : 0.0F;

    // Pivot the camera orbits around -- directly above the capsule, with no
    // backward pull -- so the occlusion ray tests the same segment the
    // camera actually sits on.
    auto const anchor = capsule_position + glm::vec3{0.0F, follow_height_ + bob_offset, 0.0F};
    auto const target_position = anchor - (flat_forward * follow_distance_);

    auto desired_position = target_position;
    auto obstructed = false;

    if (occlusion_query) {
        auto const to_target = target_position - anchor;
        auto const distance = glm::length(to_target);

        if (distance > 1e-4F) {
            auto const direction = to_target / distance;

            if (auto const hit_distance = occlusion_query(anchor, direction, distance)) {
                auto const clamped_distance = std::max(0.0F, *hit_distance - wall_margin_);
                desired_position = anchor + direction * clamped_distance;
                obstructed = true;
            }
        }
    }

    if (!initialized_) {
        position_ = desired_position;
        initialized_ = true;
    } else if (obstructed) {
        // Snap in immediately rather than springing -- easing here would let
        // the camera clip through the wall for the duration of the ease.
        position_ = desired_position;
    } else {
        auto const t = 1.0F - std::exp(-follow_stiffness_ * delta_time_seconds);
        position_ = glm::mix(position_, desired_position, t);
    }
}

auto PlayerCamera::view() const noexcept -> glm::mat4 {
    return glm::lookAtLH(position_, position_ + forward_, world_up);
}

auto PlayerCamera::projection(float aspect_ratio) const noexcept -> glm::mat4 {
    return glm::perspectiveLH_ZO(glm::radians(field_of_view_degrees_), aspect_ratio, near_clip_, far_clip_);
}

auto PlayerCamera::view_projection(float aspect_ratio) const noexcept -> glm::mat4 {
    return projection(aspect_ratio) * view();
}
