#pragma once

#include <functional>
#include <optional>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

using CameraOcclusionQuery =
        std::function<std::optional<float>(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance)>;

struct PlayerCameraCreateInfo {
    float follow_distance = 4.0F;
    float follow_height = 1.6F;
    float look_ahead_height = 0.8F;

    float field_of_view_degrees = 65.0F;
    float near_clip = 0.1F;
    float far_clip = 10000.0F;

    float follow_stiffness = 12.0F;

    float bob_amplitude = 0.06F;
    float bob_frequency = 9.0F;

    float wall_margin = 0.3F;
};

class PlayerCamera {
public:
    PlayerCamera() noexcept = default;
    explicit PlayerCamera(PlayerCameraCreateInfo const &create_info) noexcept;

    auto update(glm::vec3 const &capsule_position, float yaw_degrees, float pitch_degrees, float speed_factor,
                float delta_time_seconds, CameraOcclusionQuery const &occlusion_query = {}) -> void;

    [[nodiscard]] auto view() const noexcept -> glm::mat4;
    [[nodiscard]] auto projection(float aspect_ratio) const noexcept -> glm::mat4;
    [[nodiscard]] auto view_projection(float aspect_ratio) const noexcept -> glm::mat4;

    [[nodiscard]] auto position() const noexcept -> glm::vec3 { return position_; }
    [[nodiscard]] auto forward() const noexcept -> glm::vec3 { return forward_; }

    [[nodiscard]] auto near_clip() const noexcept -> float { return near_clip_; }
    [[nodiscard]] auto far_clip() const noexcept -> float { return far_clip_; }
    [[nodiscard]] auto field_of_view_degrees() const noexcept -> float { return field_of_view_degrees_; }

private:
    glm::vec3 position_{0.0F, 2.0F, 6.0F};
    glm::vec3 forward_{0.0F, 0.0F, -1.0F};

    float follow_distance_ = 4.0F;
    float follow_height_ = 1.6F;
    float look_ahead_height_ = 0.8F;
    float follow_stiffness_ = 12.0F;
    float bob_amplitude_ = 0.06F;
    float bob_frequency_ = 9.0F;
    float wall_margin_ = 0.3F;

    float field_of_view_degrees_ = 65.0F;
    float near_clip_ = 0.1F;
    float far_clip_ = 10000.0F;

    float bob_phase_ = 0.0F;
    bool initialized_ = false;
};
