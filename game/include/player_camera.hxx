#pragma once

#include <functional>
#include <optional>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// Queries whether anything blocks the straight line from `origin` toward
// `direction` within `max_distance`, returning the distance to the nearest
// obstruction if so. Lets PlayerCamera pull itself in front of walls without
// depending on the physics engine directly -- the caller (which already has
// a PhysicsWorld) supplies this via a raycast.
using CameraOcclusionQuery =
        std::function<std::optional<float>(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance)>;

// Over-the-shoulder follow camera for a capsule player controller.
// Deliberately NOT a first-person eye camera -- it trails behind and
// above the capsule, matching the interface shape of EditorCamera
// (view/projection/near_clip/etc.) so BasicGame::camera() can build a
// CameraParams from it with a single branch.
struct PlayerCameraCreateInfo {
    float follow_distance = 4.0F; // behind the capsule, along -forward
    float follow_height = 1.6F; // above the capsule origin
    float look_ahead_height = 0.8F; // camera looks toward a point this far above capsule origin

    float field_of_view_degrees = 65.0F;
    float near_clip = 0.1F;
    float far_clip = 10000.0F;

    // Position spring -- how quickly the camera catches up to its target
    // each second (higher = snappier, lower = laggier/floatier).
    float follow_stiffness = 12.0F;

    float bob_amplitude = 0.06F;
    float bob_frequency = 9.0F; // radians/sec at full move speed

    // Minimum clearance kept between the camera and whatever obstruction it
    // was pulled in front of, so the near plane doesn't poke through a wall.
    float wall_margin = 0.3F;
};

class PlayerCamera {
public:
    PlayerCamera() noexcept = default;
    explicit PlayerCamera(PlayerCameraCreateInfo const &create_info) noexcept;

    // capsule_position/yaw/pitch come from the entity's Transform and the
    // PlayerController; speed_factor is desired_horizontal_velocity's
    // length / move_speed (0 = idle, 1 = walking, > 1 = sprinting) and
    // drives the bob -- pass 0 to disable bob entirely (e.g. airborne).
    //
    // occlusion_query, if set, is used to pull the camera in between the
    // player and the desired follow position whenever a wall or other
    // obstruction sits between them -- see CameraOcclusionQuery. Leave it
    // empty to fall back to the un-clamped spring behaviour.
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
    bool initialized_ = false; // first update() snaps instead of springing, avoids a slide-in from origin
};
