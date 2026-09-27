#pragma once

#include <glm/vec3.hpp>

#include <span>

// One stop on a scripted camera flight: position and look-at target.
struct CameraKeyframe {
    glm::vec3 position{0.0F};
    glm::vec3 target{0.0F, 0.0F, -1.0F};
};

// Closed uniform Catmull-Rom loop through the keyframes, with position and target splined separately. `t` in
// [0, 1) covers the loop and passes keyframe i at t = i / size; other values wrap. Empty input returns a
// default keyframe.
[[nodiscard]]
auto sample_camera_path(std::span<CameraKeyframe const> keyframes, float t) noexcept -> CameraKeyframe;
