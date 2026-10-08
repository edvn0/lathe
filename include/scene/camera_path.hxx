#pragma once

#include <glm/vec3.hpp>

#include <span>

struct CameraKeyframe {
    glm::vec3 position{0.0F};
    glm::vec3 target{0.0F, 0.0F, -1.0F};
};

[[nodiscard]]
auto sample_camera_path(std::span<CameraKeyframe const> keyframes, float t) noexcept -> CameraKeyframe;
