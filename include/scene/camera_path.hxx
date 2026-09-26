#pragma once

#include <glm/vec3.hpp>

#include <span>

// One stop on a scripted camera flight: where the camera is, and the point
// it looks at. The benchmark (app/benchmark.hxx) flies a closed loop
// through a game-supplied list of these -- see IGame::benchmark_camera_path.
struct CameraKeyframe {
    glm::vec3 position{0.0F};
    glm::vec3 target{0.0F, 0.0F, -1.0F};
};

// Uniform Catmull-Rom spline through every keyframe as a closed loop (the
// last keyframe blends back into the first), with position and target
// splined independently. `t` in [0, 1) covers the whole loop, one
// keyframe-to-keyframe segment per 1/size; values outside wrap. The curve
// passes exactly through each keyframe at t = i / size. An empty span
// returns a default keyframe.
[[nodiscard]]
auto sample_camera_path(std::span<CameraKeyframe const> keyframes, float t) noexcept -> CameraKeyframe;
