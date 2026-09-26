#include "scene/camera_path.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

    [[nodiscard]] auto catmull_rom(glm::vec3 const &p0, glm::vec3 const &p1, glm::vec3 const &p2, glm::vec3 const &p3,
                                   float s) noexcept -> glm::vec3 {
        auto const s2 = s * s;
        auto const s3 = s2 * s;

        return 0.5F * ((2.0F * p1) + (-p0 + p2) * s + (2.0F * p0 - 5.0F * p1 + 4.0F * p2 - p3) * s2 +
                       (-p0 + 3.0F * p1 - 3.0F * p2 + p3) * s3);
    }

} // namespace

auto sample_camera_path(std::span<CameraKeyframe const> keyframes, float t) noexcept -> CameraKeyframe {
    if (keyframes.empty()) {
        return {};
    }

    auto const count = keyframes.size();
    auto const wrapped = t - std::floor(t);
    auto const scaled = wrapped * static_cast<float>(count);

    // min() guards the float edge where `wrapped` rounds up to exactly 1.
    auto const segment = std::min(static_cast<std::size_t>(scaled), count - 1);
    auto const s = scaled - static_cast<float>(segment);

    auto const &k0 = keyframes[(segment + count - 1) % count];
    auto const &k1 = keyframes[segment];
    auto const &k2 = keyframes[(segment + 1) % count];
    auto const &k3 = keyframes[(segment + 2) % count];

    return CameraKeyframe{
            .position = catmull_rom(k0.position, k1.position, k2.position, k3.position, s),
            .target = catmull_rom(k0.target, k1.target, k2.target, k3.target, s),
    };
}
