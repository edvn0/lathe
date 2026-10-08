#include "animation/clip.hxx"

#include <algorithm>
#include <cmath>

namespace Animation {
    namespace {
        struct Segment {
            std::size_t first{0};
            std::size_t second{0};
            float t{0.0F};
        };

        auto find_segment(std::vector<float> const &times, float const time) -> Segment {
            auto const upper = std::ranges::upper_bound(times, time);
            if (upper == times.begin()) {
                return {};
            }
            auto const second = static_cast<std::size_t>(upper - times.begin());
            if (upper == times.end()) {
                return {second - 1, second - 1, 0.0F};
            }
            auto const first = second - 1;
            return {first, second, (time - times[first]) / (times[second] - times[first])};
        }
    }

    void KeyframeClip::sample(float const phase, PoseView const out) const {
        copy_pose(base_.view(), out);
        auto const time = std::clamp(phase, 0.0F, 1.0F) * duration_;

        for (auto const &track : translations_) {
            if (track.times.empty()) {
                continue;
            }
            auto const seg = find_segment(track.times, time);
            out.translation[track.joint] = glm::mix(track.values[seg.first], track.values[seg.second], seg.t);
        }
        for (auto const &track : rotations_) {
            if (track.times.empty()) {
                continue;
            }
            auto const seg = find_segment(track.times, time);
            out.rotation[track.joint] =
                    glm::normalize(glm::slerp(track.values[seg.first], track.values[seg.second], seg.t));
        }
        for (auto const &track : scales_) {
            if (track.times.empty()) {
                continue;
            }
            auto const seg = find_segment(track.times, time);
            out.scale[track.joint] = glm::mix(track.values[seg.first], track.values[seg.second], seg.t);
        }
    }

    auto LocomotionBlend::weight(float const speed) const -> float {
        auto const range = fast_speed - slow_speed;
        return range > 0.0F ? std::clamp((speed - slow_speed) / range, 0.0F, 1.0F) : 0.0F;
    }

    auto LocomotionBlend::stride(float const speed) const -> float {
        return glm::mix(slow_stride, fast_stride, weight(speed));
    }

    auto LocomotionBlend::advance_phase(float const phase, float const speed, float const dt) const -> float {
        return wrap_phase(phase + speed * dt / stride(speed));
    }

    void LocomotionBlend::sample(float const phase, float const speed, PoseView const out, PoseView const scratch) const {
        auto const w = weight(speed);
        slow->sample(phase, out);
        if (w <= 0.0F) {
            return;
        }
        fast->sample(phase, scratch);
        blend_poses(out, scratch, w, out);
    }

    auto wrap_phase(float const phase) -> float {
        auto const wrapped = phase - std::floor(phase);
        return wrapped;
    }
}
