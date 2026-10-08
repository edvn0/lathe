#pragma once

#include "animation/skeleton.hxx"

#include <cstdint>
#include <functional>
#include <vector>

namespace Animation {
    // A clip turns a normalized phase into a full pose. Phase is in [0, 1]; looping clips treat 1 == 0.
    // Implementations must write every joint of `out` (start from the bind pose for untouched joints).
    class Clip {
    public:
        Clip() = default;
        Clip(Clip const &) = delete;
        auto operator=(Clip const &) -> Clip & = delete;
        Clip(Clip &&) = delete;
        auto operator=(Clip &&) -> Clip & = delete;
        virtual ~Clip() = default;

        // Seconds for one full phase cycle at playback speed 1.
        [[nodiscard]] virtual auto duration() const -> float = 0;
        virtual void sample(float phase, PoseView out) const = 0;
    };

    struct Vec3Track {
        std::uint32_t joint{0};
        std::vector<float> times; // seconds, strictly increasing
        std::vector<glm::vec3> values;
    };

    struct QuatTrack {
        std::uint32_t joint{0};
        std::vector<float> times;
        std::vector<glm::quat> values;
    };

    // Authored/imported clip: per-joint T/R/S keyframes, lerp/slerp between them. Joints without a track keep `base`.
    class KeyframeClip final : public Clip {
    public:
        KeyframeClip(Pose base, float duration) : base_{std::move(base)}, duration_{duration} {}

        void add_translation_track(Vec3Track track) { translations_.push_back(std::move(track)); }
        void add_rotation_track(QuatTrack track) { rotations_.push_back(std::move(track)); }
        void add_scale_track(Vec3Track track) { scales_.push_back(std::move(track)); }

        [[nodiscard]] auto duration() const -> float override { return duration_; }
        void sample(float phase, PoseView out) const override;

    private:
        Pose base_;
        float duration_;
        std::vector<Vec3Track> translations_;
        std::vector<QuatTrack> rotations_;
        std::vector<Vec3Track> scales_;
    };

    // Code-driven clip with the same interface as keyframed ones.
    class ProceduralClip final : public Clip {
    public:
        using Function = std::function<void(float phase, PoseView out)>;

        ProceduralClip(float duration, Function function) : duration_{duration}, function_{std::move(function)} {}

        [[nodiscard]] auto duration() const -> float override { return duration_; }
        void sample(float const phase, PoseView const out) const override { function_(phase, out); }

    private:
        float duration_;
        Function function_;
    };

    // Speed-driven blend of two gaits. Both clips are sampled at the same phase, so their cycles must be
    // foot-aligned (phase 0 = same foot contact), which is what makes the blend look coherent.
    struct LocomotionBlend {
        Clip const *slow{nullptr};
        Clip const *fast{nullptr};
        float slow_speed{0.0F};   // m/s at which weight == 0
        float fast_speed{1.0F};   // m/s at which weight == 1
        float slow_stride{1.0F};  // metres travelled per full cycle of `slow`
        float fast_stride{2.0F};

        [[nodiscard]] auto weight(float speed) const -> float;
        [[nodiscard]] auto stride(float speed) const -> float;
        // Advance by distance, not time, so feet don't skate: phase += distance / stride.
        [[nodiscard]] auto advance_phase(float phase, float speed, float dt) const -> float;
        // `scratch` must have the same joint count as `out`.
        void sample(float phase, float speed, PoseView out, PoseView scratch) const;
    };

    [[nodiscard]] auto wrap_phase(float phase) -> float;
} // namespace Animation
