#pragma once

#include "animation/skeleton.hxx"

#include <cstdint>
#include <functional>
#include <vector>

namespace Animation {
    class Clip {
    public:
        Clip() = default;
        Clip(Clip const &) = delete;
        auto operator=(Clip const &) -> Clip & = delete;
        Clip(Clip &&) = delete;
        auto operator=(Clip &&) -> Clip & = delete;
        virtual ~Clip() = default;

        [[nodiscard]] virtual auto duration() const -> float = 0;
        virtual void sample(float phase, PoseView out) const = 0;
    };

    struct Vec3Track {
        std::uint32_t joint{0};
        std::vector<float> times;
        std::vector<glm::vec3> values;
    };

    struct QuatTrack {
        std::uint32_t joint{0};
        std::vector<float> times;
        std::vector<glm::quat> values;
    };

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

    struct LocomotionBlend {
        Clip const *slow{nullptr};
        Clip const *fast{nullptr};
        float slow_speed{0.0F};
        float fast_speed{1.0F};
        float slow_stride{1.0F};
        float fast_stride{2.0F};

        [[nodiscard]] auto weight(float speed) const -> float;
        [[nodiscard]] auto stride(float speed) const -> float;
        [[nodiscard]] auto advance_phase(float phase, float speed, float dt) const -> float;
        void sample(float phase, float speed, PoseView out, PoseView scratch) const;
    };

    [[nodiscard]] auto wrap_phase(float phase) -> float;
}
