#pragma once

#include "animation/clip.hxx"
#include "animation/state_machine.hxx"

#include <memory>
#include <vector>

namespace Animation {
    class FrozenClip final : public Clip {
    public:
        FrozenClip(Clip const &source, float phase) : source_{&source}, phase_{phase} {}

        [[nodiscard]] auto duration() const -> float override { return source_->duration(); }
        void sample(float, PoseView const out) const override { source_->sample(phase_, out); }

    private:
        Clip const *source_;
        float phase_;
    };

    class ReversedClip final : public Clip {
    public:
        explicit ReversedClip(Clip const &source) : source_{&source} {}

        [[nodiscard]] auto duration() const -> float override { return source_->duration(); }
        void sample(float const phase, PoseView const out) const override { source_->sample(1.0F - phase, out); }

    private:
        Clip const *source_;
    };

    struct LocomotionClipSet {
        Clip const *idle{nullptr};
        Clip const *walk{nullptr};
        Clip const *run{nullptr};
        Clip const *jump{nullptr};
        Clip const *lie_down{nullptr};
        float walk_stride{1.6F};
        float run_stride{3.6F};
        float jump_rise_phase{0.3F};
        float jump_apex_phase{0.5F};
        float jump_fall_phase{0.7F};
    };

    class LocomotionStateTable {
    public:
        explicit LocomotionStateTable(LocomotionClipSet const &clips);
        LocomotionStateTable(LocomotionStateTable const &) = delete;
        auto operator=(LocomotionStateTable const &) -> LocomotionStateTable & = delete;
        LocomotionStateTable(LocomotionStateTable &&) = delete;
        auto operator=(LocomotionStateTable &&) -> LocomotionStateTable & = delete;
        ~LocomotionStateTable() = default;

        [[nodiscard]] auto table() const -> StateTable const & { return table_; }

    private:
        std::vector<std::unique_ptr<Clip>> owned_;
        StateTable table_{};
    };
}
