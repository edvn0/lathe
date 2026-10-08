#pragma once

#include "animation/clip.hxx"
#include "animation/state_machine.hxx"

#include <memory>
#include <vector>

namespace Animation {
    // Plays `source` at one fixed phase (a held pose). Used for states that have no clip of their own, e.g. the
    // airborne poses taken from a single jump clip.
    class FrozenClip final : public Clip {
    public:
        FrozenClip(Clip const &source, float phase) : source_{&source}, phase_{phase} {}

        [[nodiscard]] auto duration() const -> float override { return source_->duration(); }
        void sample(float, PoseView const out) const override { source_->sample(phase_, out); }

    private:
        Clip const *source_;
        float phase_;
    };

    // Plays `source` backwards.
    class ReversedClip final : public Clip {
    public:
        explicit ReversedClip(Clip const &source) : source_{&source} {}

        [[nodiscard]] auto duration() const -> float override { return source_->duration(); }
        void sample(float const phase, PoseView const out) const override { source_->sample(1.0F - phase, out); }

    private:
        Clip const *source_;
    };

    // Skeleton-agnostic source clips for the locomotion state machine; any may be null (see make below).
    struct LocomotionClipSet {
        Clip const *idle{nullptr};
        Clip const *walk{nullptr};
        Clip const *run{nullptr};
        Clip const *jump{nullptr}; // one clip; rise/apex/fall are held poses taken from it
        Clip const *lie_down{nullptr}; // standing -> lying; prone is its last frame, getting up plays it reversed
        float walk_stride{1.6F};  // metres per cycle
        float run_stride{3.6F};
        float jump_rise_phase{0.3F};
        float jump_apex_phase{0.5F};
        float jump_fall_phase{0.7F};
    };

    // Builds the StateTable for arbitrary clips (imported or procedural). Missing clips fall back: run -> walk ->
    // idle, jump -> idle, lie_down -> idle (so the character just stays standing when prone). `idle` is required.
    // The table points at clips owned by this object and at the set's clips, which must outlive it.
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
} // namespace Animation
