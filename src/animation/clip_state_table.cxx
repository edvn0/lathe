#include "animation/clip_state_table.hxx"

#include <cassert>

namespace Animation {
    LocomotionStateTable::LocomotionStateTable(LocomotionClipSet const &clips) {
        assert(clips.idle != nullptr);

        auto const walk = clips.walk != nullptr ? clips.walk : clips.idle;
        auto const run = clips.run != nullptr ? clips.run : walk;
        auto const walk_stride = clips.walk != nullptr ? clips.walk_stride : 0.0F;
        auto const run_stride = clips.run != nullptr ? clips.run_stride : walk_stride;

        auto const own = [this](std::unique_ptr<Clip> clip) -> Clip const * {
            owned_.push_back(std::move(clip));
            return owned_.back().get();
        };
        auto const frozen = [&](Clip const *source, float phase) -> Clip const * {
            return source != nullptr ? own(std::make_unique<FrozenClip>(*source, phase)) : clips.idle;
        };

        auto const rise = frozen(clips.jump, clips.jump_rise_phase);
        auto const apex = frozen(clips.jump, clips.jump_apex_phase);
        auto const fall = frozen(clips.jump, clips.jump_fall_phase);

        auto const lying = clips.lie_down != nullptr ? clips.lie_down : clips.idle;
        auto const prone = frozen(clips.lie_down, 1.0F);
        auto const getting_up = clips.lie_down != nullptr ? own(std::make_unique<ReversedClip>(*clips.lie_down))
                                                           : clips.idle;

        auto const set = [this](State const state, StateDefinition definition) {
            table_[static_cast<std::size_t>(state)] = definition;
        };
        // name, clip, stride, loop, locomotion, fade_in, transition
        set(State::Idle, {"idle", clips.idle, 0.0F, true, false, 0.25F, Transitions::from_ground});
        set(State::Walk, {"walk", walk, walk_stride, true, walk_stride > 0.0F, 0.2F, Transitions::from_ground});
        set(State::Run, {"run", run, run_stride, true, run_stride > 0.0F, 0.2F, Transitions::from_ground});
        set(State::JumpRise, {"jump_rise", rise, 0.0F, true, false, 0.1F, Transitions::from_air});
        set(State::JumpApex, {"jump_apex", apex, 0.0F, true, false, 0.15F, Transitions::from_air});
        set(State::JumpFall, {"jump_fall", fall, 0.0F, true, false, 0.15F, Transitions::from_air});
        set(State::LyingDown, {"lying_down", lying, 0.0F, false, false, 0.1F, Transitions::from_lying_down});
        set(State::Prone, {"prone", prone, 0.0F, true, false, 0.1F, Transitions::from_prone});
        set(State::GettingUp, {"getting_up", getting_up, 0.0F, false, false, 0.1F, Transitions::from_getting_up});
    }
} // namespace Animation
