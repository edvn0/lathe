#include "animation/state_machine.hxx"

#include <algorithm>
#include <cmath>

namespace Animation {
    namespace {
        constexpr float idle_enter_walk = 0.25F;
        constexpr float walk_exit_to_idle = 0.1F;
        constexpr float walk_to_run = 3.5F;
        constexpr float run_to_walk = 3.0F; // lower than walk_to_run: hysteresis against flicker
        constexpr float apex_band = 1.0F;   // |vy| below this counts as the top of the jump

        auto ground_state_for(State const current, float const speed) -> State {
            switch (current) {
            case State::Run:
                return speed < run_to_walk ? (speed < walk_exit_to_idle ? State::Idle : State::Walk) : State::Run;
            case State::Walk:
                if (speed >= walk_to_run) {
                    return State::Run;
                }
                return speed < walk_exit_to_idle ? State::Idle : State::Walk;
            default:
                if (speed >= walk_to_run) {
                    return State::Run;
                }
                return speed >= idle_enter_walk ? State::Walk : State::Idle;
            }
        }

        auto air_state_for(float const vertical_velocity) -> State {
            if (vertical_velocity > apex_band) {
                return State::JumpRise;
            }
            return vertical_velocity < -apex_band ? State::JumpFall : State::JumpApex;
        }
    } // namespace

    namespace Transitions {
        auto from_ground(TransitionContext const &context) -> State {
            auto const &in = context.inputs;
            if (in.prone_requested && in.grounded) {
                return State::LyingDown;
            }
            if (!in.grounded) {
                return air_state_for(in.vertical_velocity);
            }
            return ground_state_for(context.current, in.horizontal_speed);
        }

        auto from_air(TransitionContext const &context) -> State {
            auto const &in = context.inputs;
            if (in.grounded) {
                return ground_state_for(State::Idle, in.horizontal_speed);
            }
            return air_state_for(in.vertical_velocity);
        }

        auto from_lying_down(TransitionContext const &context) -> State {
            if (!context.finished) {
                return State::LyingDown;
            }
            return context.inputs.prone_requested ? State::Prone : State::GettingUp;
        }

        auto from_prone(TransitionContext const &context) -> State {
            return context.inputs.prone_requested ? State::Prone : State::GettingUp;
        }

        auto from_getting_up(TransitionContext const &context) -> State {
            if (context.inputs.prone_requested) {
                return State::LyingDown;
            }
            if (!context.finished) {
                return State::GettingUp;
            }
            return from_ground({context.inputs, State::Idle, context.state_time, true});
        }
    } // namespace Transitions

    void AnimStateMachine::advance_phase(State const which, float &phase, float const speed,
                                         float const dt) const {
        auto const &def = definition(which);
        auto const delta = def.stride > 0.0F ? speed * dt / def.stride : dt / std::max(def.clip->duration(), 1e-4F);
        phase = def.loop ? wrap_phase(phase + delta) : std::min(phase + delta, 1.0F);
    }

    void AnimStateMachine::update(StateMachineState &state, AnimInputs const &inputs, float const dt, PoseView const out,
                                  PoseView const scratch) const {
        state.state_time += dt;
        advance_phase(state.current, state.phase, inputs.horizontal_speed, dt);

        auto const &current_def = definition(state.current);
        auto const finished = !current_def.loop && state.phase >= 1.0F;
        auto const next = current_def.next({inputs, state.current, state.state_time, finished});

        if (next != state.current) {
            auto const &next_def = definition(next);
            state.previous = state.current;
            state.previous_phase = state.phase;
            state.current = next;
            state.phase = (current_def.locomotion && next_def.locomotion) ? state.phase : 0.0F;
            state.state_time = 0.0F;
            state.fade = 0.0F;
            state.fade_duration = std::max(next_def.fade_in, 1e-4F);
        }

        if (state.fade < 1.0F) {
            state.fade = std::min(state.fade + dt / state.fade_duration, 1.0F);
            advance_phase(state.previous, state.previous_phase, inputs.horizontal_speed, dt);
        }

        definition(state.current).clip->sample(state.phase, out);
        if (state.fade < 1.0F) {
            definition(state.previous).clip->sample(state.previous_phase, scratch);
            // Smoothstep so the fade has no velocity pop at either end.
            auto const t = state.fade * state.fade * (3.0F - 2.0F * state.fade);
            blend_poses(scratch, out, t, out);
        }
    }
} // namespace Animation
