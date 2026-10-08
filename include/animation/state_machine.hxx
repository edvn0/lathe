#pragma once

#include "animation/clip.hxx"

#include <array>
#include <cstdint>
#include <string_view>

namespace Animation {
    enum class Lod : std::uint8_t {
        Full,
        Reduced,
        Hold,
    };

    struct AnimInputs {
        float horizontal_speed{0.0F};
        bool grounded{true};
        float vertical_velocity{0.0F};
        bool prone_requested{false};
        Lod lod{Lod::Full};
    };

    enum class State : std::uint8_t {
        Idle,
        Walk,
        Run,
        JumpRise,
        JumpApex,
        JumpFall,
        LyingDown,
        Prone,
        GettingUp,
        Count,
    };
    inline constexpr std::size_t state_count = static_cast<std::size_t>(State::Count);

    struct StateMachineState {
        State current{State::Idle};
        State previous{State::Idle};
        float phase{0.0F};
        float previous_phase{0.0F};
        float state_time{0.0F};
        float fade{1.0F};
        float fade_duration{0.2F};
    };

    struct TransitionContext {
        AnimInputs const &inputs;
        State current;
        float state_time;
        bool finished;
    };

    using TransitionFunction = State (*)(TransitionContext const &);

    struct StateDefinition {
        std::string_view name;
        Clip const *clip{nullptr};
        float stride{0.0F};
        bool loop{true};
        bool locomotion{false};
        float fade_in{0.2F};
        TransitionFunction next{nullptr};
    };

    using StateTable = std::array<StateDefinition, state_count>;

    namespace Transitions {
        [[nodiscard]] auto from_ground(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_air(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_lying_down(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_prone(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_getting_up(TransitionContext const &context) -> State;
    }

    class AnimStateMachine {
    public:
        explicit AnimStateMachine(StateTable table) : table_{table} {}

        [[nodiscard]] auto definition(State state) const -> StateDefinition const & {
            return table_[static_cast<std::size_t>(state)];
        }

        void update(StateMachineState &state, AnimInputs const &inputs, float dt, PoseView out, PoseView scratch) const;

    private:
        void advance_phase(State which, float &phase, float speed, float dt) const;
        StateTable table_;
    };
}
