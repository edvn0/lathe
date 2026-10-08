#pragma once

#include "animation/clip.hxx"

#include <array>
#include <cstdint>
#include <string_view>

namespace Animation {
    // Animation level of detail, chosen by the caller (typically from camera distance / screen size).
    enum class Lod : std::uint8_t {
        Full,    // evaluated every update
        Reduced, // evaluated at a fixed lower rate, with the accumulated dt
        Hold,    // frozen: keeps last palette, costs nothing
    };

    // Everything the gameplay side tells the animation system each update.
    struct AnimInputs {
        float horizontal_speed{0.0F}; // m/s
        bool grounded{true};
        float vertical_velocity{0.0F}; // m/s, +up
        bool prone_requested{false};
        Lod lod{Lod::Full};
    };

    // To add a state: add an enumerator before `Count`, write one transition function, add one table entry.
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

    // Per-character runtime state: plain data so a crowd can keep it in one contiguous array.
    struct StateMachineState {
        State current{State::Idle};
        State previous{State::Idle};
        float phase{0.0F};
        float previous_phase{0.0F};
        float state_time{0.0F};     // seconds since entering `current`
        float fade{1.0F};           // 0..1 progress of the crossfade from `previous`; 1 == no fade
        float fade_duration{0.2F};
    };

    struct TransitionContext {
        AnimInputs const &inputs;
        State current;
        float state_time;
        bool finished; // a non-looping clip reached its end
    };

    using TransitionFunction = State (*)(TransitionContext const &);

    struct StateDefinition {
        std::string_view name;
        Clip const *clip{nullptr};
        // Metres per cycle; > 0 drives phase by distance travelled (no foot skating), 0 drives it by time.
        float stride{0.0F};
        bool loop{true};
        // Phase carries over when crossfading between two locomotion states so footfalls stay aligned.
        bool locomotion{false};
        float fade_in{0.2F}; // seconds to crossfade into this state
        TransitionFunction next{nullptr};
    };

    using StateTable = std::array<StateDefinition, state_count>;

    // Reusable transition rules (the humanoid table points at these).
    namespace Transitions {
        [[nodiscard]] auto from_ground(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_air(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_lying_down(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_prone(TransitionContext const &context) -> State;
        [[nodiscard]] auto from_getting_up(TransitionContext const &context) -> State;
    } // namespace Transitions

    // Stateless (shared by all characters); per-character data lives in StateMachineState.
    class AnimStateMachine {
    public:
        explicit AnimStateMachine(StateTable table) : table_{table} {}

        [[nodiscard]] auto definition(State state) const -> StateDefinition const & {
            return table_[static_cast<std::size_t>(state)];
        }

        // Advances `state` by `dt` and writes the resulting pose to `out`. `scratch` holds the outgoing
        // state's pose while crossfading; both views must match the skeleton's joint count.
        void update(StateMachineState &state, AnimInputs const &inputs, float dt, PoseView out, PoseView scratch) const;

    private:
        void advance_phase(State which, float &phase, float speed, float dt) const;
        StateTable table_;
    };
} // namespace Animation
