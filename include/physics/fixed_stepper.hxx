#pragma once

#include <algorithm>

// Accumulates variable frame time into fixed simulation steps.
class FixedStepper {
public:
    explicit FixedStepper(float step_dt = 1.0F / 60.0F, int max_steps_per_frame = 5) :
        step_dt_{step_dt}, max_steps_{max_steps_per_frame} {}

    // Runs step_fn(step_dt) for each whole step owed and returns the interpolation alpha in [0, 1) for the
    // remainder. Beyond max_steps the backlog is dropped, so a long stall doesn't spiral into ever more catch-up.
    template<typename StepFn>
    auto advance(float frame_dt, StepFn &&step_fn) -> float {
        accumulator_ += std::max(frame_dt, 0.0F);

        int steps = 0;
        while (accumulator_ >= step_dt_ && steps < max_steps_) {
            step_fn(step_dt_);
            accumulator_ -= step_dt_;
            ++steps;
        }
        if (accumulator_ >= step_dt_) {
            accumulator_ = 0.0F;
        }
        return accumulator_ / step_dt_;
    }

    [[nodiscard]] auto step_dt() const noexcept -> float { return step_dt_; }

private:
    float step_dt_;
    int max_steps_;
    float accumulator_{0.0F};
};
