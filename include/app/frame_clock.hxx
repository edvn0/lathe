#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

// Where the main loop's CPU time goes in one frame. The wait phases are time the CPU spends blocked rather than
// working: slot_wait is the frame slot's timeline wait (the GPU is still busy with the frame from frames_in_flight
// ago, i.e. the CPU is ahead), acquire is vkAcquireNextImageKHR (the presentation engine has no free image) and
// present is vkQueuePresentKHR, which blocks under FIFO and on some compositors.
enum class CpuPhase : std::uint8_t {
    events,
    update,
    slot_wait,
    acquire,
    scene_submit,
    ui,
    prepare,
    record,
    submit,
    present,
    count,
};

inline constexpr std::size_t cpu_phase_count = static_cast<std::size_t>(CpuPhase::count);

[[nodiscard]] constexpr auto cpu_phase_name(CpuPhase phase) noexcept -> std::string_view {
    switch (phase) {
        case CpuPhase::events:
            return "events";
        case CpuPhase::update:
            return "update";
        case CpuPhase::slot_wait:
            return "slot_wait";
        case CpuPhase::acquire:
            return "acquire";
        case CpuPhase::scene_submit:
            return "scene_submit";
        case CpuPhase::ui:
            return "ui";
        case CpuPhase::prepare:
            return "prepare";
        case CpuPhase::record:
            return "record";
        case CpuPhase::submit:
            return "submit";
        case CpuPhase::present:
            return "present";
        case CpuPhase::count:
            break;
    }
    return "unknown";
}

[[nodiscard]] constexpr auto cpu_phase_is_wait(CpuPhase phase) noexcept -> bool {
    return phase == CpuPhase::slot_wait || phase == CpuPhase::acquire || phase == CpuPhase::present;
}

// One frame's CPU timings, in milliseconds.
struct CpuFrameTimes {
    std::array<float, cpu_phase_count> phase_ms{};

    // Main loop iteration, from the top of the loop to the point the frame was handed to the benchmark.
    float frame_ms = 0.0F;

    // Present to present: the wall time between this frame's vkQueuePresentKHR returning and the previous one's. The
    // closest thing the CPU can see to what a player sees; 0 for the first frame.
    float present_interval_ms = 0.0F;

    [[nodiscard]] constexpr auto phase(CpuPhase which) const noexcept -> float {
        return phase_ms[static_cast<std::size_t>(which)];
    }

    // Time blocked on the GPU or the presentation engine.
    [[nodiscard]] constexpr auto wait_ms() const noexcept -> float {
        auto total = 0.0F;
        for (std::size_t i = 0; i < cpu_phase_count; ++i) {
            if (cpu_phase_is_wait(static_cast<CpuPhase>(i))) {
                total += phase_ms[i];
            }
        }
        return total;
    }

    // Time the CPU spent working: the frame minus its waits.
    [[nodiscard]] constexpr auto busy_ms() const noexcept -> float {
        auto const busy = frame_ms - wait_ms();
        return busy > 0.0F ? busy : 0.0F;
    }
};

// Accumulates CpuFrameTimes for the main loop. Two steady_clock reads per phase, so it is always on: it costs well
// under a microsecond per frame, against the milliseconds it measures.
class FrameClock {
public:
    using Clock = std::chrono::steady_clock;

    class Scope {
    public:
        Scope(FrameClock &clock, CpuPhase phase) noexcept : clock_(&clock), phase_(phase), start_(Clock::now()) {}
        ~Scope() { clock_->add(phase_, Clock::now() - start_); }

        Scope(Scope const &) = delete;
        Scope(Scope &&) = delete;
        auto operator=(Scope const &) -> Scope & = delete;
        auto operator=(Scope &&) -> Scope & = delete;

    private:
        FrameClock *clock_;
        CpuPhase phase_;
        Clock::time_point start_;
    };

    // Top of the main loop: starts a new frame.
    auto begin_frame() noexcept -> void {
        frame_start_ = Clock::now();
        current_ = CpuFrameTimes{};
    }

    [[nodiscard]] auto scope(CpuPhase phase) noexcept -> Scope { return Scope{*this, phase}; }

    auto add(CpuPhase phase, Clock::duration duration) noexcept -> void {
        current_.phase_ms[static_cast<std::size_t>(phase)] += to_ms(duration);
    }

    // Right after vkQueuePresentKHR returned.
    auto mark_present() noexcept -> void {
        auto const now = Clock::now();
        current_.present_interval_ms = has_presented_ ? to_ms(now - last_present_) : 0.0F;
        last_present_ = now;
        has_presented_ = true;
    }

    // The frame so far, with frame_ms measured up to now.
    [[nodiscard]] auto finish_frame() noexcept -> CpuFrameTimes {
        current_.frame_ms = to_ms(Clock::now() - frame_start_);
        return current_;
    }

private:
    [[nodiscard]] static auto to_ms(Clock::duration duration) noexcept -> float {
        return std::chrono::duration<float, std::milli>(duration).count();
    }

    Clock::time_point frame_start_{};
    Clock::time_point last_present_{};
    bool has_presented_ = false;
    CpuFrameTimes current_{};
};
