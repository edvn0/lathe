#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

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

struct CpuFrameTimes {
    std::array<float, cpu_phase_count> phase_ms{};

    float frame_ms = 0.0F;

    float present_interval_ms = 0.0F;

    [[nodiscard]] constexpr auto phase(CpuPhase which) const noexcept -> float {
        return phase_ms[static_cast<std::size_t>(which)];
    }

    [[nodiscard]] constexpr auto wait_ms() const noexcept -> float {
        auto total = 0.0F;
        for (std::size_t i = 0; i < cpu_phase_count; ++i) {
            if (cpu_phase_is_wait(static_cast<CpuPhase>(i))) {
                total += phase_ms[i];
            }
        }
        return total;
    }

    [[nodiscard]] constexpr auto busy_ms() const noexcept -> float {
        auto const busy = frame_ms - wait_ms();
        return busy > 0.0F ? busy : 0.0F;
    }
};

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

    auto begin_frame() noexcept -> void {
        frame_start_ = Clock::now();
        current_ = CpuFrameTimes{};
    }

    [[nodiscard]] auto scope(CpuPhase phase) noexcept -> Scope { return Scope{*this, phase}; }

    auto add(CpuPhase phase, Clock::duration duration) noexcept -> void {
        current_.phase_ms[static_cast<std::size_t>(phase)] += to_ms(duration);
    }

    auto mark_present() noexcept -> void {
        auto const now = Clock::now();
        current_.present_interval_ms = has_presented_ ? to_ms(now - last_present_) : 0.0F;
        last_present_ = now;
        has_presented_ = true;
    }

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
