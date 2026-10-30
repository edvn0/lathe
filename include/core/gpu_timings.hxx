#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Smoothed per-pass GPU timings with a short history, fed one frame at a time.
// Render/UI thread only; not synchronised.
class GpuTimings {
public:
    static constexpr std::size_t history_length = 120;
    static constexpr float smoothing = 0.05F;

    struct Entry {
        std::string id;
        std::string label;
        // Which timeline the pass ran on (0 graphics, 1 compute, 2 overlays); the view groups by it.
        std::uint8_t lane = 0;
        bool active = false;
        // Active on the previous frame, so the average carries on rather than restarting.
        bool primed = false;
        float ema_ms = 0.0F;
        float last_ms = 0.0F;
        // Ring buffer of raw samples, oldest at `history_head` once full.
        std::array<float, history_length> history{};
        std::size_t history_head = 0;
        std::size_t history_count = 0;

        [[nodiscard]] auto samples() const noexcept -> std::size_t { return history_count; }

        // i-th oldest sample.
        [[nodiscard]] auto sample(std::size_t i) const noexcept -> float {
            auto const start = history_count < history_length ? 0 : history_head;
            return history[(start + i) % history_length];
        }

        [[nodiscard]] auto min_ms() const noexcept -> float {
            return history_count == 0 ? 0.0F : *std::min_element(history.begin(), history.begin() + count());
        }

        [[nodiscard]] auto max_ms() const noexcept -> float {
            return history_count == 0 ? 0.0F : *std::max_element(history.begin(), history.begin() + count());
        }

    private:
        [[nodiscard]] auto count() const noexcept -> std::ptrdiff_t { return static_cast<std::ptrdiff_t>(history_count); }
    };

    // Marks every pass idle; `record` revives the ones that ran this frame.
    auto begin_frame() noexcept -> void {
        for (auto &entry: entries_) {
            entry.primed = entry.active;
            entry.active = false;
        }
    }

    // A pass with no measurement this frame (queue can't time, or its query wasn't ready) stays idle.
    auto record(std::string_view id, std::string_view label, std::uint8_t lane, std::optional<float> ms) -> void {
        auto &entry = find_or_add(id, label, lane);
        if (!ms) {
            return;
        }

        // Restart the average when the pass comes back from idle, so stale numbers don't drag it.
        entry.ema_ms = entry.primed ? entry.ema_ms + smoothing * (*ms - entry.ema_ms) : *ms;
        entry.last_ms = *ms;
        entry.active = true;

        entry.history[entry.history_head] = *ms;
        entry.history_head = (entry.history_head + 1) % history_length;
        entry.history_count = std::min(entry.history_count + 1, history_length);
    }

    [[nodiscard]] auto entries() const noexcept -> std::span<Entry const> { return entries_; }

    [[nodiscard]] auto find(std::string_view id) const noexcept -> Entry const * {
        auto const found = std::ranges::find_if(entries_, [&](Entry const &entry) { return entry.id == id; });
        return found == entries_.end() ? nullptr : &*found;
    }

    // Sum of the smoothed times of the active passes in one lane.
    [[nodiscard]] auto lane_total_ms(std::uint8_t lane) const noexcept -> float {
        auto total = 0.0F;
        for (auto const &entry: entries_) {
            if (entry.active && entry.lane == lane) {
                total += entry.ema_ms;
            }
        }
        return total;
    }

    [[nodiscard]] auto total_ms() const noexcept -> float {
        auto total = 0.0F;
        for (auto const &entry: entries_) {
            if (entry.active) {
                total += entry.ema_ms;
            }
        }
        return total;
    }

    auto clear() noexcept -> void { entries_.clear(); }

private:
    auto find_or_add(std::string_view id, std::string_view label, std::uint8_t lane) -> Entry & {
        auto const found = std::ranges::find_if(entries_, [&](Entry const &entry) { return entry.id == id; });
        if (found != entries_.end()) {
            found->label = label;
            found->lane = lane;
            return *found;
        }

        auto &entry = entries_.emplace_back();
        entry.id = id;
        entry.label = label;
        entry.lane = lane;
        return entry;
    }

    std::vector<Entry> entries_;
};
