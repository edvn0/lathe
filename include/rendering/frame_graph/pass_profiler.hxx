#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tracy/TracyVulkan.hpp>

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    struct PassTiming {
        std::string name_id;
        std::string label;
        LogicalQueue queue = LogicalQueue::graphics;
        std::optional<float> milliseconds;
    };

    struct PassProfilerCreateInfo {
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        std::array<std::uint32_t, logical_queue_count> queue_family{};
        float timestamp_period = 1.0F;
        std::uint32_t slots = 2;
        std::uint32_t max_passes = 64;
    };

    class PassProfiler {
    public:
        PassProfiler() = default;
        ~PassProfiler();

        PassProfiler(PassProfiler const &) = delete;
        auto operator=(PassProfiler const &) -> PassProfiler & = delete;

        PassProfiler(PassProfiler &&) = delete;
        auto operator=(PassProfiler &&) -> PassProfiler & = delete;

        [[nodiscard]] auto initialize(PassProfilerCreateInfo const &create_info) -> bool;
        auto destroy() -> void;

        auto begin_slot(std::uint32_t slot) -> void;

        auto write_begin(VkCommandBuffer command_buffer, LogicalQueue queue, std::uint32_t slot,
                         std::uint32_t timestamp_slot, std::string_view name_id, std::string_view label) -> void;
        auto write_end(VkCommandBuffer command_buffer, LogicalQueue queue, std::uint32_t slot,
                       std::uint32_t timestamp_slot) -> void;

        [[nodiscard]] auto timings() const noexcept -> std::span<PassTiming const> { return timings_; }

        [[nodiscard]] auto can_time(LogicalQueue queue) const noexcept -> bool {
            return valid_bits_[static_cast<std::size_t>(queue)] != 0;
        }

        [[nodiscard]] auto source_location(std::string_view label, std::uint32_t color)
                -> tracy::SourceLocationData const *;

    private:
        struct Written {
            std::string name_id;
            std::string label;
            std::uint32_t timestamp_slot = 0;
        };

        struct SlotQueue {
            VkQueryPool pool = VK_NULL_HANDLE;
            std::vector<Written> written;
        };

        VkDevice device_ = VK_NULL_HANDLE;
        float timestamp_period_ = 1.0F;
        std::uint32_t max_passes_ = 0;
        std::array<std::uint32_t, logical_queue_count> valid_bits_{};

        std::vector<std::array<SlotQueue, logical_queue_count>> pools_;
        std::vector<PassTiming> timings_;

#ifdef TRACY_ENABLE
        std::deque<std::string> names_;
        std::deque<tracy::SourceLocationData> locations_;
        std::map<std::pair<std::string, std::uint32_t>, tracy::SourceLocationData const *> interned_;
#endif
    };

}
