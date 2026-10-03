#pragma once

#include <volk.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

// How async compute is requested (--async-compute=auto|off|same-family). `off` still discovers and creates the
// compute queue, so the single-queue fallback can be exercised on any hardware; the frame graph compiler is what
// puts every pass on graphics.
enum class AsyncComputeMode : std::uint8_t { automatic, off, same_family };

[[nodiscard]] auto parse_async_compute_mode(std::string_view value) noexcept -> std::optional<AsyncComputeMode>;

// single: compute is the graphics queue. same_family: a second queue of the graphics family. dedicated: a family with
// COMPUTE and without GRAPHICS.
enum class QueueTopologyKind : std::uint8_t { single, same_family, dedicated };

[[nodiscard]] auto queue_topology_name(QueueTopologyKind kind) noexcept -> std::string_view;

struct QueueFamilies {
    static constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t graphics = invalid;
    std::uint32_t present = invalid;

    // Equals `graphics` unless the topology is dedicated.
    std::uint32_t compute = invalid;
    std::uint32_t compute_queue_index = 0;
    QueueTopologyKind topology = QueueTopologyKind::single;

    [[nodiscard]]
    auto complete() const noexcept -> bool {
        return graphics != invalid && present != invalid && compute != invalid;
    }
};

// What discovery needs from one VkQueueFamilyProperties plus a surface support query.
struct QueueFamilyInfo {
    VkQueueFlags flags = 0;
    std::uint32_t queue_count = 0;
    std::uint32_t timestamp_valid_bits = 0;
    bool supports_present = false;
};

// Picks the graphics and present families as before (the first family that completes the pair), then the compute
// family:
//   1. a family with COMPUTE and without GRAPHICS, preferring timestampValidBits > 0; otherwise
//   2. the graphics family's second queue, if it has queueCount >= 2; otherwise
//   3. the graphics queue itself.
// AsyncComputeMode::same_family prefers option 2 over option 1 when the graphics family has two queues.
[[nodiscard]] auto choose_queue_families(std::span<QueueFamilyInfo const> families,
                                         AsyncComputeMode mode) noexcept -> QueueFamilies;

struct QueueRequest {
    std::uint32_t family = QueueFamilies::invalid;
    std::uint32_t count = 1;

    auto operator==(QueueRequest const &) const -> bool = default;
};

// One request per unique family, in graphics, present, compute order. The graphics family asks for two queues when the
// topology is same_family. Every queue has priority 1.
[[nodiscard]] auto queue_requests(QueueFamilies const &families) -> std::vector<QueueRequest>;
