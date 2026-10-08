#pragma once

#include <volk.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

enum class AsyncComputeMode : std::uint8_t { automatic, off, same_family };

enum class QueueTopologyKind : std::uint8_t { single, same_family, dedicated };

[[nodiscard]] auto queue_topology_name(QueueTopologyKind kind) noexcept -> std::string_view;

struct QueueFamilies {
    static constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t graphics = invalid;
    std::uint32_t present = invalid;

    std::uint32_t compute = invalid;
    std::uint32_t compute_queue_index = 0;
    QueueTopologyKind topology = QueueTopologyKind::single;

    [[nodiscard]]
    auto complete() const noexcept -> bool {
        return graphics != invalid && present != invalid && compute != invalid;
    }
};

struct QueueFamilyInfo {
    VkQueueFlags flags = 0;
    std::uint32_t queue_count = 0;
    std::uint32_t timestamp_valid_bits = 0;
    bool supports_present = false;
};

[[nodiscard]] auto choose_queue_families(std::span<QueueFamilyInfo const> families,
                                         AsyncComputeMode mode) noexcept -> QueueFamilies;

struct QueueRequest {
    std::uint32_t family = QueueFamilies::invalid;
    std::uint32_t count = 1;

    auto operator==(QueueRequest const &) const -> bool = default;
};

[[nodiscard]] auto queue_requests(QueueFamilies const &families) -> std::vector<QueueRequest>;
