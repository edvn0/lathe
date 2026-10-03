#include "gpu/queue_selection.hxx"

#include <algorithm>

auto parse_async_compute_mode(std::string_view value) noexcept -> std::optional<AsyncComputeMode> {
    if (value == "auto") {
        return AsyncComputeMode::automatic;
    }
    if (value == "off") {
        return AsyncComputeMode::off;
    }
    if (value == "same-family") {
        return AsyncComputeMode::same_family;
    }
    return std::nullopt;
}

auto queue_topology_name(QueueTopologyKind kind) noexcept -> std::string_view {
    switch (kind) {
        case QueueTopologyKind::single:
            return "single";
        case QueueTopologyKind::same_family:
            return "same-family";
        case QueueTopologyKind::dedicated:
            return "dedicated";
    }
    return "unknown";
}

auto choose_queue_families(std::span<QueueFamilyInfo const> families, AsyncComputeMode mode) noexcept -> QueueFamilies {
    auto result = QueueFamilies{};

    for (auto index = std::uint32_t{0}; index < families.size(); ++index) {
        if ((families[index].flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            result.graphics = index;
        }
        if (families[index].supports_present) {
            result.present = index;
        }
        if (result.graphics != QueueFamilies::invalid && result.present != QueueFamilies::invalid) {
            break;
        }
    }

    if (result.graphics == QueueFamilies::invalid || result.present == QueueFamilies::invalid) {
        return result;
    }

    // Default: compute is the graphics queue.
    result.compute = result.graphics;
    result.compute_queue_index = 0;
    result.topology = QueueTopologyKind::single;

    auto const graphics_has_second_queue = families[result.graphics].queue_count >= 2;

    auto dedicated = QueueFamilies::invalid;
    for (auto index = std::uint32_t{0}; index < families.size(); ++index) {
        auto const &family = families[index];
        if ((family.flags & VK_QUEUE_COMPUTE_BIT) == 0 || (family.flags & VK_QUEUE_GRAPHICS_BIT) != 0 ||
            family.queue_count == 0) {
            continue;
        }
        // Prefer a family whose queues can write timestamps, so per-queue GPU zones work.
        if (dedicated == QueueFamilies::invalid ||
            (families[dedicated].timestamp_valid_bits == 0 && family.timestamp_valid_bits != 0)) {
            dedicated = index;
        }
    }

    auto const force_same_family = mode == AsyncComputeMode::same_family && graphics_has_second_queue;

    if (dedicated != QueueFamilies::invalid && !force_same_family) {
        result.compute = dedicated;
        result.topology = QueueTopologyKind::dedicated;
    } else if (graphics_has_second_queue) {
        result.compute_queue_index = 1;
        result.topology = QueueTopologyKind::same_family;
    }
    return result;
}

auto queue_requests(QueueFamilies const &families) -> std::vector<QueueRequest> {
    auto requests = std::vector<QueueRequest>{};
    auto const add = [&](std::uint32_t family) {
        if (family == QueueFamilies::invalid) {
            return;
        }
        if (std::ranges::none_of(requests, [&](QueueRequest const &r) { return r.family == family; })) {
            requests.push_back(QueueRequest{.family = family, .count = 1});
        }
    };
    add(families.graphics);
    add(families.present);
    add(families.compute);

    if (families.topology == QueueTopologyKind::same_family) {
        auto const graphics =
                std::ranges::find_if(requests, [&](QueueRequest const &r) { return r.family == families.graphics; });
        if (graphics != requests.end()) {
            graphics->count = 2;
        }
    }
    return requests;
}
