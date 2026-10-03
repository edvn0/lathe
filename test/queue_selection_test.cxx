#include <doctest/doctest.h>

#include <array>

#include "gpu/queue_selection.hxx"

namespace {

    constexpr auto graphics_family = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    constexpr auto compute_family = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;

    auto info(VkQueueFlags flags, std::uint32_t count, std::uint32_t timestamp_bits = 64,
              bool present = false) -> QueueFamilyInfo {
        return QueueFamilyInfo{
                .flags = flags,
                .queue_count = count,
                .timestamp_valid_bits = timestamp_bits,
                .supports_present = present,
        };
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("one graphics queue is the single topology") {
        auto const families = std::array{info(graphics_family, 1, 64, true)};
        auto const chosen = choose_queue_families(families, AsyncComputeMode::automatic);
        CHECK(chosen.complete());
        CHECK(chosen.graphics == 0);
        CHECK(chosen.present == 0);
        CHECK(chosen.compute == 0);
        CHECK(chosen.compute_queue_index == 0);
        CHECK(chosen.topology == QueueTopologyKind::single);
    }

    TEST_CASE("a graphics family with two queues is the same-family topology") {
        auto const families = std::array{info(graphics_family, 2, 64, true)};
        auto const chosen = choose_queue_families(families, AsyncComputeMode::automatic);
        CHECK(chosen.compute == 0);
        CHECK(chosen.compute_queue_index == 1);
        CHECK(chosen.topology == QueueTopologyKind::same_family);
    }

    TEST_CASE("a compute-only family is the dedicated topology") {
        auto const families = std::array{info(graphics_family, 1, 64, true), info(compute_family, 4)};
        auto const chosen = choose_queue_families(families, AsyncComputeMode::automatic);
        CHECK(chosen.graphics == 0);
        CHECK(chosen.compute == 1);
        CHECK(chosen.compute_queue_index == 0);
        CHECK(chosen.topology == QueueTopologyKind::dedicated);
    }

    TEST_CASE("dedicated wins over a second graphics queue unless same-family is forced") {
        auto const families = std::array{info(graphics_family, 2, 64, true), info(compute_family, 4)};
        CHECK(choose_queue_families(families, AsyncComputeMode::automatic).topology == QueueTopologyKind::dedicated);
        CHECK(choose_queue_families(families, AsyncComputeMode::off).topology == QueueTopologyKind::dedicated);

        auto const forced = choose_queue_families(families, AsyncComputeMode::same_family);
        CHECK(forced.topology == QueueTopologyKind::same_family);
        CHECK(forced.compute == 0);
        CHECK(forced.compute_queue_index == 1);
    }

    TEST_CASE("forcing same-family without a second queue falls back") {
        auto const without_dedicated = std::array{info(graphics_family, 1, 64, true)};
        CHECK(choose_queue_families(without_dedicated, AsyncComputeMode::same_family).topology ==
              QueueTopologyKind::single);

        auto const with_dedicated = std::array{info(graphics_family, 1, 64, true), info(compute_family, 2)};
        CHECK(choose_queue_families(with_dedicated, AsyncComputeMode::same_family).topology ==
              QueueTopologyKind::dedicated);
    }

    TEST_CASE("a dedicated family that can write timestamps is preferred") {
        auto const families = std::array{
                info(graphics_family, 1, 64, true),
                info(compute_family, 2, 0),
                info(compute_family, 2, 64),
        };
        auto const chosen = choose_queue_families(families, AsyncComputeMode::automatic);
        CHECK(chosen.compute == 2);

        auto const only_untimed = std::array{info(graphics_family, 1, 64, true), info(compute_family, 2, 0)};
        CHECK(choose_queue_families(only_untimed, AsyncComputeMode::automatic).compute == 1);
    }

    TEST_CASE("a family with graphics is never a dedicated compute family") {
        auto const families = std::array{info(graphics_family, 1, 64, true), info(graphics_family, 1)};
        CHECK(choose_queue_families(families, AsyncComputeMode::automatic).topology == QueueTopologyKind::single);
    }

    TEST_CASE("graphics and present keep the first complete pair") {
        auto const families = std::array{info(graphics_family, 1), info(compute_family, 1, 64, true)};
        auto const chosen = choose_queue_families(families, AsyncComputeMode::automatic);
        CHECK(chosen.graphics == 0);
        CHECK(chosen.present == 1);
        CHECK(chosen.compute == 1);
        CHECK(chosen.topology == QueueTopologyKind::dedicated);
    }

    TEST_CASE("no present support is incomplete") {
        auto const families = std::array{info(graphics_family, 2)};
        CHECK_FALSE(choose_queue_families(families, AsyncComputeMode::automatic).complete());
        CHECK_FALSE(choose_queue_families({}, AsyncComputeMode::automatic).complete());
    }

    TEST_CASE("queue requests are one per unique family") {
        auto const single =
                choose_queue_families(std::array{info(graphics_family, 1, 64, true)}, AsyncComputeMode::automatic);
        CHECK(queue_requests(single) == std::vector<QueueRequest>{{0, 1}});

        auto const same_family =
                choose_queue_families(std::array{info(graphics_family, 2, 64, true)}, AsyncComputeMode::automatic);
        CHECK(queue_requests(same_family) == std::vector<QueueRequest>{{0, 2}});

        auto const dedicated = choose_queue_families(
                std::array{info(graphics_family, 1, 64, true), info(compute_family, 4)}, AsyncComputeMode::automatic);
        CHECK(queue_requests(dedicated) == std::vector<QueueRequest>{{0, 1}, {1, 1}});

        // The present family can also be the dedicated compute family: one request covers both.
        auto const shared_family = choose_queue_families(
                std::array{info(graphics_family, 2), info(compute_family, 1, 64, true), info(compute_family, 4)},
                AsyncComputeMode::automatic);
        CHECK(shared_family.present == 1);
        CHECK(shared_family.compute == 1);
        CHECK(queue_requests(shared_family) == std::vector<QueueRequest>{{0, 1}, {1, 1}});

        // Graphics, present and compute on three different families.
        auto const three = choose_queue_families(
                std::array{info(VK_QUEUE_GRAPHICS_BIT, 1), info(0, 1, 0, true), info(compute_family, 2)},
                AsyncComputeMode::automatic);
        CHECK(three.graphics == 0);
        CHECK(three.present == 1);
        CHECK(three.compute == 2);
        CHECK(queue_requests(three) == std::vector<QueueRequest>{{0, 1}, {1, 1}, {2, 1}});
    }

    TEST_CASE("--async-compute values parse") {
        CHECK(parse_async_compute_mode("auto") == AsyncComputeMode::automatic);
        CHECK(parse_async_compute_mode("off") == AsyncComputeMode::off);
        CHECK(parse_async_compute_mode("same-family") == AsyncComputeMode::same_family);
        CHECK_FALSE(parse_async_compute_mode("on").has_value());
        CHECK_FALSE(parse_async_compute_mode("").has_value());
    }

    TEST_CASE("topology names") {
        CHECK(queue_topology_name(QueueTopologyKind::single) == "single");
        CHECK(queue_topology_name(QueueTopologyKind::same_family) == "same-family");
        CHECK(queue_topology_name(QueueTopologyKind::dedicated) == "dedicated");
    }
}
