#include "rendering/cluster_grid.hxx"

#include <format>

namespace {

    [[nodiscard]] auto check_range(std::string_view name, std::uint32_t value, std::uint32_t maximum)
            -> std::expected<void, std::string> {
        if (value == 0 || value > maximum) {
            return std::unexpected(std::format("{} is {}, but must be 1 to {}", name, value, maximum));
        }

        return {};
    }

}

auto validate_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string> {
    auto const ranges =
            check_range("tiles_x", grid.tiles_x, cluster_grid_maximum_tiles)
                    .and_then([&] { return check_range("tiles_y", grid.tiles_y, cluster_grid_maximum_tiles); })
                    .and_then([&] {
                        return check_range("depth_slices", grid.depth_slices, cluster_grid_maximum_depth_slices);
                    })
                    .and_then([&] {
                        return check_range("light_capacity", grid.light_capacity, cluster_grid_maximum_light_capacity);
                    });

    if (!ranges) {
        return ranges;
    }

    if (auto const bytes = cluster_buffer_bytes(grid); bytes > cluster_grid_maximum_buffer_bytes) {
        return std::unexpected(
                std::format("the cluster lists would take {} MiB per frame in flight, over the {} MiB limit",
                            bytes >> 20U, cluster_grid_maximum_buffer_bytes >> 20U));
    }

    return {};
}
