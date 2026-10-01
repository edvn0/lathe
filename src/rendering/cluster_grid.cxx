#include "rendering/cluster_grid.hxx"

#include <charconv>
#include <format>

namespace {

    [[nodiscard]] auto parse_dimension(std::string_view text, std::string_view whole)
            -> std::expected<std::uint32_t, std::string> {
        std::uint32_t value = 0;
        auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);

        if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
            return std::unexpected(std::format("'{}' is not XxYxZ or XxYxZ:capacity", whole));
        }

        return value;
    }

    [[nodiscard]] auto check_range(std::string_view name, std::uint32_t value, std::uint32_t maximum)
            -> std::expected<void, std::string> {
        if (value == 0 || value > maximum) {
            return std::unexpected(std::format("{} is {}, but must be 1 to {}", name, value, maximum));
        }

        return {};
    }

} // namespace

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

auto parse_cluster_grid(std::string_view text) -> std::expected<ClusterGridSettings, std::string> {
    ClusterGridSettings grid{};

    auto dimensions = text;

    if (auto const colon = text.find(':'); colon != std::string_view::npos) {
        dimensions = text.substr(0, colon);

        auto const capacity = parse_dimension(text.substr(colon + 1), text);
        if (!capacity) {
            return std::unexpected(capacity.error());
        }

        grid.light_capacity = *capacity;
    }

    auto const first_x = dimensions.find('x');
    auto const second_x = first_x == std::string_view::npos ? first_x : dimensions.find('x', first_x + 1);

    if (second_x == std::string_view::npos) {
        return std::unexpected(std::format("'{}' is not XxYxZ or XxYxZ:capacity", text));
    }

    auto const tiles_x = parse_dimension(dimensions.substr(0, first_x), text);
    auto const tiles_y = parse_dimension(dimensions.substr(first_x + 1, second_x - first_x - 1), text);
    auto const depth_slices = parse_dimension(dimensions.substr(second_x + 1), text);

    for (auto const *dimension: {&tiles_x, &tiles_y, &depth_slices}) {
        if (!*dimension) {
            return std::unexpected(dimension->error());
        }
    }

    grid.tiles_x = *tiles_x;
    grid.tiles_y = *tiles_y;
    grid.depth_slices = *depth_slices;

    if (auto const valid = validate_cluster_grid(grid); !valid) {
        return std::unexpected(valid.error());
    }

    return grid;
}
