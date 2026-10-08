#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

struct ClusterGridSettings {
    std::uint32_t tiles_x = 16;
    std::uint32_t tiles_y = 9;
    std::uint32_t depth_slices = 24;

    std::uint32_t light_capacity = 256;

    auto operator==(ClusterGridSettings const &) const -> bool = default;
};

inline constexpr std::uint32_t cluster_grid_maximum_tiles = 64;

inline constexpr std::uint32_t cluster_grid_maximum_depth_slices = 64;

inline constexpr std::uint32_t cluster_grid_maximum_light_capacity = 1024;

inline constexpr std::uint64_t cluster_grid_maximum_buffer_bytes = std::uint64_t{128} << 20U;

inline constexpr std::uint64_t cluster_stats_bytes = 4 * sizeof(std::uint32_t);

[[nodiscard]] constexpr auto cluster_tile_count(ClusterGridSettings const &grid) noexcept -> std::uint32_t {
    return grid.tiles_x * grid.tiles_y;
}

[[nodiscard]] constexpr auto cluster_count(ClusterGridSettings const &grid) noexcept -> std::uint32_t {
    return cluster_tile_count(grid) * grid.depth_slices;
}

[[nodiscard]] constexpr auto cluster_buffer_bytes(ClusterGridSettings const &grid) noexcept -> std::uint64_t {
    return cluster_stats_bytes +
           (sizeof(std::uint32_t) * std::uint64_t{cluster_count(grid)} * (1 + std::uint64_t{grid.light_capacity}));
}

[[nodiscard]] auto validate_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string>;

struct ClusterGridPreset {
    std::string_view name;
    ClusterGridSettings grid;
};

inline constexpr std::array cluster_grid_presets{
        ClusterGridPreset{.name = "Coarse",
                          .grid = {.tiles_x = 8, .tiles_y = 5, .depth_slices = 16, .light_capacity = 256}},
        ClusterGridPreset{.name = "Default", .grid = {}},
        ClusterGridPreset{.name = "Fine",
                          .grid = {.tiles_x = 32, .tiles_y = 18, .depth_slices = 32, .light_capacity = 256}},
        ClusterGridPreset{.name = "Very fine",
                          .grid = {.tiles_x = 48, .tiles_y = 27, .depth_slices = 48, .light_capacity = 128}},
};

struct ClusterStats {
    ClusterGridSettings grid{};

    std::uint32_t occupied_clusters = 0;

    std::uint32_t overflowing_clusters = 0;

    std::uint32_t maximum_lights = 0;

    std::uint32_t stored_lights = 0;

    bool valid = false;
};
