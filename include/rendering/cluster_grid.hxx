#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

// The clustered-lighting grid (docs/clustered-lighting.md): tiles_x * tiles_y screen tiles in NDC, by depth_slices
// exponential depth slices, each cluster holding a sorted list of up to light_capacity light indices.
//
// A finer grid gives each fragment fewer lights to shade and overflows less, but the build has more clusters to fill
// and the lists take more memory. A coarser one is cheaper to build and wastes shading on lights that miss the pixel.
struct ClusterGridSettings {
    std::uint32_t tiles_x = 16;
    std::uint32_t tiles_y = 9;
    std::uint32_t depth_slices = 24;

    // A cluster touching more lights keeps the lowest light indices and drops the rest.
    std::uint32_t light_capacity = 256;

    auto operator==(ClusterGridSettings const &) const -> bool = default;
};

inline constexpr std::uint32_t cluster_grid_maximum_tiles = 64;

// light_cluster.slang computes one slice's bounds per lane and sizes its shared slice arrays by this.
inline constexpr std::uint32_t cluster_grid_maximum_depth_slices = 64;

inline constexpr std::uint32_t cluster_grid_maximum_light_capacity = 1024;

// Per frame in flight.
inline constexpr std::uint64_t cluster_grid_maximum_buffer_bytes = std::uint64_t{128} << 20U;

// light_cluster.slang's statistics, ahead of the counts in the cluster buffer.
inline constexpr std::uint64_t cluster_stats_bytes = 4 * sizeof(std::uint32_t);

[[nodiscard]] constexpr auto cluster_tile_count(ClusterGridSettings const &grid) noexcept -> std::uint32_t {
    return grid.tiles_x * grid.tiles_y;
}

[[nodiscard]] constexpr auto cluster_count(ClusterGridSettings const &grid) noexcept -> std::uint32_t {
    return cluster_tile_count(grid) * grid.depth_slices;
}

// The cluster buffer: the statistics, then cluster_count counts, then cluster_count * light_capacity light indices.
[[nodiscard]] constexpr auto cluster_buffer_bytes(ClusterGridSettings const &grid) noexcept -> std::uint64_t {
    return cluster_stats_bytes +
           (sizeof(std::uint32_t) * std::uint64_t{cluster_count(grid)} * (1 + std::uint64_t{grid.light_capacity}));
}

// Every dimension at least 1 and at most its maximum, and the buffer within cluster_grid_maximum_buffer_bytes;
// otherwise the reason it is not.
[[nodiscard]] auto validate_cluster_grid(ClusterGridSettings const &grid) -> std::expected<void, std::string>;

// "XxYxZ" or "XxYxZ:capacity", e.g. "32x18x32:128". The capacity defaults to 256. The result is validated.
[[nodiscard]] auto parse_cluster_grid(std::string_view text) -> std::expected<ClusterGridSettings, std::string>;

struct ClusterGridPreset {
    std::string_view name;
    ClusterGridSettings grid;
};

// Very fine's clusters are small enough to get by with a smaller capacity. Fine keeps 256: the 5,000-light
// light_field.lbf puts up to 140 lights in one of its clusters.
inline constexpr std::array cluster_grid_presets{
        ClusterGridPreset{.name = "Coarse",
                          .grid = {.tiles_x = 8, .tiles_y = 5, .depth_slices = 16, .light_capacity = 256}},
        ClusterGridPreset{.name = "Default", .grid = {}},
        ClusterGridPreset{.name = "Fine",
                          .grid = {.tiles_x = 32, .tiles_y = 18, .depth_slices = 32, .light_capacity = 256}},
        ClusterGridPreset{.name = "Very fine",
                          .grid = {.tiles_x = 48, .tiles_y = 27, .depth_slices = 48, .light_capacity = 128}},
};

// light_cluster.slang's per-frame statistics, read back a frames-in-flight cycle later.
struct ClusterStats {
    // The grid the statistics were gathered with.
    ClusterGridSettings grid{};

    // Clusters touching at least one light.
    std::uint32_t occupied_clusters = 0;

    // Clusters touching more than light_capacity lights, which drop the rest.
    std::uint32_t overflowing_clusters = 0;

    // The most lights touching one cluster, counting any it dropped.
    std::uint32_t maximum_lights = 0;

    // Light indices stored over all clusters.
    std::uint32_t stored_lights = 0;

    bool valid = false;
};
