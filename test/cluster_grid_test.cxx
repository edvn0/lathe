#include <doctest/doctest.h>

#include "rendering/cluster_grid.hxx"

TEST_CASE("parse_cluster_grid reads XxYxZ with an optional capacity") {
    auto const grid = parse_cluster_grid("32x18x32");
    REQUIRE(grid);
    CHECK(*grid == ClusterGridSettings{.tiles_x = 32, .tiles_y = 18, .depth_slices = 32, .light_capacity = 256});

    auto const with_capacity = parse_cluster_grid("8x5x16:64");
    REQUIRE(with_capacity);
    CHECK(*with_capacity == ClusterGridSettings{.tiles_x = 8, .tiles_y = 5, .depth_slices = 16, .light_capacity = 64});
}

TEST_CASE("parse_cluster_grid rejects malformed text") {
    for (auto const *text: {"", "16x9", "16x9x", "x9x24", "16x9x24x2", "16x9x24:", "16x9x24:abc", "16 x9x24", "-1x9x24",
                            "16x9x24:128:1", "sixteen"}) {
        CAPTURE(text);
        CHECK_FALSE(parse_cluster_grid(text));
    }
}

TEST_CASE("validate_cluster_grid enforces the ranges and the buffer budget") {
    CHECK(validate_cluster_grid({}));

    CHECK_FALSE(validate_cluster_grid({.tiles_x = 0}));
    CHECK_FALSE(validate_cluster_grid({.tiles_y = cluster_grid_maximum_tiles + 1}));
    CHECK_FALSE(validate_cluster_grid({.depth_slices = cluster_grid_maximum_depth_slices + 1}));
    CHECK_FALSE(validate_cluster_grid({.light_capacity = 0}));
    CHECK_FALSE(validate_cluster_grid({.light_capacity = cluster_grid_maximum_light_capacity + 1}));

    // Every dimension at its maximum: 262,144 clusters of 1,024 lights is about 1 GiB.
    auto const largest = ClusterGridSettings{
            .tiles_x = cluster_grid_maximum_tiles,
            .tiles_y = cluster_grid_maximum_tiles,
            .depth_slices = cluster_grid_maximum_depth_slices,
            .light_capacity = cluster_grid_maximum_light_capacity,
    };
    auto const refused = validate_cluster_grid(largest);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().find("MiB") != std::string::npos);

    CHECK_FALSE(parse_cluster_grid("64x64x64:1024"));
}

TEST_CASE("cluster_buffer_bytes covers the statistics, counts and lists") {
    ClusterGridSettings const grid{.tiles_x = 2, .tiles_y = 3, .depth_slices = 4, .light_capacity = 5};

    CHECK(cluster_tile_count(grid) == 6);
    CHECK(cluster_count(grid) == 24);
    CHECK(cluster_buffer_bytes(grid) == cluster_stats_bytes + (4ULL * 24 * (1 + 5)));

    // The default grid's lists were 3.4 MiB before they became configurable.
    CHECK(cluster_buffer_bytes({}) == cluster_stats_bytes + (4ULL * 3456 * 257));
}

TEST_CASE("Every preset is valid") {
    for (auto const &preset: cluster_grid_presets) {
        CAPTURE(preset.name);
        CHECK(validate_cluster_grid(preset.grid));
    }

    CHECK(cluster_grid_presets[1].grid == ClusterGridSettings{});
}
