#include <doctest/doctest.h>

#include "terrain/terrain_greedy_mesh.hxx"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace {

    struct Grid {
        std::uint32_t samples_x = 0;
        std::uint32_t samples_z = 0;
        std::vector<float> heights;

        [[nodiscard]] auto at(std::uint32_t column, std::uint32_t row) const -> float {
            return heights[static_cast<std::size_t>(row) * samples_x + column];
        }
    };

    template<typename HeightFn>
    [[nodiscard]] auto make_grid(std::uint32_t samples_x, std::uint32_t samples_z, HeightFn &&height_fn) -> Grid {
        Grid grid{.samples_x = samples_x, .samples_z = samples_z, .heights = {}};
        grid.heights.reserve(static_cast<std::size_t>(samples_x) * samples_z);

        for (std::uint32_t row = 0; row < samples_z; ++row) {
            for (std::uint32_t column = 0; column < samples_x; ++column) {
                grid.heights.push_back(height_fn(static_cast<float>(column), static_cast<float>(row)));
            }
        }

        return grid;
    }

    // Plateaus, a ramp and some rough ground, so merges of every shape show up next to single cells.
    [[nodiscard]] auto mixed_grid() -> Grid {
        return make_grid(33, 29, [](float x, float z) {
            if (x < 10.0F) {
                return 2.0F;
            }
            if (z < 12.0F) {
                return 0.25F * x - 0.5F * z;
            }
            return std::sin(x * 0.9F) * std::cos(z * 1.3F);
        });
    }

    struct Point {
        std::uint32_t column = 0;
        std::uint32_t row = 0;
    };

    [[nodiscard]] auto point_of(Grid const &grid, std::uint32_t index) -> Point {
        return Point{.column = index % grid.samples_x, .row = index / grid.samples_x};
    }

    // Twice the signed area in (column, row); negative for the winding an unmerged cell uses.
    [[nodiscard]] auto doubled_area(Point a, Point b, Point c) -> std::int64_t {
        return (static_cast<std::int64_t>(b.column) - a.column) * (static_cast<std::int64_t>(c.row) - a.row) -
               (static_cast<std::int64_t>(b.row) - a.row) * (static_cast<std::int64_t>(c.column) - a.column);
    }

    [[nodiscard]] auto is_boundary_edge(Grid const &grid, Point a, Point b) -> bool {
        auto const on_column = [](Point p, std::uint32_t column) { return p.column == column; };
        auto const on_row = [](Point p, std::uint32_t row) { return p.row == row; };

        return (on_column(a, 0) && on_column(b, 0)) ||
               (on_column(a, grid.samples_x - 1) && on_column(b, grid.samples_x - 1)) ||
               (on_row(a, 0) && on_row(b, 0)) || (on_row(a, grid.samples_z - 1) && on_row(b, grid.samples_z - 1));
    }

    // Each directed edge must appear once and, away from the grid boundary, its reverse once too: a manifold,
    // consistently wound surface. A T-junction leaves a long edge whose reverse is split in pieces, so it fails here.
    auto check_watertight(Grid const &grid, std::vector<std::uint32_t> const &indices) -> void {
        std::map<std::pair<std::uint32_t, std::uint32_t>, int> directed;

        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            for (std::size_t corner = 0; corner < 3; ++corner) {
                ++directed[{indices[i + corner], indices[i + (corner + 1) % 3]}];
            }
        }

        for (auto const &[edge, count]: directed) {
            CHECK(count == 1);

            auto const a = point_of(grid, edge.first);
            auto const b = point_of(grid, edge.second);

            if (is_boundary_edge(grid, a, b)) {
                // Boundary edges stay unit length so skirts and neighbours line up with every vertex.
                CHECK(std::max(a.column, b.column) - std::min(a.column, b.column) + std::max(a.row, b.row) -
                              std::min(a.row, b.row) ==
                      1U);
                CHECK_FALSE(directed.contains({edge.second, edge.first}));
            } else {
                CHECK(directed.contains({edge.second, edge.first}));
            }
        }
    }

    auto check_covers_grid_once(Grid const &grid, std::vector<TerrainGreedyQuad> const &quads) -> void {
        auto const cells_x = grid.samples_x - 1;
        auto const cells_z = grid.samples_z - 1;
        std::vector<int> coverage(static_cast<std::size_t>(cells_x) * cells_z, 0);

        for (auto const &quad: quads) {
            REQUIRE(quad.width >= 1);
            REQUIRE(quad.height >= 1);
            REQUIRE(quad.column + quad.width <= cells_x);
            REQUIRE(quad.row + quad.height <= cells_z);

            for (std::uint32_t row = quad.row; row < quad.row + quad.height; ++row) {
                for (std::uint32_t column = quad.column; column < quad.column + quad.width; ++column) {
                    ++coverage[static_cast<std::size_t>(row) * cells_x + column];
                }
            }
        }

        CHECK(std::ranges::all_of(coverage, [](int count) { return count == 1; }));
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("greedy meshing collapses a tilted plane to one quad") {
        auto const grid = make_grid(65, 65, [](float x, float z) { return 0.5F * x - 0.25F * z + 3.0F; });

        auto const quads = greedy_merge_terrain_cells(grid.heights, grid.samples_x, grid.samples_z, 1e-3F);
        REQUIRE(quads.size() == 1);
        CHECK(quads[0].width == 64);
        CHECK(quads[0].height == 64);

        // The boundary stays at full resolution, so the one quad is stitched along all four edges.
        auto const indices = triangulate_terrain_quads(quads, grid.samples_x, grid.samples_z);
        CHECK(indices.size() / 3 == 4U * 64U - 2U);
        check_watertight(grid, indices);
    }

    TEST_CASE("zero tolerance on rough ground keeps every cell") {
        auto const grid = make_grid(17, 17, [](float x, float z) { return std::sin(x * 1.7F) + std::cos(z * 2.3F); });

        auto const quads = greedy_merge_terrain_cells(grid.heights, grid.samples_x, grid.samples_z, 0.0F);
        CHECK(quads.size() == 16U * 16U);

        // Unmerged cells triangulate exactly like the plain grid.
        auto const indices = triangulate_terrain_quads(quads, grid.samples_x, grid.samples_z);
        CHECK(indices.size() == 16U * 16U * 6U);
        check_watertight(grid, indices);
    }

    TEST_CASE("greedy quads tile the grid and triangulate without T-junctions") {
        auto const grid = mixed_grid();

        for (auto const tolerance: {0.0F, 0.05F, 0.5F, 100.0F}) {
            CAPTURE(tolerance);

            auto const quads = greedy_merge_terrain_cells(grid.heights, grid.samples_x, grid.samples_z, tolerance);
            check_covers_grid_once(grid, quads);

            auto const indices = triangulate_terrain_quads(quads, grid.samples_x, grid.samples_z);
            REQUIRE(indices.size() % 3 == 0);
            CHECK(indices.size() <= std::size_t{grid.samples_x - 1} * (grid.samples_z - 1) * 6);

            std::int64_t doubled_total = 0;

            for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
                auto const area = doubled_area(point_of(grid, indices[i]), point_of(grid, indices[i + 1]),
                                               point_of(grid, indices[i + 2]));
                CHECK(area < 0); // non-degenerate, and wound like an unmerged cell
                doubled_total -= area;
            }

            // Non-overlapping (watertight, below) and summing to the grid's area: the grid is covered exactly.
            CHECK(doubled_total == 2 * std::int64_t{grid.samples_x - 1} * (grid.samples_z - 1));
            check_watertight(grid, indices);
        }
    }

    TEST_CASE("merged quads keep every sample within tolerance of the plane through their corners") {
        auto const grid = mixed_grid();
        constexpr auto tolerance = 0.05F;

        auto const quads = greedy_merge_terrain_cells(grid.heights, grid.samples_x, grid.samples_z, tolerance);

        auto merged = 0;

        for (auto const &quad: quads) {
            if (quad.width == 1 && quad.height == 1) {
                continue;
            }
            ++merged;

            auto const origin = grid.at(quad.column, quad.row);
            auto const slope_x =
                    (grid.at(quad.column + quad.width, quad.row) - origin) / static_cast<float>(quad.width);
            auto const slope_z =
                    (grid.at(quad.column, quad.row + quad.height) - origin) / static_cast<float>(quad.height);

            for (std::uint32_t dz = 0; dz <= quad.height; ++dz) {
                for (std::uint32_t dx = 0; dx <= quad.width; ++dx) {
                    auto const expected = origin + slope_x * static_cast<float>(dx) + slope_z * static_cast<float>(dz);
                    CHECK(std::fabs(grid.at(quad.column + dx, quad.row + dz) - expected) <= tolerance);
                }
            }
        }

        // The plateau and the ramp both merge.
        CHECK(merged >= 2);
    }
}
