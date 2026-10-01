#include "terrain/terrain_greedy_mesh.hxx"

#include <cmath>
#include <cstddef>
#include <utility>

namespace {

    struct GridPoint {
        std::uint32_t column = 0;
        std::uint32_t row = 0;
    };

    class HeightGrid {
    public:
        HeightGrid(std::span<float const> heights, std::uint32_t samples_x) :
            heights_{heights}, samples_x_{samples_x} {}

        [[nodiscard]] auto at(std::uint32_t column, std::uint32_t row) const -> float {
            return heights_[static_cast<std::size_t>(row) * samples_x_ + column];
        }

    private:
        std::span<float const> heights_;
        std::uint32_t samples_x_ = 0;
    };

    // Plane through the min/min, max/min and min/max corners. The max/max corner and every interior sample are
    // checked against it like any other sample.
    [[nodiscard]] auto is_planar(HeightGrid const &grid, TerrainGreedyQuad const &quad, float tolerance) -> bool {
        if (quad.width == 1 && quad.height == 1) {
            return true;
        }

        auto const origin = grid.at(quad.column, quad.row);
        auto const slope_x = (grid.at(quad.column + quad.width, quad.row) - origin) / static_cast<float>(quad.width);
        auto const slope_z = (grid.at(quad.column, quad.row + quad.height) - origin) / static_cast<float>(quad.height);

        for (std::uint32_t dz = 0; dz <= quad.height; ++dz) {
            for (std::uint32_t dx = 0; dx <= quad.width; ++dx) {
                auto const expected = origin + slope_x * static_cast<float>(dx) + slope_z * static_cast<float>(dz);

                if (std::fabs(grid.at(quad.column + dx, quad.row + dz) - expected) > tolerance) {
                    return false;
                }
            }
        }

        return true;
    }

    // Emits a, b, c wound like an unmerged cell's (min/min, min/max, max/max) triangle: clockwise in (column, row),
    // which is front-facing +Y with X = column and Z = row.
    auto emit_triangle(std::vector<std::uint32_t> &indices, std::uint32_t samples_x, GridPoint a, GridPoint b,
                       GridPoint c) -> void {
        auto const signed_area =
                (static_cast<std::int64_t>(b.column) - a.column) * (static_cast<std::int64_t>(c.row) - a.row) -
                (static_cast<std::int64_t>(b.row) - a.row) * (static_cast<std::int64_t>(c.column) - a.column);

        if (signed_area > 0) {
            std::swap(b, c);
        }

        auto const index_of = [samples_x](GridPoint point) { return point.row * samples_x + point.column; };

        indices.push_back(index_of(a));
        indices.push_back(index_of(b));
        indices.push_back(index_of(c));
    }

    struct QuadEdges {
        std::vector<std::uint32_t> bottom; // columns on row `row`, ascending, both corners included
        std::vector<std::uint32_t> top; // columns on row `row + height`, ascending, both corners included
        std::vector<std::uint32_t> left; // rows strictly between the corners on column `column`, ascending
        std::vector<std::uint32_t> right; // rows strictly between the corners on column `column + width`, ascending
    };

    // Left and right edge points are fanned to the nearest bottom/top point, then the trapezoid left between the
    // bottom and top chains is zig-zagged. Every triangle has two points on one grid line and the third off it, so
    // none are degenerate.
    auto triangulate_quad(std::vector<std::uint32_t> &indices, std::uint32_t samples_x, TerrainGreedyQuad const &quad,
                          QuadEdges const &edges) -> void {
        auto const min_column = quad.column;
        auto const max_column = quad.column + quad.width;
        auto const min_row = quad.row;
        auto const max_row = quad.row + quad.height;

        auto const emit = [&](GridPoint a, GridPoint b, GridPoint c) { emit_triangle(indices, samples_x, a, b, c); };

        if (edges.bottom.size() == 2 && edges.top.size() == 2 && edges.left.empty() && edges.right.empty()) {
            // Same diagonal as an unmerged cell.
            emit({min_column, min_row}, {min_column, max_row}, {max_column, max_row});
            emit({min_column, min_row}, {max_column, max_row}, {max_column, min_row});
            return;
        }

        auto const fan_edge = [&](std::uint32_t edge_column, std::vector<std::uint32_t> const &rows, GridPoint apex) {
            GridPoint previous{edge_column, min_row};

            for (auto const row: rows) {
                GridPoint const next{edge_column, row};
                emit(previous, next, apex);
                previous = next;
            }

            emit(previous, {edge_column, max_row}, apex);
        };

        std::size_t bottom_begin = 0;
        std::size_t top_end = edges.top.size() - 1;

        if (!edges.left.empty()) {
            fan_edge(min_column, edges.left, GridPoint{edges.bottom[1], min_row});
            bottom_begin = 1;
        }

        if (!edges.right.empty()) {
            top_end = edges.top.size() - 2;
            fan_edge(max_column, edges.right, GridPoint{edges.top[top_end], max_row});
        }

        auto const bottom_end = edges.bottom.size() - 1;
        auto bottom = bottom_begin;
        std::size_t top = 0;

        while (bottom < bottom_end || top < top_end) {
            auto const advance_bottom =
                    top == top_end || (bottom < bottom_end && edges.bottom[bottom + 1] <= edges.top[top + 1]);

            if (advance_bottom) {
                emit({edges.bottom[bottom], min_row}, {edges.bottom[bottom + 1], min_row}, {edges.top[top], max_row});
                ++bottom;
            } else {
                emit({edges.bottom[bottom], min_row}, {edges.top[top + 1], max_row}, {edges.top[top], max_row});
                ++top;
            }
        }
    }

} // namespace

auto greedy_merge_terrain_cells(std::span<float const> heights, std::uint32_t samples_x, std::uint32_t samples_z,
                                float tolerance) -> std::vector<TerrainGreedyQuad> {
    std::vector<TerrainGreedyQuad> quads;

    if (samples_x < 2 || samples_z < 2 || heights.size() < static_cast<std::size_t>(samples_x) * samples_z) {
        return quads;
    }

    auto const cells_x = samples_x - 1;
    auto const cells_z = samples_z - 1;

    HeightGrid const grid{heights, samples_x};
    std::vector<std::uint8_t> covered(static_cast<std::size_t>(cells_x) * cells_z, 0);

    auto const is_covered = [&](std::uint32_t column, std::uint32_t row) {
        return covered[static_cast<std::size_t>(row) * cells_x + column] != 0;
    };

    for (std::uint32_t row = 0; row < cells_z; ++row) {
        for (std::uint32_t column = 0; column < cells_x; ++column) {
            if (is_covered(column, row)) {
                continue;
            }

            TerrainGreedyQuad quad{.column = column, .row = row, .width = 1, .height = 1};

            while (quad.column + quad.width < cells_x && !is_covered(quad.column + quad.width, row)) {
                auto wider = quad;
                ++wider.width;

                if (!is_planar(grid, wider, tolerance)) {
                    break;
                }

                quad = wider;
            }

            auto const row_is_free = [&](std::uint32_t candidate_row) {
                for (std::uint32_t dx = 0; dx < quad.width; ++dx) {
                    if (is_covered(quad.column + dx, candidate_row)) {
                        return false;
                    }
                }
                return true;
            };

            while (quad.row + quad.height < cells_z && row_is_free(quad.row + quad.height)) {
                auto taller = quad;
                ++taller.height;

                if (!is_planar(grid, taller, tolerance)) {
                    break;
                }

                quad = taller;
            }

            for (std::uint32_t dz = 0; dz < quad.height; ++dz) {
                for (std::uint32_t dx = 0; dx < quad.width; ++dx) {
                    covered[static_cast<std::size_t>(quad.row + dz) * cells_x + quad.column + dx] = 1;
                }
            }

            quads.push_back(quad);
        }
    }

    return quads;
}

auto triangulate_terrain_quads(std::span<TerrainGreedyQuad const> quads, std::uint32_t samples_x,
                               std::uint32_t samples_z) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> indices;

    if (samples_x < 2 || samples_z < 2) {
        return indices;
    }

    // Vertices some triangle passes through: every quad corner, plus the whole grid boundary.
    std::vector<std::uint8_t> pinned(static_cast<std::size_t>(samples_x) * samples_z, 0);

    auto const pin = [&](std::uint32_t column, std::uint32_t row) {
        pinned[static_cast<std::size_t>(row) * samples_x + column] = 1;
    };
    auto const is_pinned = [&](std::uint32_t column, std::uint32_t row) {
        return pinned[static_cast<std::size_t>(row) * samples_x + column] != 0;
    };

    for (std::uint32_t column = 0; column < samples_x; ++column) {
        pin(column, 0);
        pin(column, samples_z - 1);
    }

    for (std::uint32_t row = 0; row < samples_z; ++row) {
        pin(0, row);
        pin(samples_x - 1, row);
    }

    for (auto const &quad: quads) {
        pin(quad.column, quad.row);
        pin(quad.column + quad.width, quad.row);
        pin(quad.column, quad.row + quad.height);
        pin(quad.column + quad.width, quad.row + quad.height);
    }

    // Single cells are the common case on rough terrain: two triangles each.
    indices.reserve(quads.size() * 6);

    QuadEdges edges;

    for (auto const &quad: quads) {
        auto const max_column = quad.column + quad.width;
        auto const max_row = quad.row + quad.height;

        edges.bottom.clear();
        edges.top.clear();
        edges.left.clear();
        edges.right.clear();

        for (auto column = quad.column; column <= max_column; ++column) {
            if (is_pinned(column, quad.row)) {
                edges.bottom.push_back(column);
            }
            if (is_pinned(column, max_row)) {
                edges.top.push_back(column);
            }
        }

        for (auto row = quad.row + 1; row < max_row; ++row) {
            if (is_pinned(quad.column, row)) {
                edges.left.push_back(row);
            }
            if (is_pinned(max_column, row)) {
                edges.right.push_back(row);
            }
        }

        triangulate_quad(indices, samples_x, quad, edges);
    }

    return indices;
}
