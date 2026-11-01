#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>

#include "rendering/emitter_grid.hxx"

namespace {

    // The emitters in the cell a point falls in (what a hit would loop over).
    auto emitters_at(EmitterGrid const &grid, glm::vec3 const &point) -> std::vector<std::uint32_t> {
        auto const c = glm::ivec3{glm::floor((point - grid.origin) / grid.cell)};
        if (glm::any(glm::lessThan(c, glm::ivec3{0})) || glm::any(glm::greaterThanEqual(c, glm::ivec3{grid.dims}))) {
            return {};
        }
        auto const cell = static_cast<std::uint32_t>(c.x) + grid.dims.x * (static_cast<std::uint32_t>(c.y) + grid.dims.y * static_cast<std::uint32_t>(c.z));
        return {grid.indices.begin() + grid.first[cell], grid.indices.begin() + grid.first[cell] + grid.count[cell]};
    }

    auto random_bulbs(std::size_t n, float extent, std::uint32_t seed) -> std::vector<EmitterSphere> {
        std::mt19937 engine{seed};
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};
        std::vector<EmitterSphere> bulbs;
        for (std::size_t i = 0; i < n; ++i) {
            bulbs.push_back({{unit(engine) * extent, unit(engine) * 6.0F, unit(engine) * extent}, 0.05F + unit(engine) * 0.25F});
        }
        return bulbs;
    }

    // Every emitter that reaches a point is in the point's cell.
    auto lookup_is_complete(std::vector<EmitterSphere> const &bulbs, EmitterGrid const &grid, std::uint32_t seed) -> bool {
        std::mt19937 engine{seed};
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};
        for (auto probe = 0; probe < 400; ++probe) {
            auto const point = grid.origin + glm::vec3{unit(engine), unit(engine), unit(engine)} * glm::vec3{grid.dims} * grid.cell;
            auto const found = emitters_at(grid, point);
            for (auto e = std::uint32_t{0}; e < bulbs.size(); ++e) {
                if (glm::distance(point, bulbs[e].centre) < emitter_reach(bulbs[e].radius) &&
                    std::ranges::find(found, e) == found.end()) {
                    return false;
                }
            }
        }
        return true;
    }

}

TEST_SUITE("unit") {
    TEST_CASE("emitter grid: nothing to build for no emitters") {
        CHECK_FALSE(build_emitter_grid({}, 16384, 32768).valid);
    }

    TEST_CASE("emitter grid: a point's cell holds every emitter that reaches it") {
        for (auto const n: {1U, 40U, 1000U}) {
            auto const bulbs = random_bulbs(n, 160.0F, n);
            auto const grid = build_emitter_grid(bulbs, 16384, 4096 * 8);
            REQUIRE(grid.valid);
            CHECK(lookup_is_complete(bulbs, grid, 7));
        }
    }

    TEST_CASE("emitter grid: thousands of emitters in a small field still fit the buffer") {
        // Cells half a reach wide would put 27 cells to an emitter and overflow the buffer (the bug this guards).
        auto const bulbs = random_bulbs(4096, 160.0F, 3);
        auto const grid = build_emitter_grid(bulbs, 16384, 4096 * 8);
        REQUIRE(grid.valid);
        CHECK(grid.indices.size() <= 4096U * 8U);
        CHECK(grid.first.size() <= 16384U);
        CHECK(lookup_is_complete(bulbs, grid, 11));
    }

    TEST_CASE("emitter grid: no grid when the limits cannot be met") {
        auto const bulbs = random_bulbs(500, 160.0F, 5);
        CHECK_FALSE(build_emitter_grid(bulbs, 16384, 10).valid);
        CHECK_FALSE(build_emitter_grid(bulbs, 0, 4096 * 8).valid);
    }
}
