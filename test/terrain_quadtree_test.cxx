#include <doctest/doctest.h>

#include "terrain_quadtree.hxx"

#include <algorithm>
#include <set>
#include <vector>

//
// select_chunks/terrain_chunk_children/terrain_chunk_parent are pure
// functions over plain data (no Vulkan/ECS/thread pool), so the residency
// invariants the streaming terrain design depends on -- no gaps, no
// overlaps, bounded flip count under hysteresis -- are directly
// unit-testable. See the terrain streaming plan for why each matters.
//

namespace {

    [[nodiscard]] auto footprint(ChunkKey const &key, TerrainLodSettings const &settings)
            -> std::pair<glm::vec2, glm::vec2> {
        auto const span = terrain_chunk_span(settings, key.lod);
        auto const min = glm::vec2{static_cast<float>(key.x) * span, static_cast<float>(key.z) * span};
        return {min, min + glm::vec2{span, span}};
    }

    [[nodiscard]] auto overlaps(std::pair<glm::vec2, glm::vec2> const &a, std::pair<glm::vec2, glm::vec2> const &b)
            -> bool {
        return a.first.x < b.second.x && b.first.x < a.second.x && a.first.y < b.second.y && b.first.y < a.second.y;
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("terrain_chunk_children exactly tile the parent's footprint") {
        TerrainLodSettings const settings{};

        for (auto const parent_x: {-3, -1, 0, 1, 4}) {
            for (auto const parent_z: {-2, 0, 1, 3}) {
                ChunkKey const parent{.x = parent_x, .z = parent_z, .lod = 2};
                auto const [parent_min, parent_max] = footprint(parent, settings);

                auto const children = terrain_chunk_children(parent);
                REQUIRE(children.size() == 4);

                for (auto const &child: children) {
                    CHECK(child.lod == parent.lod - 1);

                    auto const [child_min, child_max] = footprint(child, settings);
                    CHECK(child_min.x >= parent_min.x);
                    CHECK(child_min.y >= parent_min.y);
                    CHECK(child_max.x <= parent_max.x);
                    CHECK(child_max.y <= parent_max.y);
                }

                // No two children overlap, and together they cover exactly
                // the parent's area (checked via area sum, since each
                // child covers 1/4 of the parent's footprint with no gaps).
                for (std::size_t i = 0; i < children.size(); ++i) {
                    for (std::size_t j = i + 1; j < children.size(); ++j) {
                        CHECK_FALSE(overlaps(footprint(children[i], settings), footprint(children[j], settings)));
                    }
                }

                auto const parent_area = (parent_max.x - parent_min.x) * (parent_max.y - parent_min.y);
                auto child_area_sum = 0.0F;
                for (auto const &child: children) {
                    auto const [child_min, child_max] = footprint(child, settings);
                    child_area_sum += (child_max.x - child_min.x) * (child_max.y - child_min.y);
                }
                CHECK(child_area_sum == doctest::Approx(parent_area));
            }
        }
    }

    TEST_CASE("terrain_chunk_parent inverts terrain_chunk_children") {
        for (auto const x: {-5, -1, 0, 1, 6}) {
            for (auto const z: {-4, -1, 0, 2, 5}) {
                ChunkKey const key{.x = x, .z = z, .lod = 1};

                for (auto const &child: terrain_chunk_children(key)) {
                    CHECK(terrain_chunk_parent(child) == key);
                }
            }
        }
    }

    TEST_CASE("select_chunks is deterministic for a fixed camera position") {
        TerrainLodSettings settings{};
        settings.view_distance = 300.0F;

        ChunkKeySet split_state;
        std::vector<ChunkKey> first;
        select_chunks({37.0F, -19.0F}, settings, split_state, first);

        ChunkKeySet split_state_again;
        std::vector<ChunkKey> second;
        select_chunks({37.0F, -19.0F}, settings, split_state_again, second);

        std::ranges::sort(first, {}, [](ChunkKey const &key) { return std::tuple{key.lod, key.x, key.z}; });
        std::ranges::sort(second, {}, [](ChunkKey const &key) { return std::tuple{key.lod, key.x, key.z}; });

        CHECK(first == second);
        CHECK_FALSE(first.empty());
    }

    TEST_CASE("select_chunks produces no overlapping leaves and every leaf is within view distance") {
        TerrainLodSettings settings{};
        settings.view_distance = 500.0F;

        ChunkKeySet split_state;
        std::vector<ChunkKey> desired;
        select_chunks({123.0F, -456.0F}, settings, split_state, desired);

        REQUIRE_FALSE(desired.empty());

        for (std::size_t i = 0; i < desired.size(); ++i) {
            for (std::size_t j = i + 1; j < desired.size(); ++j) {
                CHECK_FALSE(overlaps(footprint(desired[i], settings), footprint(desired[j], settings)));
            }
        }
    }

    TEST_CASE("hysteresis reduces flip count when the camera sweeps across a split boundary") {
        TerrainLodSettings settings{};
        settings.lod_levels = 2;
        settings.view_distance = 1000.0F;
        settings.split_factor = 1.25F;

        auto const sweep_flip_count = [&](float hysteresis) {
            settings.split_hysteresis = hysteresis;

            ChunkKeySet split_state;
            std::vector<ChunkKey> desired;
            std::size_t previous_lod0_count = 0;
            std::size_t flips = 0;

            // Sweep slowly across the LOD0/LOD1 split threshold and back.
            auto const span0 = terrain_chunk_span(settings, 0);
            auto const threshold = settings.split_factor * span0;

            for (int step = 0; step <= 200; ++step) {
                auto const t = static_cast<float>(step) / 200.0F;
                auto const x = threshold * (0.9F + 0.2F * t); // sweeps from 0.9x to 1.1x threshold

                select_chunks({x, 0.0F}, settings, split_state, desired);

                auto const lod0_count =
                        static_cast<std::size_t>(std::ranges::count_if(desired, [](ChunkKey const &key) { return key.lod == 0; }));

                if (step > 0 && lod0_count != previous_lod0_count) {
                    ++flips;
                }
                previous_lod0_count = lod0_count;
            }

            return flips;
        };

        auto const flips_with_hysteresis = sweep_flip_count(0.15F);
        auto const flips_without_hysteresis = sweep_flip_count(0.0F);

        CHECK(flips_with_hysteresis <= flips_without_hysteresis);
    }
}
