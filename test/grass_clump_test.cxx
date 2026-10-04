#include <doctest/doctest.h>

#include <glm/geometric.hpp>

#include <algorithm>
#include <cstddef>

#include "assets/primitive_meshes.hxx"
#include "assets/procedural_textures.hxx"
#include "gpu/image.hxx"

namespace {

    [[nodiscard]] auto triangle_count(std::vector<std::uint32_t> const &indices) -> std::size_t {
        return indices.size() / 3;
    }

    // Every triangle's winding normal agrees with its vertices' normals, so the face the shader sees as front is
    // the side the normals point to (forward flips them on back faces of double-sided materials).
    auto check_winding_matches_normals(PrimitiveMeshData const &mesh,
                                       std::vector<std::uint32_t> const &indices) -> void {
        for (std::size_t triangle = 0; triangle + 2 < indices.size(); triangle += 3) {
            auto const &a = mesh.vertices[indices[triangle]];
            auto const &b = mesh.vertices[indices[triangle + 1]];
            auto const &c = mesh.vertices[indices[triangle + 2]];

            auto const winding = glm::cross(b.position - a.position, c.position - a.position);
            REQUIRE(glm::length(winding) > 0.0F);

            auto const vertex_normal = a.normal + b.normal + c.normal;
            CHECK(glm::dot(winding, vertex_normal) > 0.0F);
        }
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("grass clump: hand-made LODs share one vertex buffer and get cheaper with distance") {
        auto const mesh = make_grass_clump_mesh();
        REQUIRE(mesh.has_value());

        REQUIRE(mesh->lod_indices[0].has_value());
        REQUIRE(mesh->lod_indices[1].has_value());
        CHECK_FALSE(mesh->lod_indices[2].has_value()); // LOD3 reuses the cards

        // 12 blades of 5 triangles, 6 of 3, 3 cards of 2. The old two-sided clump was 144.
        CHECK(triangle_count(mesh->indices) == 60);
        CHECK(triangle_count(*mesh->lod_indices[0]) == 18);
        CHECK(triangle_count(*mesh->lod_indices[1]) == 6);
        CHECK(grass_clump_card_lod == 2);

        auto const vertex_count = mesh->vertices.size();
        auto const in_range = [&](std::vector<std::uint32_t> const &indices) {
            return std::ranges::all_of(indices, [&](std::uint32_t index) { return index < vertex_count; });
        };
        CHECK(in_range(mesh->indices));
        CHECK(in_range(*mesh->lod_indices[0]));
        CHECK(in_range(*mesh->lod_indices[1]));
    }

    TEST_CASE("grass clump: every LOD is wound the way its normals face") {
        auto const mesh = make_grass_clump_mesh();
        REQUIRE(mesh.has_value());

        check_winding_matches_normals(*mesh, mesh->indices);
        check_winding_matches_normals(*mesh, *mesh->lod_indices[0]);
        check_winding_matches_normals(*mesh, *mesh->lod_indices[1]);
    }

    TEST_CASE("grass clump: planted at y = 0 and under a metre tall") {
        auto const mesh = make_grass_clump_mesh();
        REQUIRE(mesh.has_value());

        auto const [lowest, highest] =
                std::ranges::minmax(mesh->vertices, {}, [](ModelVertex const &vertex) { return vertex.position.y; });
        CHECK(lowest.position.y == doctest::Approx(0.0F));
        CHECK(highest.position.y <= 0.95F);
    }

    TEST_CASE("grass card texture: a full mip chain whose alpha coverage holds at every level") {
        constexpr float cutoff = 0.5F;
        auto const texture = make_grass_card_texture(128, cutoff);

        CHECK(texture.width == 128);
        CHECK(texture.height == 128);
        REQUIRE(texture.mip_levels == 8);
        CHECK(texture.pixels.size() == mip_chain_offset(128, 128, 4, 8));
        CHECK(texture.level_offset(1) == std::size_t{128} * 128 * 4);
        CHECK(texture.level_width(7) == 1);

        auto const base_coverage = texture.alpha_coverage(0, cutoff);
        CHECK(base_coverage > 0.2F);
        CHECK(base_coverage < 0.8F);

        // A plain box filter loses most of the blades by 8x8; these levels keep level 0's share.
        for (std::uint32_t level = 1; level < texture.mip_levels; ++level) {
            if (texture.level_width(level) < 8) {
                break;
            }
            CAPTURE(level);
            CHECK(texture.alpha_coverage(level, cutoff) == doctest::Approx(base_coverage).epsilon(0.15));
        }
    }

    TEST_CASE("grass card texture: sparse blade tips at the top, dense roots at the bottom") {
        auto const texture = make_grass_card_texture(64);

        auto const row_alpha = [&](std::uint32_t row) {
            auto total = 0.0F;
            for (std::uint32_t column = 0; column < texture.width; ++column) {
                auto const offset = (static_cast<std::size_t>(row) * texture.width + column) * 4U + 3U;
                total += static_cast<float>(std::to_integer<std::uint8_t>(texture.pixels[offset])) / 255.0F;
            }
            return total / static_cast<float>(texture.width);
        };

        CHECK(row_alpha(0) < 0.2F);
        CHECK(row_alpha(texture.height - 1) > 0.4F);
        CHECK(row_alpha(texture.height - 1) > 3.0F * row_alpha(texture.height / 8));
    }

    TEST_CASE("mip_chain_offset sums the levels before the one asked for") {
        CHECK(mip_chain_offset(4, 4, 4, 0) == 0);
        CHECK(mip_chain_offset(4, 4, 4, 1) == 64);
        CHECK(mip_chain_offset(4, 4, 4, 3) == 64 + 16 + 4);
        CHECK(mip_chain_offset(8, 2, 1, 3) == 16 + 4 + 2); // 8x2, 4x1, 2x1
    }
}
