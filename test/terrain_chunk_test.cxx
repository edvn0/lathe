#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_mesh.hxx"

#include <cmath>
#include <optional>

namespace {

    [[nodiscard]] auto default_field() -> TerrainField {
        TerrainParams params{};
        params.height_range_min = -2.0F;
        params.height_range_max = 2.0F;
        return TerrainField{params};
    }

    [[nodiscard]] auto decode_position(CompressedModelVertex const &vertex) -> glm::vec3 {
        return glm::vec3{glm::unpackHalf1x16(vertex.position_x), glm::unpackHalf1x16(vertex.position_y),
                         glm::unpackHalf1x16(vertex.position_z)};
    }

    // The skirt group an index belongs to and that group's outward normal.
    [[nodiscard]] auto skirt_outward_direction(std::uint32_t index) -> std::optional<glm::vec3> {
        if (index < terrain_chunk_interior_vertex_count) {
            return std::nullopt;
        }

        auto const offset = index - terrain_chunk_interior_vertex_count;

        if (offset < terrain_chunk_samples) {
            return glm::vec3{0.0F, 0.0F, -1.0F}; // min_row
        }
        if (offset < 2U * terrain_chunk_samples) {
            return glm::vec3{0.0F, 0.0F, 1.0F}; // max_row
        }
        if (offset < 3U * terrain_chunk_samples) {
            return glm::vec3{-1.0F, 0.0F, 0.0F}; // min_col
        }
        return glm::vec3{1.0F, 0.0F, 0.0F}; // max_col
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("make_terrain_chunk produces the fixed layout at every LOD") {
        auto const field = default_field();

        for (auto const cell_size: {1.0F, 2.0F, 4.0F, 8.0F, 16.0F}) {
            auto const chunk = make_terrain_chunk(field, TerrainChunkRequest{
                                                                 .world_origin_x = 128.0F,
                                                                 .world_origin_z = -256.0F,
                                                                 .cell_size = cell_size,
                                                         });

            CHECK(chunk.vertices.size() == terrain_chunk_vertex_count);
            CHECK(chunk.heights.size() == terrain_chunk_interior_vertex_count);
        }

        CHECK(terrain_chunk_indices().size() == terrain_chunk_index_count);
    }

    TEST_CASE("terrain_chunk_indices stays within bounds and has non-degenerate triangles") {
        auto const &indices = terrain_chunk_indices();

        for (auto const index: indices) {
            CHECK(index < terrain_chunk_vertex_count);
        }

        auto const field = default_field();
        auto const chunk = make_terrain_chunk(field, TerrainChunkRequest{.cell_size = 1.0F});

        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            auto const a = decode_position(chunk.vertices[indices[i]]);
            auto const b = decode_position(chunk.vertices[indices[i + 1]]);
            auto const c = decode_position(chunk.vertices[indices[i + 2]]);

            auto const area = glm::length(glm::cross(b - a, c - a));
            CHECK(area > 0.0F);
        }
    }

    TEST_CASE("skirt triangles wind outward") {
        auto const field = default_field();
        auto const chunk = make_terrain_chunk(field, TerrainChunkRequest{.cell_size = 1.0F});
        auto const &indices = terrain_chunk_indices();

        int checked = 0;

        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            auto const i0 = indices[i];
            auto const i1 = indices[i + 1];
            auto const i2 = indices[i + 2];

            // Skirt triangles use two interior and two same-group skirt corners, so any skirt index gives the group.
            auto outward = skirt_outward_direction(i0);
            if (!outward) {
                outward = skirt_outward_direction(i1);
            }
            if (!outward) {
                outward = skirt_outward_direction(i2);
            }
            if (!outward) {
                continue; // interior triangle
            }

            auto const a = decode_position(chunk.vertices[i0]);
            auto const b = decode_position(chunk.vertices[i1]);
            auto const c = decode_position(chunk.vertices[i2]);

            auto const face_normal = glm::cross(b - a, c - a);
            CHECK(glm::dot(face_normal, *outward) > 0.0F);
            ++checked;
        }

        CHECK(checked == static_cast<int>(terrain_chunk_skirt_index_count / 3));
    }

    TEST_CASE("shared edge heights and normals are bit-identical between adjacent same-LOD chunks") {
        auto const field = default_field();
        constexpr auto cell_size = 1.0F;
        constexpr auto span = static_cast<float>(terrain_chunk_cells) * cell_size;

        auto const chunk_a = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = 0.0F,
                                                                           .world_origin_z = 0.0F,
                                                                           .cell_size = cell_size});
        auto const chunk_b = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = span,
                                                                           .world_origin_z = 0.0F,
                                                                           .cell_size = cell_size});

        for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
            auto const a_index = terrain_chunk_interior_index(terrain_chunk_cells, row); // chunk_a's +X edge
            auto const b_index = terrain_chunk_interior_index(0, row); // chunk_b's -X edge

            CHECK(chunk_a.heights[a_index] == chunk_b.heights[b_index]);
            CHECK(chunk_a.vertices[a_index].normal_oct == chunk_b.vertices[b_index].normal_oct);
        }
    }

    TEST_CASE("cross-LOD vertices at the same world position agree exactly") {
        auto const field = default_field();

        auto const lod0 = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = 0.0F,
                                                                        .world_origin_z = 0.0F,
                                                                        .cell_size = 1.0F});
        auto const lod1 = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = 0.0F,
                                                                        .world_origin_z = 0.0F,
                                                                        .cell_size = 2.0F});

        // Both chunks are centred on (0,0), which every LOD's grid passes through, and height() depends only on world
        // position.
        auto const centre = terrain_chunk_interior_index(32, 32);
        CHECK(lod0.heights[centre] == lod1.heights[centre]);
        CHECK(lod0.heights[centre] == field.height(0.0F, 0.0F));

        // LOD0 column 16 and LOD1 column 24 are both at world x = -16.
        auto const lod0_point = terrain_chunk_interior_index(16, 32);
        auto const lod1_point = terrain_chunk_interior_index(24, 32);
        CHECK(lod0.heights[lod0_point] == lod1.heights[lod1_point]);
        CHECK(lod0.heights[lod0_point] == field.height(-16.0F, 0.0F));
    }

    TEST_CASE("mid_height is invariant across chunks with different observed min/max") {
        // Different observed extremes but the same fixed height range, so vertex Y at a shared reference height must
        // agree.
        auto const field = default_field();

        auto const near = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = 0.0F, .cell_size = 1.0F});
        auto const far = make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = 10000.0F, .cell_size = 1.0F});

        REQUIRE(near.min_height != far.min_height); // different noise neighbourhoods

        constexpr auto mid_height = (-2.0F + 2.0F) * 0.5F; // == 0 for this field's fixed range

        for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
            for (std::uint32_t column = 0; column < terrain_chunk_samples; ++column) {
                auto const index = terrain_chunk_interior_index(column, row);
                auto const expected_y = near.heights[index] - mid_height;
                CHECK(decode_position(near.vertices[index]).y == doctest::Approx(expected_y).epsilon(0.01));
            }
        }
    }

    TEST_CASE("UV wraps to a bounded value far from the origin and stays continuous across a chunk boundary") {
        auto const field = default_field();
        constexpr auto cell_size = 1.0F;
        constexpr auto span = static_cast<float>(terrain_chunk_cells) * cell_size;
        constexpr auto far_origin = 1'000'000.0F;

        auto const chunk_a =
                make_terrain_chunk(field, TerrainChunkRequest{.world_origin_x = far_origin, .cell_size = cell_size});
        auto const chunk_b = make_terrain_chunk(
                field, TerrainChunkRequest{.world_origin_x = far_origin + span, .cell_size = cell_size});

        auto const uv_scale = field.params().uv_scale;

        for (auto const &vertex: chunk_a.vertices) {
            auto const u = glm::unpackHalf1x16(vertex.texcoord_u);
            // UVs stay within the chunk's local span however far it is from the origin, keeping half-float precision.
            CHECK(std::fabs(u) <= span * uv_scale + 1.0F);
        }

        auto const frac = [](float value) { return value - std::floor(value); };

        for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
            auto const a_index = terrain_chunk_interior_index(terrain_chunk_cells, row);
            auto const b_index = terrain_chunk_interior_index(0, row);

            auto const u_a = glm::unpackHalf1x16(chunk_a.vertices[a_index].texcoord_u);
            auto const u_b = glm::unpackHalf1x16(chunk_b.vertices[b_index].texcoord_u);

            CHECK(frac(u_a) == doctest::Approx(frac(u_b)).epsilon(0.01));
        }
    }

    TEST_CASE("sample_terrain_height is deterministic and matches TerrainField::height directly") {
        // sample_terrain_height must match TerrainField and be deterministic.
        TerrainParams const params{};

        CHECK(sample_terrain_height(params, 0.0F, 0.0F) == sample_terrain_height(params, 0.0F, 0.0F));

        auto const a = sample_terrain_height(params, 12.5F, -7.25F);
        auto const b = TerrainField{params}.height(12.5F, -7.25F);
        CHECK(a == b);
    }
}
