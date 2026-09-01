#include "terrain_chunk.hxx"

#include <glm/glm.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <limits>

namespace {

    [[nodiscard]] auto skirt_min_row_index(std::uint32_t column) -> std::uint32_t {
        return terrain_chunk_interior_vertex_count + column;
    }

    [[nodiscard]] auto skirt_max_row_index(std::uint32_t column) -> std::uint32_t {
        return terrain_chunk_interior_vertex_count + terrain_chunk_samples + column;
    }

    [[nodiscard]] auto skirt_min_col_index(std::uint32_t row) -> std::uint32_t {
        return terrain_chunk_interior_vertex_count + 2U * terrain_chunk_samples + row;
    }

    [[nodiscard]] auto skirt_max_col_index(std::uint32_t row) -> std::uint32_t {
        return terrain_chunk_interior_vertex_count + 3U * terrain_chunk_samples + row;
    }

    // Emits two triangles for the quad v00-v01-v11-v10 (in that loop order)
    // using the same winding convention make_terrain_mesh uses for its
    // interior quads (v00,v01,v11 then v00,v11,v10). Every skirt call site
    // below picks its own v00..v10 order so the resulting face normal
    // points outward, away from the chunk's interior -- verified by direct
    // cross-product calculation per edge (see terrain streaming plan and
    // the winding unit tests in test/terrain_chunk_test.cxx).
    auto emit_quad(std::vector<std::uint32_t> &indices, std::uint32_t v00, std::uint32_t v01, std::uint32_t v11,
                   std::uint32_t v10) -> void {
        indices.push_back(v00);
        indices.push_back(v01);
        indices.push_back(v11);

        indices.push_back(v00);
        indices.push_back(v11);
        indices.push_back(v10);
    }

    [[nodiscard]] auto build_terrain_chunk_indices() -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> indices;
        indices.reserve(terrain_chunk_index_count);

        for (std::uint32_t row = 0; row < terrain_chunk_cells; ++row) {
            for (std::uint32_t column = 0; column < terrain_chunk_cells; ++column) {
                emit_quad(indices, terrain_chunk_interior_index(column, row),
                         terrain_chunk_interior_index(column, row + 1),
                         terrain_chunk_interior_index(column + 1, row + 1),
                         terrain_chunk_interior_index(column + 1, row));
            }
        }

        for (std::uint32_t column = 0; column < terrain_chunk_cells; ++column) {
            // min_row edge (row 0): outward = -Z.
            emit_quad(indices, terrain_chunk_interior_index(column, 0), terrain_chunk_interior_index(column + 1, 0),
                     skirt_min_row_index(column + 1), skirt_min_row_index(column));

            // max_row edge (row 64): outward = +Z -- column order flipped
            // relative to min_row so the face normal flips too.
            auto const row = terrain_chunk_cells;
            emit_quad(indices, terrain_chunk_interior_index(column + 1, row), terrain_chunk_interior_index(column, row),
                     skirt_max_row_index(column), skirt_max_row_index(column + 1));
        }

        for (std::uint32_t row = 0; row < terrain_chunk_cells; ++row) {
            // min_col edge (column 0): outward = -X -- row order flipped
            // relative to max_col.
            emit_quad(indices, terrain_chunk_interior_index(0, row + 1), terrain_chunk_interior_index(0, row),
                     skirt_min_col_index(row), skirt_min_col_index(row + 1));

            // max_col edge (column 64): outward = +X.
            auto const column = terrain_chunk_cells;
            emit_quad(indices, terrain_chunk_interior_index(column, row), terrain_chunk_interior_index(column, row + 1),
                     skirt_max_col_index(row + 1), skirt_max_col_index(row));
        }

        return indices;
    }

} // namespace

auto terrain_chunk_indices() -> std::vector<std::uint32_t> const & {
    // Thread-safe lazy init (magic statics) -- TerrainStreamer generation
    // tasks may call this concurrently from multiple pool workers the first
    // time a chunk is generated.
    static auto const indices = [] {
        auto built = build_terrain_chunk_indices();

        // Reorders indices only (never vertices), so the row-major vertex
        // layout every chunk depends on for shared-slot recycling is
        // preserved -- see terrain_chunk.hxx.
        meshopt_optimizeVertexCache(built.data(), built.data(), built.size(), terrain_chunk_vertex_count);

        return built;
    }();

    return indices;
}

auto make_terrain_chunk(TerrainField const &field, TerrainChunkRequest const &request) -> TerrainChunkResult {
    auto const &params = field.params();

    auto const cell_size = request.cell_size;
    auto const half_span = static_cast<float>(terrain_chunk_cells / 2) * cell_size;

    auto const local_x = [&](std::uint32_t column) { return static_cast<float>(column) * cell_size - half_span; };
    auto const local_z = [&](std::uint32_t row) { return static_cast<float>(row) * cell_size - half_span; };

    auto const world_x = [&](std::uint32_t column) { return request.world_origin_x + local_x(column); };
    auto const world_z = [&](std::uint32_t row) { return request.world_origin_z + local_z(row); };

    std::vector<float> heights(terrain_chunk_interior_vertex_count);

    float min_height = std::numeric_limits<float>::max();
    float max_height = std::numeric_limits<float>::lowest();

    for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
        for (std::uint32_t column = 0; column < terrain_chunk_samples; ++column) {
            auto const height = field.height(world_x(column), world_z(row));
            heights[terrain_chunk_interior_index(column, row)] = height;

            min_height = std::min(min_height, height);
            max_height = std::max(max_height, height);
        }
    }

    // Fixed across every chunk/LOD (see TerrainParams::height_range_min/max)
    // so adjacent chunks never disagree about where local Y = 0 sits.
    auto const mid_height = (params.height_range_min + params.height_range_max) * 0.5F;

    // Provably exceeds any height difference the field can produce between
    // this chunk and a neighbour at a different LOD, so the skirt can never
    // fail to hide a crack regardless of the LOD delta at the boundary.
    auto const skirt_depth = params.height_range_max - params.height_range_min;

    auto const uv_origin_x = glm::fract(request.world_origin_x * params.uv_scale);
    auto const uv_origin_z = glm::fract(request.world_origin_z * params.uv_scale);

    std::vector<ModelVertex> vertices(terrain_chunk_vertex_count);

    for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
        for (std::uint32_t column = 0; column < terrain_chunk_samples; ++column) {
            auto const height = heights[terrain_chunk_interior_index(column, row)];

            // One-ring overlap: sampled directly against the field at true
            // world coordinates rather than a clamped lookup into
            // `heights`, so edge vertices get a real two-sided derivative
            // and two same-LOD neighbouring chunks -- which evaluate this
            // same field at the same world coordinates just past their
            // shared edge -- compute bit-identical normals there.
            auto const left = field.height(world_x(column) - cell_size, world_z(row));
            auto const right = field.height(world_x(column) + cell_size, world_z(row));
            auto const down = field.height(world_x(column), world_z(row) - cell_size);
            auto const up = field.height(world_x(column), world_z(row) + cell_size);

            auto const dhdx = (right - left) / (2.0F * cell_size);
            auto const dhdz = (up - down) / (2.0F * cell_size);

            auto const normal = glm::normalize(glm::vec3{-dhdx, 1.0F, -dhdz});

            vertices[terrain_chunk_interior_index(column, row)] = ModelVertex{
                    .position = glm::vec3{local_x(column), height - mid_height, local_z(row)},
                    .normal = normal,
                    // u along +X, v along +Z; the shader computes
                    // bitangent = cross(normal, tangent) * w, and
                    // cross(+Y, +X) = -Z, so w = -1 gives a bitangent along
                    // +Z. Analytic -- generate_tangents() is not run here
                    // (see make_terrain_chunk's doc comment).
                    .tangent = glm::vec4{1.0F, 0.0F, 0.0F, -1.0F},
                    .texcoord = glm::vec2{uv_origin_x + local_x(column) * params.uv_scale,
                                          uv_origin_z + local_z(row) * params.uv_scale},
            };
        }
    }

    auto const make_skirt_vertex = [&](std::uint32_t interior_index) {
        auto vertex = vertices[interior_index];
        vertex.position.y -= skirt_depth;
        return vertex;
    };

    for (std::uint32_t column = 0; column < terrain_chunk_samples; ++column) {
        vertices[skirt_min_row_index(column)] = make_skirt_vertex(terrain_chunk_interior_index(column, 0));
        vertices[skirt_max_row_index(column)] =
                make_skirt_vertex(terrain_chunk_interior_index(column, terrain_chunk_cells));
    }

    for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
        vertices[skirt_min_col_index(row)] = make_skirt_vertex(terrain_chunk_interior_index(0, row));
        vertices[skirt_max_col_index(row)] =
                make_skirt_vertex(terrain_chunk_interior_index(terrain_chunk_cells, row));
    }

    std::vector<CompressedModelVertex> compressed(terrain_chunk_vertex_count);
    std::ranges::transform(vertices, compressed.begin(), [](ModelVertex const &vertex) { return compress_vertex(vertex); });

    return TerrainChunkResult{
            .vertices = std::move(compressed),
            .heights = std::move(heights),
            .min_height = min_height,
            .max_height = max_height,
    };
}
