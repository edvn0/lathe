#include "terrain/terrain_chunk.hxx"

#include "terrain/terrain_greedy_mesh.hxx"

#include <glm/glm.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <limits>
#include <span>

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

    auto emit_quad(std::vector<std::uint32_t> &indices, std::uint32_t v00, std::uint32_t v01, std::uint32_t v11,
                   std::uint32_t v10) -> void {
        indices.push_back(v00);
        indices.push_back(v01);
        indices.push_back(v11);

        indices.push_back(v00);
        indices.push_back(v11);
        indices.push_back(v10);
    }

    auto append_skirt_indices(std::vector<std::uint32_t> &indices) -> void {
        for (std::uint32_t column = 0; column < terrain_chunk_cells; ++column) {
            emit_quad(indices, terrain_chunk_interior_index(column, 0), terrain_chunk_interior_index(column + 1, 0),
                      skirt_min_row_index(column + 1), skirt_min_row_index(column));

            auto const row = terrain_chunk_cells;
            emit_quad(indices, terrain_chunk_interior_index(column + 1, row), terrain_chunk_interior_index(column, row),
                      skirt_max_row_index(column), skirt_max_row_index(column + 1));
        }

        for (std::uint32_t row = 0; row < terrain_chunk_cells; ++row) {
            emit_quad(indices, terrain_chunk_interior_index(0, row + 1), terrain_chunk_interior_index(0, row),
                      skirt_min_col_index(row), skirt_min_col_index(row + 1));

            auto const column = terrain_chunk_cells;
            emit_quad(indices, terrain_chunk_interior_index(column, row), terrain_chunk_interior_index(column, row + 1),
                      skirt_max_col_index(row + 1), skirt_max_col_index(row));
        }
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

        append_skirt_indices(indices);

        return indices;
    }

    [[nodiscard]] auto build_greedy_chunk_indices(std::span<float const> heights, float tolerance)
            -> std::vector<std::uint32_t> {
        auto const quads = greedy_merge_terrain_cells(heights, terrain_chunk_samples, terrain_chunk_samples, tolerance);
        auto indices = triangulate_terrain_quads(quads, terrain_chunk_samples, terrain_chunk_samples);

        indices.reserve(indices.size() + terrain_chunk_skirt_index_count);
        append_skirt_indices(indices);

        meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), terrain_chunk_vertex_count);

        return indices;
    }

}

auto terrain_chunk_indices() -> std::vector<std::uint32_t> const & {
    static auto const indices = [] {
        auto built = build_terrain_chunk_indices();

        meshopt_optimizeVertexCache(built.data(), built.data(), built.size(), terrain_chunk_vertex_count);

        return built;
    }();

    return indices;
}

auto make_terrain_chunk(TerrainField const &field, TerrainChunkRequest const &request) -> TerrainChunkResult {
    auto const &params = field.params();

    auto const cell_size = request.cell_size;
    auto const half_span = static_cast<float>(terrain_chunk_cells) / 2.0F * cell_size;

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

    auto const mid_height = (params.height_range_min + params.height_range_max) * 0.5F;

    auto const skirt_depth = params.height_range_max - params.height_range_min;

    auto const uv_origin_x = glm::fract(request.world_origin_x * params.uv_scale);
    auto const uv_origin_z = glm::fract(request.world_origin_z * params.uv_scale);

    std::vector<ModelVertex> vertices(terrain_chunk_vertex_count);

    for (std::uint32_t row = 0; row < terrain_chunk_samples; ++row) {
        for (std::uint32_t column = 0; column < terrain_chunk_samples; ++column) {
            auto const height = heights[terrain_chunk_interior_index(column, row)];

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
        vertices[skirt_max_col_index(row)] = make_skirt_vertex(terrain_chunk_interior_index(terrain_chunk_cells, row));
    }

    std::vector<CompressedModelVertex> compressed(terrain_chunk_vertex_count);
    std::ranges::transform(vertices, compressed.begin(),
                           [](ModelVertex const &vertex) { return compress_vertex(vertex); });

    auto indices = build_greedy_chunk_indices(heights, params.greedy_tolerance * cell_size);

    MeshletBuild meshlets{
            .topology = build_meshlet_topology(indices, terrain_chunk_vertex_count),
            .meshlets = {},
    };
    meshlets.meshlets = compute_meshlet_bounds(meshlets.topology, compressed);

    return TerrainChunkResult{
            .vertices = std::move(compressed),
            .indices = std::move(indices),
            .meshlets = std::move(meshlets),
            .heights = std::move(heights),
            .min_height = min_height,
            .max_height = max_height,
    };
}
