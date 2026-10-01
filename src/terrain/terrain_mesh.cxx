#include "terrain/terrain_mesh.hxx"

#include "terrain/noise.hxx"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

    auto noise_height(SimplexNoise2D const &noise, TerrainParams const &params, float world_x, float world_z) -> float {
        auto height = noise.fbm(world_x * params.frequency, world_z * params.frequency, params.octaves,
                                params.lacunarity, params.persistence) *
                      params.amplitude;

        for (auto const &hill: params.hills) {
            auto const dx = world_x - hill.world_x;
            auto const dz = world_z - hill.world_z;
            height += hill.height * std::exp(-(dx * dx + dz * dz) / (2.0F * hill.radius * hill.radius));
        }

        return height;
    }

    // False for the {0,0} "unset" sentinel.
    [[nodiscard]] auto has_fixed_height_range(TerrainParams const &params) -> bool {
        return params.height_range_max > params.height_range_min;
    }

} // namespace

auto TerrainField::height(float world_x, float world_z) const noexcept -> float {
    return noise_height(noise_, params_, world_x, world_z);
}

auto sample_terrain_height(TerrainParams const &params, float world_x, float world_z) -> float {
    return TerrainField{params}.height(world_x, world_z);
}

auto make_terrain_mesh(TerrainParams const &params) -> std::expected<TerrainMeshResult, ModelLoadError> {
    auto const samples_x = std::max(params.samples_x, 2U);
    auto const samples_z = std::max(params.samples_z, 2U);

    auto const cell_size_x = params.world_width / static_cast<float>(samples_x - 1);
    auto const cell_size_z = params.world_depth / static_cast<float>(samples_z - 1);

    TerrainField const field{params};

    auto heights = std::make_shared<std::vector<float>>(static_cast<std::size_t>(samples_x) * samples_z);

    auto const half_width = static_cast<float>(samples_x - 1) * cell_size_x * 0.5F;
    auto const half_depth = static_cast<float>(samples_z - 1) * cell_size_z * 0.5F;

    // World position of the local (0,0) sample, so offset patches sample one continuous field.
    auto const origin_x = params.world_origin_x - half_width;
    auto const origin_z = params.world_origin_z - half_depth;

    float min_height = std::numeric_limits<float>::max();
    float max_height = std::numeric_limits<float>::lowest();

    // Sample at the centred local coordinates, matching sample_terrain_height().
    for (std::uint32_t row = 0; row < samples_z; ++row) {
        for (std::uint32_t column = 0; column < samples_x; ++column) {
            auto const world_x = origin_x + static_cast<float>(column) * cell_size_x;
            auto const world_z = origin_z + static_cast<float>(row) * cell_size_z;

            auto const height = field.height(world_x, world_z);
            (*heights)[row * samples_x + column] = height;

            min_height = std::min(min_height, height);
            max_height = std::max(max_height, height);
        }
    }

    // A fixed range gives every chunk the same mid_height; otherwise use this patch's own extremes.
    auto const mid_height = has_fixed_height_range(params) ? (params.height_range_min + params.height_range_max) * 0.5F
                                                           : (min_height + max_height) * 0.5F;

    // Central differences against the field, one ring past the patch, so edge vertices get two-sided derivatives
    // and neighbouring chunks compute identical normals.
    auto const height_at = [&](int column, int row) {
        auto const world_x = origin_x + static_cast<float>(column) * cell_size_x;
        auto const world_z = origin_z + static_cast<float>(row) * cell_size_z;
        return field.height(world_x, world_z);
    };

    std::vector<ModelVertex> vertices(static_cast<std::size_t>(samples_x) * samples_z);

    for (std::uint32_t row = 0; row < samples_z; ++row) {
        for (std::uint32_t column = 0; column < samples_x; ++column) {
            auto const height = (*heights)[row * samples_x + column];

            auto const left = height_at(static_cast<int>(column) - 1, static_cast<int>(row));
            auto const right = height_at(static_cast<int>(column) + 1, static_cast<int>(row));
            auto const down = height_at(static_cast<int>(column), static_cast<int>(row) - 1);
            auto const up = height_at(static_cast<int>(column), static_cast<int>(row) + 1);

            auto const dhdx = (right - left) / (2.0F * cell_size_x);
            auto const dhdz = (up - down) / (2.0F * cell_size_z);

            auto const normal = glm::normalize(glm::vec3{-dhdx, 1.0F, -dhdz});

            auto const local_x = static_cast<float>(column) * cell_size_x - half_width;
            auto const local_z = static_cast<float>(row) * cell_size_z - half_depth;

            // Equivalent to `world_xz * uv_scale` under REPEAT addressing, but stays small far from the origin so it
            // survives half-float packing.
            auto const uv_origin_x = glm::fract(params.world_origin_x * params.uv_scale);
            auto const uv_origin_z = glm::fract(params.world_origin_z * params.uv_scale);

            vertices[row * samples_x + column] = ModelVertex{
                    .position = glm::vec3{local_x, height - mid_height, local_z},
                    .normal = normal,
                    // Placeholder; generate_tangents() overwrites it.
                    .tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
                    .texcoord =
                            glm::vec2{uv_origin_x + local_x * params.uv_scale, uv_origin_z + local_z * params.uv_scale},
            };
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(samples_x - 1) * (samples_z - 1) * 6);

    auto const index_of = [samples_x](std::uint32_t column, std::uint32_t row) { return row * samples_x + column; };

    for (std::uint32_t row = 0; row + 1 < samples_z; ++row) {
        for (std::uint32_t column = 0; column + 1 < samples_x; ++column) {
            auto const bottom_left = index_of(column, row);
            auto const bottom_right = index_of(column + 1, row);
            auto const top_left = index_of(column, row + 1);
            auto const top_right = index_of(column + 1, row + 1);

            indices.push_back(bottom_left);
            indices.push_back(top_left);
            indices.push_back(top_right);

            indices.push_back(bottom_left);
            indices.push_back(top_right);
            indices.push_back(bottom_right);
        }
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return TerrainMeshResult{
            .mesh = PrimitiveMeshData{.vertices = std::move(vertices), .indices = std::move(indices)},
            .heights = std::move(heights),
            .samples_x = samples_x,
            .samples_z = samples_z,
            .min_height = min_height,
            .max_height = max_height,
            .cell_size_x = cell_size_x,
            .cell_size_z = cell_size_z,
            .mid_height = mid_height,
    };
}
