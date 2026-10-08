#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/primitive_meshes.hxx"
#include "terrain/noise.hxx"

struct TerrainHill {
    float world_x = 0.0F;
    float world_z = 0.0F;
    float height = 10.0F;
    float radius = 12.0F;
};

struct TerrainParams {
    std::uint32_t samples_x = 129;
    std::uint32_t samples_z = 129;
    float world_width = 80.0F;
    float world_depth = 80.0F;
    float amplitude = 2.0F;
    float frequency = 0.05F;
    std::uint32_t octaves = 4;
    float lacunarity = 2.0F;
    float persistence = 0.5F;
    std::uint32_t seed = 1U;
    float uv_scale = 0.08F;
    std::vector<TerrainHill> hills;

    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    float height_range_min = 0.0F;
    float height_range_max = 0.0F;

    float greedy_tolerance = 0.1F;
};

class TerrainField {
public:
    explicit TerrainField(TerrainParams const &params) : params_{params}, noise_{params.seed} {}

    [[nodiscard]] auto height(float world_x, float world_z) const noexcept -> float;

    [[nodiscard]] auto params() const noexcept -> TerrainParams const & { return params_; }

private:
    TerrainParams params_;
    SimplexNoise2D noise_;
};

[[nodiscard]] auto sample_terrain_height(TerrainParams const &params, float world_x, float world_z) -> float;

struct TerrainMeshResult {
    PrimitiveMeshData mesh;

    std::shared_ptr<std::vector<float> const> heights;
    std::uint32_t samples_x = 2;
    std::uint32_t samples_z = 2;

    float min_height = 0.0F;
    float max_height = 0.0F;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    float mid_height = 0.0F;
};

[[nodiscard]] auto make_terrain_mesh(TerrainParams const &params) -> std::expected<TerrainMeshResult, ModelLoadError>;
