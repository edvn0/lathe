#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "assets/load_model.hxx"
#include "terrain/noise.hxx"
#include "assets/primitive_meshes.hxx"

// A smooth Gaussian bump added on top of the noise field.
struct TerrainHill {
    float world_x = 0.0F;
    float world_z = 0.0F;
    float height = 10.0F; // peak height in metres
    float radius = 12.0F; // Gaussian sigma in metres; the visible footprint is roughly 2.5x this
};

// A rectangular terrain patch from fbm simplex noise.
struct TerrainParams {
    std::uint32_t samples_x = 129; // vertex columns, >= 2
    std::uint32_t samples_z = 129; // vertex rows, >= 2
    float world_width = 80.0F; // world-space extent along local X
    float world_depth = 80.0F; // world-space extent along local Z
    float amplitude = 2.0F; // vertical relief in metres
    float frequency = 0.05F; // noise-space units per world unit
    std::uint32_t octaves = 4;
    float lacunarity = 2.0F;
    float persistence = 0.5F;
    std::uint32_t seed = 1U;
    float uv_scale = 0.08F; // texture-space units per world unit
    std::vector<TerrainHill> hills; // added to the noise; height_range_max must cover the tallest peak

    // World-space centre of the sample window, so streamed chunks sample one continuous field.
    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    // Fixed vertical bounds used for mid_height instead of the patch's observed min/max. min == max means unset
    // (per-patch bounds). fbm() is roughly [-1,1], so +/- amplitude works, plus the tallest hill.
    float height_range_min = 0.0F;
    float height_range_max = 0.0F;
};

// One permutation table shared across samples. Reads are const, so it's safe to share between threads.
class TerrainField {
public:
    explicit TerrainField(TerrainParams const &params) : params_{params}, noise_{params.seed} {}

    [[nodiscard]] auto height(float world_x, float world_z) const noexcept -> float;

    [[nodiscard]] auto params() const noexcept -> TerrainParams const & { return params_; }

private:
    TerrainParams params_;
    SimplexNoise2D noise_;
};

// Noise height at world (x, z), before mid_height centering. Builds a TerrainField per call; use one directly
// for more than a few samples.
[[nodiscard]] auto sample_terrain_height(TerrainParams const &params, float world_x, float world_z) -> float;

struct TerrainMeshResult {
    PrimitiveMeshData mesh;

    // Row-major (index = row * samples_x + column), as btHeightfieldTerrainShape expects.
    std::shared_ptr<std::vector<float> const> heights;
    std::uint32_t samples_x = 2;
    std::uint32_t samples_z = 2;

    float min_height = 0.0F;
    float max_height = 0.0F;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    // Subtracted from every vertex Y, matching how btHeightfieldTerrainShape centres its AABB. Add it to the
    // entity's Y so the rendered and collision surfaces line up.
    float mid_height = 0.0F;
};

[[nodiscard]] auto make_terrain_mesh(TerrainParams const &params)
        -> std::expected<TerrainMeshResult, ModelLoadError>;
