#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "load_model.hxx"
#include "noise.hxx"
#include "primitive_meshes.hxx"

// Parameters for a single rectangular terrain patch built from layered
// (fbm) simplex noise. Change `seed` for a different terrain with the same
// overall shape; the rest controls scale and roughness.
struct TerrainParams {
    std::uint32_t samples_x = 129; // vertex columns (>= 2)
    std::uint32_t samples_z = 129; // vertex rows (>= 2)
    float world_width = 80.0F; // world-space extent along local X
    float world_depth = 80.0F; // world-space extent along local Z
    float amplitude = 2.0F; // metres of vertical relief fed into the noise
    float frequency = 0.05F; // noise-space units per world unit
    std::uint32_t octaves = 4;
    float lacunarity = 2.0F;
    float persistence = 0.5F;
    std::uint32_t seed = 1U;
    float uv_scale = 0.08F; // texture-space units per world unit

    // World-space centre of this patch's sample window. make_terrain_mesh
    // has always centred its window on (0,0); a streamed terrain chunk
    // slides this so neighbouring chunks sample one continuous noise field
    // instead of each restarting at their own local origin. Defaults to 0
    // so every existing caller is unaffected.
    float world_origin_x = 0.0F;
    float world_origin_z = 0.0F;

    // Fixed, world-global vertical bounds of the noise field, used in place
    // of this patch's own observed min/max when computing
    // TerrainMeshResult::mid_height (see there for why). Leaving both at 0
    // (min == max) is the "unset" sentinel and preserves the legacy
    // per-patch behaviour -- a real range always has height_range_max >
    // height_range_min. fbm() is normalized to roughly [-1,1]
    // (see noise.cxx), so +/- amplitude is a natural choice.
    float height_range_min = 0.0F;
    float height_range_max = 0.0F;
};

// Shares one SimplexNoise2D permutation table across many height samples.
// SimplexNoise2D::sample/fbm are const and touch no mutable state after
// construction, so a single instance is safe to read concurrently from any
// number of threads -- this is what lets terrain chunk generation avoid
// paying SimplexNoise2D's construction cost (a 256-element std::shuffle) on
// every sample the way sample_terrain_height() below does.
class TerrainField {
public:
    explicit TerrainField(TerrainParams params) : params_{params}, noise_{params.seed} {}

    [[nodiscard]] auto height(float world_x, float world_z) const noexcept -> float;

    [[nodiscard]] auto params() const noexcept -> TerrainParams const & { return params_; }

private:
    TerrainParams params_;
    SimplexNoise2D noise_;
};

// Height of the noise field at a world (x, z) position, in the same units
// TerrainMeshResult::heights uses -- i.e. *before* make_terrain_mesh's
// mesh-centering offset (see TerrainMeshResult::mid_height). Lets other
// systems (placing houses, grass, the player) sit on the generated surface
// without walking the whole grid or duplicating make_terrain_mesh. Builds a
// fresh TerrainField per call; prefer TerrainField directly when sampling
// more than a handful of points against the same params.
[[nodiscard]] auto sample_terrain_height(TerrainParams const &params, float world_x, float world_z) -> float;

struct TerrainMeshResult {
    PrimitiveMeshData mesh;

    // Row-major (index = row * samples_x + column) -- matches what
    // btHeightfieldTerrainShape expects directly, see
    // Components::HeightfieldShape / PhysicsWorld::add_body.
    std::shared_ptr<std::vector<float> const> heights;
    std::uint32_t samples_x = 2;
    std::uint32_t samples_z = 2;

    float min_height = 0.0F;
    float max_height = 0.0F;
    float cell_size_x = 1.0F;
    float cell_size_z = 1.0F;

    // Vertical offset already subtracted from every mesh vertex's Y, so the
    // mesh is centered the same way btHeightfieldTerrainShape centers its
    // local AABB: (min_height + max_height) / 2. Add this back onto the
    // entity Transform's Y (on top of the desired world "sea level") so the
    // rendered surface and the collision surface land on the same heights.
    float mid_height = 0.0F;
};

[[nodiscard]] auto make_terrain_mesh(TerrainParams const &params)
        -> std::expected<TerrainMeshResult, ModelLoadError>;
