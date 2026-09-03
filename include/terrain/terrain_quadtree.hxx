#pragma once

#include <glm/vec2.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

#include "terrain/terrain_chunk.hxx"

// Every glm::vec2 in this file is a world-space (x, z) pair with x in
// .x and z in .y -- there is no vertical component here, chunks are
// selected purely by their XZ footprint.

// Identifies one terrain chunk: a `terrain_chunk_cells`-cell square at `lod`
// whose world-space footprint is [x, x+1) x [z, z+1), scaled by
// terrain_chunk_span(lod) -- i.e. corner-indexed, not centre-indexed, which
// is what makes a chunk's footprint exactly equal the union of its 4
// children's footprints at lod-1 (see terrain_chunk_children()).
struct ChunkKey {
    std::int32_t x = 0;
    std::int32_t z = 0;
    std::uint8_t lod = 0;

    friend auto operator==(ChunkKey const &, ChunkKey const &) -> bool = default;
};

struct ChunkKeyHash {
    [[nodiscard]] auto operator()(ChunkKey const &key) const noexcept -> std::size_t {
        auto const ux = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.x));
        auto const uz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.z));
        auto const ulod = static_cast<std::uint64_t>(key.lod);

        auto const packed = ux ^ (uz * 0x9E3779B97F4A7C15ULL) ^ (ulod * 0xD1B54A32D192ED03ULL);
        return std::hash<std::uint64_t>{}(packed);
    }
};

using ChunkKeySet = std::unordered_set<ChunkKey, ChunkKeyHash>;

struct TerrainLodSettings {
    std::uint8_t lod_levels = 5; // LOD0 (finest) .. lod_levels-1 (coarsest)
    float base_cell_size = 1.0F; // world units per grid cell at LOD0

    // A node splits into its 4 children once the camera is within
    // split_factor * span(lod) of its footprint, with split_hysteresis as a
    // fractional deadband around that threshold to avoid regenerating a
    // chunk every frame when the camera sits still near the boundary.
    float split_factor = 1.25F;
    float split_hysteresis = 0.15F;

    float view_distance = 2048.0F;
};

[[nodiscard]] constexpr auto terrain_cell_size(TerrainLodSettings const &settings, std::uint8_t lod) -> float {
    return settings.base_cell_size * static_cast<float>(1U << lod);
}

[[nodiscard]] constexpr auto terrain_chunk_span(TerrainLodSettings const &settings, std::uint8_t lod) -> float {
    return static_cast<float>(terrain_chunk_cells) * terrain_cell_size(settings, lod);
}

// World-space centre of `key`'s footprint -- what TerrainChunkRequest's
// world_origin_x/z should be set to for this key.
[[nodiscard]] auto terrain_chunk_centre(ChunkKey const &key, TerrainLodSettings const &settings) -> glm::vec2;

// The 4 children covering exactly the same world-space footprint as `key`,
// at lod-1. Undefined to call with key.lod == 0.
[[nodiscard]] auto terrain_chunk_children(ChunkKey const &key) -> std::array<ChunkKey, 4>;

// The parent whose footprint contains `key`, at lod+1.
[[nodiscard]] auto terrain_chunk_parent(ChunkKey const &key) -> ChunkKey;

// Recomputes which chunks should be resident this frame: a quadtree
// descent from the coarsest LOD down, splitting a node into its 4 children
// once the camera is close enough (see TerrainLodSettings), and culling
// nodes entirely outside view_distance. Pure function of its arguments --
// no Vulkan/ECS/allocation beyond `out_desired`/`split_state`'s own storage,
// so it's directly unit-testable and safe to call from a residency manager
// with no GPU or thread-pool dependency.
//
// `split_state` carries hysteresis across frames: it records which nodes
// were split as of the last call, and is updated in place to reflect this
// call's decisions. Pass the same object back in on the next frame. Safe to
// mutate in place (rather than double-buffered) because the descent visits
// each key at most once per call, so a key's entry is always read before
// it's written.
//
// `out_desired` is cleared and refilled with every leaf (a chunk that
// should be resident) in descending order of size. Reuse the same vector
// across frames to avoid reallocating.
auto select_chunks(glm::vec2 camera_xz, TerrainLodSettings const &settings, ChunkKeySet &split_state,
                   std::vector<ChunkKey> &out_desired) -> void;
