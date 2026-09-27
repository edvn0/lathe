#pragma once

#include <glm/vec2.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

#include "terrain/terrain_chunk.hxx"

// glm::vec2 values here are world-space (x, z).

// One chunk: a `terrain_chunk_cells`-cell square at `lod` covering [x, x+1) x [z, z+1) scaled by
// terrain_chunk_span(lod). Corner-indexed, so a chunk's footprint is exactly the union of its 4 children's.
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
    float base_cell_size = 1.0F; // world units per cell at LOD0

    // A node splits once the camera is within split_factor * span(lod) of its footprint; split_hysteresis is a
    // fractional deadband around that threshold.
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

// World-space centre of `key`'s footprint, for TerrainChunkRequest::world_origin_x/z.
[[nodiscard]] auto terrain_chunk_centre(ChunkKey const &key, TerrainLodSettings const &settings) -> glm::vec2;

// The 4 children covering `key`'s footprint at lod-1. key.lod must be > 0.
[[nodiscard]] auto terrain_chunk_children(ChunkKey const &key) -> std::array<ChunkKey, 4>;

// The parent whose footprint contains `key`, at lod+1.
[[nodiscard]] auto terrain_chunk_parent(ChunkKey const &key) -> ChunkKey;

// Selects the chunks to keep resident: descends from the coarsest LOD, splitting nodes near the camera and
// culling ones beyond view_distance. Pure function, so it's unit-testable.
//
// `split_state` carries hysteresis between frames and is updated in place; pass the same object every frame.
//
// `out_desired` is refilled with every leaf, largest first.
auto select_chunks(glm::vec2 camera_xz, TerrainLodSettings const &settings, ChunkKeySet &split_state,
                   std::vector<ChunkKey> &out_desired) -> void;
