#pragma once

#include <glm/vec2.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

#include "terrain/terrain_chunk.hxx"

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
    std::uint8_t lod_levels = 5;
    float base_cell_size = 1.0F;

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

[[nodiscard]] auto terrain_chunk_centre(ChunkKey const &key, TerrainLodSettings const &settings) -> glm::vec2;

[[nodiscard]] auto terrain_chunk_children(ChunkKey const &key) -> std::array<ChunkKey, 4>;

[[nodiscard]] auto terrain_chunk_parent(ChunkKey const &key) -> ChunkKey;

auto select_chunks(glm::vec2 camera_xz, TerrainLodSettings const &settings, ChunkKeySet &split_state,
                   std::vector<ChunkKey> &out_desired) -> void;
