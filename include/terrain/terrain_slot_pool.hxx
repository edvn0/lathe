#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "assets/geometry.hxx"
#include "assets/load_model.hxx"
#include "assets/material.hxx"
#include "assets/mesh_sink.hxx"
#include "assets/meshlet.hxx"
#include "assets/model.hxx"
#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_quadtree.hxx"

struct TerrainSlotPoolError {
    std::string message;
};

struct TerrainSlotHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainSlotPoolCreateInfo {
    std::uint8_t lod_levels = 5;
    std::uint32_t slots_per_lod = 48;

    MaterialHandle material{};
    TerrainLodSettings lod_settings{};

    // Must match the height range every chunk is generated with; sets each slot's fixed AABB.
    float height_range_min = -2.0F;
    float height_range_max = 2.0F;
};

// A fixed set of GPU mesh slots, `slots_per_lod` per LOD, created once and recycled. Every chunk has exactly
// terrain_chunk_vertex_count vertices, so streaming a chunk in rewrites an existing slot's vertex range
// instead of allocating.
//
// A slot's local AABB depends only on its LOD (fixed height range, fixed span), so it is set once.
//
// The index slice (terrain_chunk_indices()) is allocated once and shared by every slot.
class TerrainSlotPool {
public:
    TerrainSlotPool() = default;

    [[nodiscard]] static auto create(IMeshSink &mesh_sink, VkCommandBuffer command_buffer,
                                     TerrainSlotPoolCreateInfo const &create_info)
            -> std::expected<TerrainSlotPool, TerrainSlotPoolError>;

    // Reserves a free slot for `lod`, or nullopt if that LOD is exhausted. Callers should warn: running out
    // leaves a hole in the terrain.
    [[nodiscard]] auto acquire(std::uint8_t lod) -> std::optional<TerrainSlotHandle>;

    // Overwrites `handle`'s vertex range with `vertices` (exactly terrain_chunk_vertex_count entries).
    [[nodiscard]] auto write(IMeshSink &mesh_sink, VkCommandBuffer command_buffer, TerrainSlotHandle handle,
                             std::span<CompressedModelVertex const> vertices) -> bool;

    // Frees `handle` after frames_in_flight tick_retirement() calls, once the GPU is done reading it.
    auto release_deferred(TerrainSlotHandle handle) -> void;

    // Call exactly once per frame.
    auto tick_retirement() -> void;

    [[nodiscard]] auto mesh(TerrainSlotHandle handle) const -> MeshHandle;

    [[nodiscard]] auto resident_count(std::uint8_t lod) const -> std::uint32_t;
    [[nodiscard]] auto capacity_per_lod() const -> std::uint32_t { return slots_per_lod_; }

private:
    struct SlotRecord {
        MeshHandle mesh{};
        GeometrySlice vertex_bytes{};

        // Per-slot meshlet bounds; the topology is shared, but bounds depend on the heights.
        GeometrySlice meshlet_bytes{};
        std::uint8_t lod = 0;
    };

    struct RetiringSlot {
        std::uint32_t slot_index = 0;
        std::uint32_t frames_remaining = 0;
    };

    std::vector<SlotRecord> slots_{};
    std::vector<std::vector<std::uint32_t>> free_by_lod_{}; // indices into slots_, per LOD
    std::vector<RetiringSlot> retiring_{};
    std::uint32_t slots_per_lod_ = 0;

    MeshletTopology meshlet_topology_{};
};
