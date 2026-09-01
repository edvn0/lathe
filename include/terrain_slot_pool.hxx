#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "geometry.hxx"
#include "load_model.hxx"
#include "material.hxx"
#include "model.hxx"
#include "terrain_chunk.hxx"
#include "terrain_quadtree.hxx"

struct Renderer;

struct TerrainSlotPoolError {
    std::string message;
};

struct TerrainSlotHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainSlotPoolCreateInfo {
    std::uint8_t lod_levels = 5;
    std::uint32_t slots_per_lod = 48; // see the terrain streaming plan's residency-count estimate

    MaterialHandle material{};
    TerrainLodSettings lod_settings{};

    // Must match the TerrainParams::height_range_min/max every chunk in
    // this pool is generated with -- used once, at creation, to compute
    // each slot's fixed local-space AABB (see TerrainSlotPool's class
    // comment).
    float height_range_min = -2.0F;
    float height_range_max = 2.0F;
};

// A fixed set of GPU mesh slots, `slots_per_lod` per LOD level, created once
// and recycled in place for the lifetime of the pool -- never
// created/destroyed per chunk. This is what lets terrain streaming avoid
// GeometryArena's lack of a free-list (see docs/engine_review_followups.md
// and the terrain streaming plan): every chunk at every LOD has exactly
// terrain_chunk_vertex_count vertices, so a "new" chunk is just new vertex
// data written into an existing slot's GPU range via
// GeometryArena::rewrite_slice, never a new allocation.
//
// A slot's local-space AABB (Submesh::bounds_min/max) is a constant of its
// LOD -- height_range_min/max is fixed across every chunk (see
// TerrainParams), and every slot at a given LOD has the same span -- so it
// is set once at creation and never updated, even though the slot's vertex
// contents change every time a chunk streams in.
//
// All index/vertex slices for a slot are permanent GeometryArena
// allocations; only their contents are rewritten. The single index slice
// (terrain_chunk_indices()) is shared read-only across every slot at every
// LOD and allocated once regardless of slots_per_lod.
class TerrainSlotPool {
public:
    TerrainSlotPool() = default;

    [[nodiscard]] static auto create(Renderer &renderer, VkCommandBuffer command_buffer,
                                     TerrainSlotPoolCreateInfo const &create_info)
            -> std::expected<TerrainSlotPool, TerrainSlotPoolError>;

    // Reserves a free slot for `lod`. Returns nullopt if that LOD's pool is
    // exhausted -- callers must treat this loudly (a warning, not a silent
    // skip): the symptom of under-provisioning slots_per_lod is a hole in
    // the terrain, not a performance hiccup.
    [[nodiscard]] auto acquire(std::uint8_t lod) -> std::optional<TerrainSlotHandle>;

    // Uploads `vertices` (must be exactly terrain_chunk_vertex_count
    // entries) into `handle`'s existing GPU vertex range in place. Safe to
    // call repeatedly on the same handle; each call fully overwrites the
    // slot's previous contents.
    [[nodiscard]] auto write(Renderer &renderer, VkCommandBuffer command_buffer, TerrainSlotHandle handle,
                             std::span<CompressedModelVertex const> vertices) -> bool;

    // Marks `handle` free again after frames_in_flight further
    // tick_retirement() calls -- never handed back out by acquire() before
    // then, so any draw already submitted against it (up to the frame this
    // was called from) has finished being read by the GPU first. See the
    // terrain streaming plan for why frames_in_flight is exactly sufficient
    // (not just conservative) given where this is called from in the frame.
    auto release_deferred(TerrainSlotHandle handle) -> void;

    // Must be called exactly once per frame -- see TerrainWorld::process_ready.
    auto tick_retirement() -> void;

    [[nodiscard]] auto mesh(TerrainSlotHandle handle) const -> MeshHandle;

    [[nodiscard]] auto resident_count(std::uint8_t lod) const -> std::uint32_t;
    [[nodiscard]] auto capacity_per_lod() const -> std::uint32_t { return slots_per_lod_; }

private:
    struct SlotRecord {
        MeshHandle mesh{};
        GeometrySlice vertex_bytes{};
        std::uint8_t lod = 0;
    };

    struct RetiringSlot {
        std::uint32_t slot_index = 0;
        std::uint32_t frames_remaining = 0;
    };

    std::vector<SlotRecord> slots_{};
    std::vector<std::vector<std::uint32_t>> free_by_lod_{}; // indices into slots_, one free-list per LOD
    std::vector<RetiringSlot> retiring_{};
    std::uint32_t slots_per_lod_ = 0;
};
