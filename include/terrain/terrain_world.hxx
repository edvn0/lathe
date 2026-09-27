#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "assets/material.hxx"
#include "assets/mesh_sink.hxx"
#include "physics/physics_world.hxx"
#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_mesh.hxx"
#include "terrain/terrain_quadtree.hxx"
#include "terrain/terrain_slot_pool.hxx"
#include "terrain/terrain_streamer.hxx"

struct TerrainWorldCreateInfo {
    // height_range_min/max must be a real range (max > min) that every chunk agrees on. world_origin_x/z are
    // ignored; the quadtree supplies chunk origins.
    TerrainParams params{};

    TerrainLodSettings lod_settings{};
    std::uint32_t slots_per_lod = 48;

    std::uint32_t max_generations_in_flight = 4;
    std::uint32_t max_uploads_per_frame = 4;
    std::uint32_t max_collider_updates_per_frame = 2;
    std::uint32_t max_evictions_per_frame = 8;

    // Frames an undesired chunk stays resident, giving its replacement time to load.
    std::uint32_t eviction_grace_frames = 6;

    MaterialHandle material{};

    // World-space Y of the terrain's local origin.
    float ground_y = 0.0F;
};

// Streams camera-relative terrain: noise, quadtree residency, async chunk generation and a fixed GPU slot
// pool. Creates no entities; chunks are submitted directly and collide through PhysicsWorld's terrain
// colliders, so registry clones don't need to know about it.
class TerrainWorld {
public:
    TerrainWorld() = default;

    [[nodiscard]] static auto create(IMeshSink &mesh_sink, VkCommandBuffer command_buffer,
                                     TerrainWorldCreateInfo const &create_info)
            -> std::expected<TerrainWorld, TerrainSlotPoolError>;

    // Recomputes residency around `camera_xz` and starts generating missing chunks. Call once per frame, in the
    // editor too.
    auto update(glm::vec2 camera_xz) -> void;

    // Uploads finished chunks, retires slots and colliders, binds LOD0 colliders and evicts stale chunks. Record
    // into `command_buffer` before any draw that reads terrain. With a null `physics`, terrain has no collision.
    auto process_ready(IMeshSink &mesh_sink, VkCommandBuffer command_buffer, PhysicsWorld *physics) -> void;

    // Submits every resident chunk. Call once per frame.
    auto submit(IMeshSink &mesh_sink) const -> void;

    // Call whenever the PhysicsWorld is recreated or destroyed; colliders rebind over the following frames.
    auto on_physics_world_changed(PhysicsWorld *physics) -> void;

    // Blocks until chunk generation finishes. Call before destroying the Renderer.
    auto wait_all() -> void { streamer_.wait_all(); }

    // Everything the last update() asked for is resident.
    [[nodiscard]] auto streaming_idle() const noexcept -> bool { return in_flight_.empty(); }

private:
    struct ResidentChunk {
        TerrainSlotHandle slot;
        glm::vec3 centre{0.0F}; // world-space, Y includes ground_y + mid_height
        std::vector<float> heights; // LOD0 collider binding only
        TerrainColliderHandle collider{}; // LOD0 only, while a PhysicsWorld exists
        std::uint32_t frames_undesired = 0;
    };

    [[nodiscard]] auto chunk_request_for(ChunkKey const &key) const -> TerrainChunkRequest;

    auto request_missing() -> void;
    auto upload_ready(IMeshSink &mesh_sink, VkCommandBuffer command_buffer) -> void;
    auto update_colliders(PhysicsWorld *physics) -> void;
    auto evict(PhysicsWorld *physics) -> void;

    std::shared_ptr<TerrainField const> field_;
    TerrainWorldCreateInfo create_info_{};

    TerrainSlotPool slot_pool_;
    TerrainStreamer streamer_;

    glm::vec2 camera_xz_{0.0F};
    ChunkKeySet split_state_;
    std::vector<ChunkKey> desired_;
    ChunkKeySet desired_set_;
    ChunkKeySet in_flight_;

    // Unbound collider handles, filled in bulk by on_physics_world_changed() and refilled by evict().
    std::vector<TerrainColliderHandle> collider_free_list_;

    std::unordered_map<ChunkKey, ResidentChunk, ChunkKeyHash> resident_;
};
