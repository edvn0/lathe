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
    TerrainParams params{};

    TerrainLodSettings lod_settings{};
    std::uint32_t slots_per_lod = 48;

    std::uint32_t max_generations_in_flight = 4;
    std::uint32_t max_uploads_per_frame = 4;
    std::uint32_t max_collider_updates_per_frame = 2;
    std::uint32_t max_evictions_per_frame = 8;

    std::uint32_t eviction_grace_frames = 6;

    MaterialHandle material{};

    float ground_y = 0.0F;
};

class TerrainWorld {
public:
    TerrainWorld() = default;

    [[nodiscard]] static auto create(IMeshSink &mesh_sink, VkCommandBuffer command_buffer,
                                     TerrainWorldCreateInfo const &create_info)
            -> std::expected<TerrainWorld, TerrainSlotPoolError>;

    auto update(glm::vec2 camera_xz) -> void;

    auto process_ready(IMeshSink &mesh_sink, VkCommandBuffer command_buffer, PhysicsWorld *physics) -> void;

    auto submit(IMeshSink &mesh_sink) const -> void;

    auto on_physics_world_changed(PhysicsWorld *physics) -> void;

    auto wait_all() -> void { streamer_.wait_all(); }

    [[nodiscard]] auto streaming_idle() const noexcept -> bool { return in_flight_.empty(); }

private:
    struct ResidentChunk {
        TerrainSlotHandle slot;
        glm::vec3 centre{0.0F};
        std::vector<float> heights;
        TerrainColliderHandle collider{};
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

    std::vector<TerrainColliderHandle> collider_free_list_;

    std::unordered_map<ChunkKey, ResidentChunk, ChunkKeyHash> resident_;
};
