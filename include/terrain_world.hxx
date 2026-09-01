#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <expected>
#include <memory>
#include <unordered_map>
#include <vector>

#include "material.hxx"
#include "physics_world.hxx"
#include "terrain_chunk.hxx"
#include "terrain_mesh.hxx"
#include "terrain_quadtree.hxx"
#include "terrain_slot_pool.hxx"
#include "terrain_streamer.hxx"

struct Renderer;

struct TerrainWorldCreateInfo {
    // amplitude/frequency/octaves/... plus a real height_range_min/max
    // (max > min) -- see TerrainParams and make_terrain_chunk's doc
    // comment for why every chunk must agree on the same fixed range.
    // world_origin_x/z are ignored here; per-chunk origins come from the
    // quadtree instead.
    TerrainParams params{};

    TerrainLodSettings lod_settings{};
    std::uint32_t slots_per_lod = 48; // see the terrain streaming plan's residency estimate

    std::uint32_t max_generations_in_flight = 4;
    std::uint32_t max_uploads_per_frame = 4;
    std::uint32_t max_collider_updates_per_frame = 2;
    std::uint32_t max_evictions_per_frame = 8;

    // How many frames an undesired chunk stays resident before eviction --
    // see TerrainWorld::evict()'s doc comment for why this exists instead
    // of an atomic parent/child swap.
    std::uint32_t eviction_grace_frames = 6;

    MaterialHandle material{};

    // World-space Y of the terrain's local origin (matches
    // Scene::physics_settings.ground_y in the existing static terrain).
    float ground_y = 0.0F;
};

// Owns everything needed to stream camera-relative terrain: the noise
// field, quadtree residency selection, async chunk generation
// (TerrainStreamer), and the fixed GPU slot pool (TerrainSlotPool) chunks
// are written into. Deliberately touches no entt::registry and creates no
// entities -- terrain chunks are submitted directly via
// Renderer::submit_mesh (see submit()) and collide via PhysicsWorld's
// registry-less terrain collider pool (see physics_world.hxx) -- so there
// is nothing here for Scene/Application::clone_registry to know about.
class TerrainWorld {
public:
    TerrainWorld() = default;

    [[nodiscard]] static auto create(Renderer &renderer, VkCommandBuffer command_buffer,
                                     TerrainWorldCreateInfo const &create_info)
            -> std::expected<TerrainWorld, TerrainSlotPoolError>;

    // Recomputes desired chunk residency against `camera_xz` and kicks off
    // generation for anything missing. Cheap and allocation-free after
    // warm-up; call once per frame, including in the editor (see
    // Application::update's is_playing early return -- this should be
    // called before that, not after, if streaming is wanted in-editor too).
    auto update(glm::vec2 camera_xz) -> void;

    // Drains finished chunk generations, uploads them into GPU slots, ticks
    // slot/collider retirement, updates LOD0 colliders, and evicts stale
    // chunks. Must be called with a currently-recording command buffer,
    // before any draw that could read terrain geometry is recorded into it
    // -- see the terrain streaming plan for why this ordering is
    // load-bearing (the write-before-read barrier direction), not
    // incidental. `physics` may be null (editor, or mid scene-stop) --
    // terrain then streams visually with no collision.
    auto process_ready(Renderer &renderer, VkCommandBuffer command_buffer, PhysicsWorld *physics) -> void;

    // Submits every currently-resident chunk for drawing. Call once per
    // frame from wherever submit_model/submit_mesh calls happen for the
    // rest of the scene (see src/main.cxx's submit_scene).
    auto submit(Renderer &renderer) const -> void;

    // PhysicsWorld is recreated wholesale on Scene::on_scene_start/stop
    // (see scene.cxx) -- every previously-bound TerrainColliderHandle
    // belonged to the instance that just went away. Call with the new
    // PhysicsWorld (or nullptr, e.g. on stop) whenever that happens; GPU
    // slots are untouched, and colliders rebind lazily under the normal
    // per-frame budget in process_ready().
    auto on_physics_world_changed(PhysicsWorld *physics) -> void;

    // Blocks until every outstanding chunk generation finishes. Call
    // before the Renderer (and its GeometryArena) this TerrainWorld's slot
    // pool references is destroyed.
    auto wait_all() -> void { streamer_.wait_all(); }

private:
    struct ResidentChunk {
        TerrainSlotHandle slot;
        glm::vec3 centre{0.0F}; // world-space, Y already includes ground_y + mid_height
        std::vector<float> heights; // only used for LOD0 collider binding
        TerrainColliderHandle collider{}; // valid only for LOD0 chunks while a PhysicsWorld exists
        std::uint32_t frames_undesired = 0;
    };

    [[nodiscard]] auto chunk_request_for(ChunkKey const &key) const -> TerrainChunkRequest;

    auto request_missing() -> void;
    auto upload_ready(Renderer &renderer, VkCommandBuffer command_buffer) -> void;
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

    // Collider handles this TerrainWorld currently doesn't have a resident
    // LOD0 chunk to bind -- populated in bulk (bounded by slots_per_lod, so
    // it's never the limiting factor: LOD0 GPU slot count already caps how
    // many LOD0 chunks can be resident at once) in
    // on_physics_world_changed, drawn down by update_colliders(), refilled
    // by evict(). Never grown one-off per chunk -- see PhysicsWorld's
    // terrain collider pool doc comments for why that matters.
    std::vector<TerrainColliderHandle> collider_free_list_;

    std::unordered_map<ChunkKey, ResidentChunk, ChunkKeyHash> resident_;
};
