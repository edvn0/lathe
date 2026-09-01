#include "terrain_world.hxx"

#include "error_describe.hxx"
#include "logger.hxx"
#include "renderer.hxx"

#include <algorithm>

auto TerrainWorld::create(Renderer &renderer, VkCommandBuffer command_buffer,
                          TerrainWorldCreateInfo const &create_info)
        -> std::expected<TerrainWorld, TerrainSlotPoolError> {

    if (create_info.params.height_range_max <= create_info.params.height_range_min) {
        return std::unexpected(TerrainSlotPoolError{
                .message = "terrain_world: create_info.params.height_range_min/max must be a real range (max > min)",
        });
    }

    TerrainWorld world;
    world.create_info_ = create_info;
    world.field_ = std::make_shared<TerrainField>(create_info.params);

    TerrainSlotPoolCreateInfo const pool_info{
            .lod_levels = create_info.lod_settings.lod_levels,
            .slots_per_lod = create_info.slots_per_lod,
            .material = create_info.material,
            .lod_settings = create_info.lod_settings,
            .height_range_min = create_info.params.height_range_min,
            .height_range_max = create_info.params.height_range_max,
    };

    auto pool = TerrainSlotPool::create(renderer, command_buffer, pool_info);

    if (!pool) {
        return std::unexpected(pool.error());
    }

    world.slot_pool_ = std::move(*pool);

    return world;
}

auto TerrainWorld::chunk_request_for(ChunkKey const &key) const -> TerrainChunkRequest {
    auto const centre = terrain_chunk_centre(key, create_info_.lod_settings);

    return TerrainChunkRequest{
            .world_origin_x = centre.x,
            .world_origin_z = centre.y, // glm::vec2 here is (world_x, world_z) -- see terrain_quadtree.hxx
            .cell_size = terrain_cell_size(create_info_.lod_settings, key.lod),
    };
}

auto TerrainWorld::update(glm::vec2 camera_xz) -> void {
    camera_xz_ = camera_xz;
    select_chunks(camera_xz, create_info_.lod_settings, split_state_, desired_);

    desired_set_.clear();
    for (auto const &key: desired_) {
        desired_set_.insert(key);
    }

    request_missing();
}

auto TerrainWorld::request_missing() -> void {
    std::vector<ChunkKey> candidates;

    for (auto const &key: desired_) {
        if (!resident_.contains(key) && !in_flight_.contains(key)) {
            candidates.push_back(key);
        }
    }

    // Nearest-first, so under the in-flight cap close chunks win over far
    // ones -- matters most right after a teleport/scene load when many
    // chunks are missing at once.
    std::ranges::sort(candidates, {}, [&](ChunkKey const &key) {
        return glm::distance(terrain_chunk_centre(key, create_info_.lod_settings), camera_xz_);
    });

    for (auto const &key: candidates) {
        if (streamer_.in_flight_count() >= create_info_.max_generations_in_flight) {
            break;
        }

        if (streamer_.request(field_, key, chunk_request_for(key), create_info_.max_generations_in_flight)) {
            in_flight_.insert(key);
        }
    }
}

auto TerrainWorld::process_ready(Renderer &renderer, VkCommandBuffer command_buffer, PhysicsWorld *physics) -> void {
    slot_pool_.tick_retirement();
    upload_ready(renderer, command_buffer);
    update_colliders(physics);
    evict(physics);
}

auto TerrainWorld::upload_ready(Renderer &renderer, VkCommandBuffer command_buffer) -> void {
    std::uint32_t uploaded_this_frame = 0;
    auto const mid_height = (create_info_.params.height_range_min + create_info_.params.height_range_max) * 0.5F;

    streamer_.process_ready([&](ChunkKey const &key, TerrainChunkResult &&result) {
        in_flight_.erase(key);

        if (uploaded_this_frame >= create_info_.max_uploads_per_frame) {
            // Budget exhausted this frame -- the result is dropped, but
            // it's cheap to regenerate: since this key is neither resident
            // nor in_flight_ anymore, the next update() call re-requests
            // it if it's still desired. Simpler than a second queue.
            return;
        }

        auto slot = slot_pool_.acquire(key.lod);

        if (!slot) {
            warn("terrain_world: slot pool exhausted for lod {} -- chunk ({},{}) dropped, will retry", key.lod, key.x,
                key.z);
            return;
        }

        if (!slot_pool_.write(renderer, command_buffer, *slot, result.vertices)) {
            error("terrain_world: failed to upload chunk ({},{}) lod={}", key.x, key.z, key.lod);
            slot_pool_.release_deferred(*slot);
            return;
        }

        ++uploaded_this_frame;

        auto const centre = terrain_chunk_centre(key, create_info_.lod_settings);

        resident_[key] = ResidentChunk{
                .slot = *slot,
                .centre = glm::vec3{centre.x, create_info_.ground_y + mid_height, centre.y},
                .heights = key.lod == 0 ? std::move(result.heights) : std::vector<float>{},
        };
    });
}

auto TerrainWorld::update_colliders(PhysicsWorld *physics) -> void {
    if (physics == nullptr) {
        return;
    }

    std::uint32_t updates = 0;

    for (auto &[key, chunk]: resident_) {
        if (key.lod != 0 || chunk.collider.valid()) {
            continue;
        }

        if (updates >= create_info_.max_collider_updates_per_frame) {
            break;
        }

        if (collider_free_list_.empty()) {
            warn("terrain_world: collider pool exhausted -- LOD0 chunk ({},{}) has no collision", key.x, key.z);
            break;
        }

        chunk.collider = collider_free_list_.back();
        collider_free_list_.pop_back();

        physics->bind_terrain_collider(chunk.collider, chunk.centre, chunk.heights);
        ++updates;
    }
}

auto TerrainWorld::evict(PhysicsWorld *physics) -> void {
    std::uint32_t evictions = 0;

    for (auto it = resident_.begin(); it != resident_.end();) {
        auto &chunk = it->second;

        if (desired_set_.contains(it->first)) {
            chunk.frames_undesired = 0;
            ++it;
            continue;
        }

        ++chunk.frames_undesired;

        // Grace period rather than an atomic parent/child swap -- keeps an
        // undesired chunk resident for a few extra frames so its
        // replacement (parent on merge, children on split) has a chance to
        // load first. Trades a few frames of visible overlap for never
        // showing a hole; see the terrain streaming plan's "riskiest
        // parts" for why the full atomic swap was not implemented (fiddly
        // cross-frame bookkeeping, hard to unit test, and the plan itself
        // sanctions this exact fallback).
        if (chunk.frames_undesired < create_info_.eviction_grace_frames ||
            evictions >= create_info_.max_evictions_per_frame) {
            ++it;
            continue;
        }

        if (physics != nullptr && chunk.collider.valid()) {
            physics->unbind_terrain_collider(chunk.collider);
            collider_free_list_.push_back(chunk.collider);
        }

        slot_pool_.release_deferred(chunk.slot);
        it = resident_.erase(it);
        ++evictions;
    }
}

auto TerrainWorld::submit(Renderer &renderer) const -> void {
    for (auto const &[key, chunk]: resident_) {
        glm::mat4 transform{1.0F};
        transform[3] = glm::vec4{chunk.centre, 1.0F};

        auto submitted = renderer.submit_mesh(slot_pool_.mesh(chunk.slot), transform, create_info_.material);

        if (!submitted) {
            error("terrain_world: submit_mesh failed for chunk ({},{}) lod={}: {}", key.x, key.z, key.lod,
                 describe(submitted.error()));
        }
    }
}

auto TerrainWorld::on_physics_world_changed(PhysicsWorld *physics) -> void {
    for (auto &[key, chunk]: resident_) {
        chunk.collider = TerrainColliderHandle{};
    }

    collider_free_list_.clear();

    if (physics == nullptr) {
        return;
    }

    // Bounded by slots_per_lod, so this is never the limiting factor:
    // LOD0's own GPU slot count already caps how many LOD0 chunks can be
    // resident at once. Reserved once per PhysicsWorld instance, never
    // grown per chunk -- see collider_free_list_'s doc comment.
    TerrainColliderDesc const desc{
            .samples_x = terrain_chunk_samples,
            .samples_z = terrain_chunk_samples,
            .cell_size_x = terrain_cell_size(create_info_.lod_settings, 0),
            .cell_size_z = terrain_cell_size(create_info_.lod_settings, 0),
            .min_height = create_info_.params.height_range_min,
            .max_height = create_info_.params.height_range_max,
    };

    physics->reserve_terrain_colliders(create_info_.slots_per_lod);

    for (std::uint32_t i = 0; i < create_info_.slots_per_lod; ++i) {
        collider_free_list_.push_back(physics->reserve_terrain_collider(desc));
    }
}
