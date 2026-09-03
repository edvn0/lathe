#pragma once

#include <BS_thread_pool.hpp>

#include <future>
#include <memory>
#include <vector>

#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_mesh.hxx"
#include "terrain/terrain_quadtree.hxx"
#include "core/thread_pool.hxx"

// Kicks off async chunk generation (make_terrain_chunk) on thread_pool(),
// mirroring TextureStreamer/ModelStreamer's
// submit-then-drain pattern (reserve nothing eagerly, poll futures with
// wait_for(0s) each frame, no mutex, no result queue -- see
// src/texture_streamer.cxx). There is no "pending" placeholder to render
// here, unlike TextureStreamer: a not-yet-resident chunk simply isn't
// submitted for drawing at all (see TerrainWorld).
class TerrainStreamer {
public:
    // Submits chunk generation for `key` against `field`. Returns false
    // (submitting nothing) if max_in_flight requests are already
    // outstanding -- the caller should retry the same key next frame.
    // Submitted at BS::pr::low: PhysicsWorld's ThreadPoolTaskScheduler uses
    // this same pool and blocks waiting on it every physics step (see
    // src/physics_world.cxx), so chunk generation must always yield to
    // queued physics work. Priority alone doesn't prevent a chunk task
    // already running from stalling a physics step -- see max_in_flight at
    // the call site (TerrainWorld) for the actual mitigation.
    [[nodiscard]] auto request(std::shared_ptr<TerrainField const> field, ChunkKey key, TerrainChunkRequest request,
                               std::size_t max_in_flight) -> bool {

        if (pending_.size() >= max_in_flight) {
            return false;
        }

        auto &pool = thread_pool();
        auto future = pool.submit_task(
                [field = std::move(field), request]() { return make_terrain_chunk(*field, request); }, BS::pr::low);

        pending_.push_back(PendingRequest{.key = key, .future = std::move(future)});
        return true;
    }

    [[nodiscard]] auto in_flight_count() const noexcept -> std::size_t { return pending_.size(); }

    // Drains every request whose CPU work is done, calling
    // `on_ready(ChunkKey, TerrainChunkResult&&)` for each. Main thread only.
    // Pure CPU hand-off -- no command buffer or GPU work here; that happens
    // separately through TerrainSlotPool::write, driven by TerrainWorld once
    // it decides a chunk is ready to occupy a slot.
    template<typename OnReady>
    auto process_ready(OnReady &&on_ready) -> void {
        using namespace std::chrono_literals;

        std::erase_if(pending_, [&](PendingRequest &request) {
            if (request.future.wait_for(0s) != std::future_status::ready) {
                return false;
            }

            on_ready(request.key, request.future.get());
            return true;
        });
    }

    // Blocks until every outstanding background job finishes, without
    // invoking any callback. Call before this TerrainStreamer's captured
    // TerrainField shared_ptrs could otherwise outlive their last user.
    auto wait_all() -> void {
        for (auto &request: pending_) {
            request.future.wait();
        }
        pending_.clear();
    }

private:
    struct PendingRequest {
        ChunkKey key;
        std::future<TerrainChunkResult> future;
    };

    std::vector<PendingRequest> pending_;
};
