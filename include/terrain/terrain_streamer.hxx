#pragma once

#include <BS_thread_pool.hpp>

#include <future>
#include <memory>
#include <vector>

#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_mesh.hxx"
#include "terrain/terrain_quadtree.hxx"
#include "core/thread_pool.hxx"

// Generates chunks on thread_pool() and polls the futures each frame. Chunks that aren't ready simply aren't
// drawn; there is no placeholder.
class TerrainStreamer {
public:
    // Starts generating `key`. Returns false without submitting if max_in_flight requests are outstanding; retry
    // next frame. Runs at low priority because physics blocks on the same pool every step.
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

    // Calls `on_ready(ChunkKey, TerrainChunkResult&&)` for every finished request. Main thread only; no GPU work.
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

    // Blocks until background jobs finish, without calling back.
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
