#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "assets/gpu_texture_encoder.hxx"

// Runs BC7 and BC5 compression on one thread that owns the GPU encoder, for the texture pipeline's many worker threads.
// A worker submits the levels of a texture and waits; whatever has queued up by the time the thread is free goes to the
// GPU as one batch. If the backend ever fails the service stops taking work, and every later request fails at once so
// the caller compresses on the CPU instead.
class GpuTextureService {
public:
    // Per input level, its 16-byte blocks, or why the batch failed.
    using Blocks = std::vector<std::vector<std::byte>>;
    using Outcome = std::expected<Blocks, std::string>;
    // Compresses all the levels of a batch, which share one format; called on the service's own thread, one batch at a
    // time.
    using Backend = std::function<Outcome(std::span<GpuTextureLevel const>, GpuBlockFormat)>;

    struct Stats {
        std::uint64_t requests = 0;
        std::uint64_t levels = 0;
        std::uint64_t batches = 0;
        std::uint64_t largest_batch = 0;
        std::uint64_t failed_requests = 0;
        // Time the backend spent on batches, and the time requests waited in the queue before one started.
        double busy_ms = 0.0;
        double queue_wait_ms = 0.0;
    };

    static constexpr std::uint64_t default_max_batch_bytes = 256ULL * 1024 * 1024;

    explicit GpuTextureService(Backend backend, std::uint64_t max_batch_bytes = default_max_batch_bytes);
    ~GpuTextureService();

    GpuTextureService(GpuTextureService const &) = delete;
    auto operator=(GpuTextureService const &) -> GpuTextureService & = delete;
    GpuTextureService(GpuTextureService &&) = delete;
    auto operator=(GpuTextureService &&) -> GpuTextureService & = delete;

    // A service over the GPU encoder.
    [[nodiscard]] static auto with_encoder(GpuTextureEncoder encoder) -> std::shared_ptr<GpuTextureService>;

    // Queues the levels. The spans, and what they point to, must stay valid until the future is ready.
    [[nodiscard]] auto submit(std::span<GpuTextureLevel const> levels, GpuBlockFormat format) -> std::future<Outcome>;

    // Stops the thread, releases the backend (and so the GPU encoder) and fails whatever is still queued. Safe to call
    // more than once; call it before the device is destroyed.
    auto shutdown() -> void;

    [[nodiscard]] auto stats() const -> Stats;

    // False once the backend failed or the service was shut down.
    [[nodiscard]] auto usable() const -> bool;

private:
    struct Request {
        std::span<GpuTextureLevel const> levels;
        GpuBlockFormat format = GpuBlockFormat::bc7;
        std::promise<Outcome> promise;
        std::chrono::steady_clock::time_point queued_at = std::chrono::steady_clock::now();
    };

    auto run() -> void;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Request> queue_;
    Backend backend_;
    std::uint64_t max_batch_bytes_;
    Stats stats_;
    bool stopping_ = false;
    bool failed_ = false;
    std::thread thread_;
};

// The service the texture pipeline compresses with; none means it compresses on the CPU. Installing replaces (and shuts
// down) the previous one, and passing null removes it.
auto install_gpu_texture_service(std::shared_ptr<GpuTextureService> service) -> void;

[[nodiscard]] auto gpu_texture_service() -> std::shared_ptr<GpuTextureService>;
