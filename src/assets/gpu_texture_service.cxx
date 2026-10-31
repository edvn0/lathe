#include "assets/gpu_texture_service.hxx"

#include <algorithm>
#include <utility>

#include "core/logger.hxx"

namespace {

    std::mutex installed_mutex;
    std::shared_ptr<GpuTextureService> installed; // NOLINT: process-wide, guarded by installed_mutex.

    auto request_bytes(std::span<GpuTextureLevel const> levels) -> std::uint64_t {
        std::uint64_t bytes = 0;

        for (auto const &level: levels) {
            bytes += level.rgba.size();
        }

        return bytes;
    }

} // namespace

GpuTextureService::GpuTextureService(Backend backend, std::uint64_t max_batch_bytes) :
    backend_(std::move(backend)), max_batch_bytes_(max_batch_bytes), thread_([this] { run(); }) {}

GpuTextureService::~GpuTextureService() { shutdown(); }

auto GpuTextureService::with_encoder(GpuTextureEncoder encoder) -> std::shared_ptr<GpuTextureService> {
    auto shared = std::make_shared<GpuTextureEncoder>(std::move(encoder));

    return std::make_shared<GpuTextureService>(
            [shared](std::span<GpuTextureLevel const> levels, GpuBlockFormat format) -> Outcome {
                auto result = shared->encode_batch(levels, format);

                if (!result) {
                    return std::unexpected{std::move(result.error())};
                }

                return std::move(result->blocks);
            });
}

auto GpuTextureService::submit(std::span<GpuTextureLevel const> levels, GpuBlockFormat format) -> std::future<Outcome> {
    Request request{.levels = levels, .format = format};
    auto future = request.promise.get_future();

    {
        std::scoped_lock const lock{mutex_};

        if (stopping_ || failed_) {
            request.promise.set_value(std::unexpected{
                    std::string{stopping_ ? "the GPU encoder is shut down" : "the GPU encoder failed earlier"}});

            return future;
        }

        queue_.push_back(std::move(request));
    }

    wake_.notify_one();

    return future;
}

auto GpuTextureService::shutdown() -> void {
    {
        std::scoped_lock const lock{mutex_};
        stopping_ = true;
    }

    wake_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }

    // The thread has ended; nothing else touches the queue or the backend now.
    if (stats_.requests != 0) {
        info("GPU texture service: {} textures in {} batches (largest {}), {:.0f} ms busy, {:.0f} ms of waiting in the "
             "queue "
             "in all",
             stats_.requests, stats_.batches, stats_.largest_batch, stats_.busy_ms, stats_.queue_wait_ms);
    }

    for (auto &request: queue_) {
        request.promise.set_value(std::unexpected{std::string{"the GPU encoder is shut down"}});
    }

    queue_.clear();
    backend_ = nullptr;
}

auto GpuTextureService::stats() const -> Stats {
    std::scoped_lock const lock{mutex_};

    return stats_;
}

auto GpuTextureService::usable() const -> bool {
    std::scoped_lock const lock{mutex_};

    return !stopping_ && !failed_;
}

auto GpuTextureService::run() -> void {
    std::unique_lock lock{mutex_};

    for (;;) {
        wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });

        if (stopping_) {
            return;
        }

        // Everything that is waiting in the same format, up to the batch budget (always at least one request).
        std::vector<Request> batch;
        std::uint64_t bytes = 0;
        auto const format = queue_.front().format;

        while (!queue_.empty()) {
            auto const next = request_bytes(queue_.front().levels);

            if (!batch.empty() && (bytes + next > max_batch_bytes_ || queue_.front().format != format)) {
                break;
            }

            bytes += next;
            batch.push_back(std::move(queue_.front()));
            queue_.pop_front();
        }

        lock.unlock();

        auto const started = std::chrono::steady_clock::now();
        auto waited_ms = 0.0;

        for (auto const &request: batch) {
            waited_ms += std::chrono::duration<double, std::milli>(started - request.queued_at).count();
        }

        std::vector<GpuTextureLevel> levels;

        for (auto const &request: batch) {
            levels.insert(levels.end(), request.levels.begin(), request.levels.end());
        }

        auto outcome = backend_(levels, format);
        auto const busy_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();

        if (outcome && outcome->size() != levels.size()) {
            outcome = std::unexpected{std::string{"the GPU encoder returned the wrong number of levels"}};
        }

        lock.lock();

        ++stats_.batches;
        stats_.requests += batch.size();
        stats_.levels += levels.size();
        stats_.largest_batch = std::max<std::uint64_t>(stats_.largest_batch, batch.size());
        stats_.busy_ms += busy_ms;
        stats_.queue_wait_ms += waited_ms;

        if (!outcome) {
            failed_ = true;
            stats_.failed_requests += batch.size();
            warn("GPU texture encoding failed ({}); compressing textures on the CPU from now on", outcome.error());
        }

        // The futures are fulfilled outside the lock: their owners wake straight into more submits.
        lock.unlock();

        auto next_level = std::size_t{0};

        for (auto &request: batch) {
            if (!outcome) {
                request.promise.set_value(std::unexpected{outcome.error()});

                continue;
            }

            Blocks blocks;
            blocks.reserve(request.levels.size());

            for (auto index = std::size_t{0}; index < request.levels.size(); ++index) {
                blocks.push_back(std::move((*outcome)[next_level++]));
            }

            request.promise.set_value(std::move(blocks));
        }

        lock.lock();

        // A failed backend must not be asked again; fail whatever queued up while it ran.
        if (failed_) {
            for (auto &request: queue_) {
                request.promise.set_value(std::unexpected{std::string{"the GPU encoder failed earlier"}});
            }

            queue_.clear();
        }
    }
}

auto install_gpu_texture_service(std::shared_ptr<GpuTextureService> service) -> void {
    std::shared_ptr<GpuTextureService> previous;

    {
        std::scoped_lock const lock{installed_mutex};
        previous = std::exchange(installed, std::move(service));
    }

    // Outside the lock: shutting down joins the service's thread.
    if (previous != nullptr) {
        previous->shutdown();
    }
}

auto gpu_texture_service() -> std::shared_ptr<GpuTextureService> {
    std::scoped_lock const lock{installed_mutex};

    return installed;
}
