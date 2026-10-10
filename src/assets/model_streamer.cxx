#include "assets/model_streamer.hxx"
#include "core/perf_events.hxx"

#include <algorithm>
#include <chrono>
#include <format>

#include "core/error_describe.hxx"
#include "core/logger.hxx"

auto ModelStreamer::reserve(IModelSink &sink, AssetPath const &source_path, ModelHandle fallback,
                            std::string_view debug_name) -> Reservation {
    std::size_t const path_hash = std::hash<std::string>{}(source_path.key());

    if (auto it = path_cache_.find(path_hash); it != path_cache_.end()) {
        debug("model_streamer: '{}' already loaded, reusing its model handle", debug_name);
        sink.retain_model(it->second);
        return Reservation{.handle = it->second, .path_hash = path_hash, .final = true};
    }

    auto pending_handle = sink.create_pending_model(fallback);

    if (!pending_handle) {
        warn("model_streamer: could not reserve a slot for '{}', staying on its fallback model", debug_name);
        sink.retain_model(fallback);
        return Reservation{.handle = fallback, .path_hash = path_hash, .final = true};
    }

    sink.retain_model(*pending_handle);

    return Reservation{.handle = *pending_handle, .path_hash = path_hash};
}

auto ModelStreamer::request(IModelSink &sink, AssetPath source_path, ModelHandle fallback,
                            FlyString debug_name) -> ModelHandle {
    return request(sink, std::move(source_path), fallback, std::move(debug_name), {});
}

auto ModelStreamer::request(IModelSink &sink, AssetPath source_path, ModelHandle fallback, FlyString debug_name,
                            std::function<void(ModelCpuData const &)> on_loaded) -> ModelHandle {
    auto const reservation = reserve(sink, source_path, fallback, debug_name.view());

    if (reservation.final) {
        return reservation.handle;
    }

    auto profile = std::make_shared<ModelLoadProfile>();
    auto future = load_model_cpu_async(source_path, sink.sampler_storage(), profile);

    if (on_loaded) {
        // A thread of its own rather than a pool worker, so waiting on the load cannot starve it.
        future = std::async(std::launch::async, [loading = std::move(future), on_loaded = std::move(on_loaded)]() mutable {
            auto result = loading.get();

            if (result) {
                on_loaded(*result);
            }

            return result;
        });
    }

    pending_.push_back(PendingRequest{
            .handle = reservation.handle,
            .debug_name = debug_name,
            .future = std::move(future),
            .profile = std::move(profile),
            .requested_at = std::chrono::steady_clock::now(),
            .path_hash = reservation.path_hash,
            .source_path = std::move(source_path),
    });

    return reservation.handle;
}

auto ModelStreamer::request_prepared(IModelSink &sink,
                                     std::future<std::expected<ModelCpuData, ModelLoadError>> cpu_data,
                                     AssetPath source_path, ModelHandle fallback,
                                     FlyString debug_name) -> ModelHandle {
    auto const reservation = reserve(sink, source_path, fallback, debug_name.view());

    if (reservation.final) {
        return reservation.handle;
    }

    pending_.push_back(PendingRequest{
            .handle = reservation.handle,
            .debug_name = debug_name,
            .future = std::move(cpu_data),
            .requested_at = std::chrono::steady_clock::now(),
            .path_hash = reservation.path_hash,
            .source_path = std::move(source_path),
            .prepared = true,
    });

    return reservation.handle;
}

namespace {

    constexpr std::uint32_t gpu_upload_items_per_frame = 8;

}

auto ModelStreamer::process_ready(IModelSink &sink, VkCommandBuffer command_buffer) -> void {
    ZoneScopedNC("ProcessReadyModels", tracy::Color::Goldenrod);

    using namespace std::chrono_literals;

    std::erase_if(pending_, [&](PendingRequest &request) {
        if (!request.installed) {
            if (!request.finalization.has_value() && !request.upload.has_value()) {
                if (request.future.wait_for(0s) != std::future_status::ready) {
                    return false;
                }

                auto cpu_data = request.future.get();

                if (!cpu_data) {
                    fail(sink, request, std::format("{} while loading", describe(cpu_data.error())));
                    return true;
                }

                if (request.prepared) {
                    request.upload =
                            start_model_gpu_upload(std::move(*cpu_data), sink.image_storage(), sink.texture_streamer());
                } else {
                    request.finalization = start_primitive_finalization(std::move(*cpu_data));
                }
            }

            if (!request.upload.has_value()) {
                auto finalized = step_primitive_finalization(*request.finalization);

                if (!finalized) {
                    fail(sink, request, std::format("{} while finalizing", describe(finalized.error())));
                    return true;
                }

                if (!finalized->has_value()) {
                    return false;
                }

                request.upload =
                        start_model_gpu_upload(std::move(**finalized), sink.image_storage(), sink.texture_streamer());
            }

            auto stepped =
                    step_model_gpu_upload(*request.upload, command_buffer, sink.geometry_arena(), sink.image_storage(),
                                          sink.material_storage(), gpu_upload_items_per_frame);

            if (!stepped) {
                fail(sink, request, std::format("{} while uploading", describe(stepped.error())));
                return true;
            }

            if (!stepped->has_value()) {
                return false;
            }

            auto installed = sink.install_model(request.handle, **stepped);

            if (!installed) {
                fail(sink, request, std::format("{} while installing", describe(installed.error())));
                return true;
            }

            debug("model_streamer: '{}' uploaded to GPU", request.debug_name);
            perf_events::record(PerfEvent::model_install);

            request.installed = true;
            path_cache_[request.path_hash] = request.handle;
            sink.register_model_name(request.handle, request.debug_name.view());
            sink.register_model_source(request.handle, request.source_path);

            sink.release_model(request.handle);
        }

        if (request.profile != nullptr &&
            request.profile->texture_count.load(std::memory_order_relaxed) <
                    request.profile->expected_texture_count.load(std::memory_order_relaxed)) {
            return false;
        }

        if (request.profile != nullptr) {
            auto const elapsed = std::chrono::steady_clock::now() - request.requested_at;

            request.profile->total_wall_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                                                 std::memory_order_relaxed);

            info("model_streamer: '{}' load profile:\n{}", request.debug_name,
                 format_model_load_profile(*request.profile));
        }

        return true;
    });
}

auto ModelStreamer::fail(IModelSink &sink, PendingRequest const &request, std::string reason) -> void {
    error("model_streamer: '{}' failed: {}; staying on its fallback model", request.debug_name, reason);

    failed_.push_back(FailedRequest{.handle = request.handle, .reason = std::move(reason)});
    sink.release_model(request.handle);
}

auto ModelStreamer::forget(ModelHandle handle) -> void {
    std::erase_if(path_cache_, [handle](auto const &entry) { return entry.second == handle; });
    std::erase_if(failed_, [handle](FailedRequest const &failed) { return failed.handle == handle; });
}

auto ModelStreamer::state(ModelHandle handle) const -> ModelRequestState {
    if (std::ranges::any_of(pending_, [handle](PendingRequest const &request) {
            return request.handle == handle && !request.installed;
        })) {
        return ModelRequestState::loading;
    }

    if (!failure_reason(handle).empty()) {
        return ModelRequestState::failed;
    }

    return ModelRequestState::none;
}

auto ModelStreamer::failure_reason(ModelHandle handle) const -> std::string_view {
    auto const it = std::ranges::find(failed_, handle, &FailedRequest::handle);
    return it != failed_.end() ? std::string_view{it->reason} : std::string_view{};
}

auto ModelStreamer::wait_all() -> void {
    for (auto &request: pending_) {
        if (!request.finalization.has_value() && !request.upload.has_value()) {
            request.future.wait();
        }
    }

    pending_.clear();
}
