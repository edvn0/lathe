#include "assets/model_streamer.hxx"

#include <algorithm>
#include <chrono>
#include <format>

#include "core/logger.hxx"

auto ModelStreamer::request(IModelSink &sink, std::filesystem::path source_path, ModelHandle fallback,
                            std::string debug_name) -> ModelHandle {
    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(source_path, canonicalize_error);
    auto const &cache_key_path = canonicalize_error ? source_path : canonical_path;
    std::size_t const path_hash = std::filesystem::hash_value(cache_key_path);

    if (auto it = path_cache_.find(path_hash); it != path_cache_.end()) {
        debug("model_streamer: '{}' already loaded, reusing its model handle", debug_name);
        sink.retain_model(it->second);
        return it->second;
    }

    auto pending_handle = sink.create_pending_model(fallback);

    if (!pending_handle) {
        warn("model_streamer: could not reserve a slot for '{}', staying on its fallback model", debug_name);
        // Every returned handle carries a reference for the caller.
        sink.retain_model(fallback);
        return fallback;
    }

    // The streamer's own reference, dropped once the request installs or fails.
    sink.retain_model(*pending_handle);

    auto profile = std::make_shared<ModelLoadProfile>();
    auto future = load_model_cpu_async(std::move(source_path), sink.sampler_storage(), profile);

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
            .profile = std::move(profile),
            .requested_at = std::chrono::steady_clock::now(),
            .path_hash = path_hash,
    });

    return *pending_handle;
}

namespace {

    // Materials/primitives uploaded per request per frame, bounding one model's per-frame cost.
    constexpr std::uint32_t gpu_upload_items_per_frame = 8;

} // namespace

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
                    fail(sink, request, std::format("{} while loading", cpu_data.error().type));
                    return true;
                }

                request.finalization = start_primitive_finalization(std::move(*cpu_data));
            }

            if (!request.upload.has_value()) {
                auto finalized = step_primitive_finalization(*request.finalization);

                if (!finalized) {
                    fail(sink, request, std::format("{} while finalizing", finalized.error().type));
                    return true;
                }

                if (!finalized->has_value()) {
                    return false; // more tangent/LOD work for a later frame
                }

                request.upload = start_model_gpu_upload(std::move(**finalized), sink.image_storage(),
                                                        sink.texture_streamer());
            }

            auto stepped = step_model_gpu_upload(*request.upload, command_buffer, sink.geometry_arena(),
                                                 sink.image_storage(), sink.material_storage(),
                                                 gpu_upload_items_per_frame);

            if (!stepped) {
                fail(sink, request, std::format("{} while uploading", stepped.error().type));
                return true;
            }

            if (!stepped->has_value()) {
                return false; // more GPU-upload work for a later frame
            }

            auto installed = sink.install_model(request.handle, **stepped);

            if (!installed) {
                fail(sink, request, std::format("{} while installing", installed.error().type));
                return true;
            }

            debug("model_streamer: '{}' uploaded to GPU", request.debug_name);

            request.installed = true;
            path_cache_[request.path_hash] = request.handle;
            sink.register_model_name(request.handle, request.debug_name);

            // Last, since this destroys the model if every caller already dropped it.
            sink.release_model(request.handle);
        }

        // Textures stream independently and can finish long after install. Keep the request until they're all done
        // so the logged profile covers them.
        if (request.profile != nullptr &&
            request.profile->texture_count.load(std::memory_order_relaxed) <
                    request.profile->expected_texture_count.load(std::memory_order_relaxed)) {
            return false;
        }

        if (request.profile != nullptr) {
            auto const elapsed = std::chrono::steady_clock::now() - request.requested_at;

            request.profile->total_wall_ns.store(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(), std::memory_order_relaxed);

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
