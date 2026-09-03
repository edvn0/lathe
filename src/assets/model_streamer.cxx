#include "assets/model_streamer.hxx"

#include <chrono>

#include "core/logger.hxx"

auto ModelStreamer::request(IModelSink &sink, std::filesystem::path source_path, ModelHandle fallback,
                            std::string debug_name) -> ModelHandle {
    auto pending_handle = sink.create_pending_model(fallback);

    if (!pending_handle) {
        warn("model_streamer: could not reserve a slot for '{}', staying on its fallback model", debug_name);
        return fallback;
    }

    auto profile = std::make_shared<ModelLoadProfile>();
    auto future = load_model_cpu_async(std::move(source_path), sink.sampler_storage(), profile);

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
            .profile = std::move(profile),
            .requested_at = std::chrono::steady_clock::now(),
    });

    return *pending_handle;
}

namespace {

    // Materials/mesh primitives (see step_model_gpu_upload()) processed per
    // pending request per process_ready() call. Bounds how much CPU/GPU
    // upload work a single frame can be made to do by one streamed model --
    // Sponza's ~25 materials and dozens of mesh primitives finish over
    // several frames instead of all at once, keeping any one frame's cost
    // small regardless of how big the model is.
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
                    error("model_streamer: '{}' failed to load ({}); staying on its fallback model",
                          request.debug_name, cpu_data.error().type);

                    return true;
                }

                request.finalization = start_primitive_finalization(std::move(*cpu_data));
            }

            if (!request.upload.has_value()) {
                auto finalized = step_primitive_finalization(*request.finalization);

                if (!finalized) {
                    error("model_streamer: '{}' failed to finalize ({}); staying on its fallback model",
                          request.debug_name, finalized.error().type);

                    return true;
                }

                if (!finalized->has_value()) {
                    return false; // more tangent/LOD work left for a later frame
                }

                request.upload = start_model_gpu_upload(std::move(**finalized), sink.image_storage(),
                                                        sink.texture_streamer());
            }

            auto stepped = step_model_gpu_upload(*request.upload, command_buffer, sink.geometry_arena(),
                                                 sink.image_storage(), sink.material_storage(),
                                                 gpu_upload_items_per_frame);

            if (!stepped) {
                error("model_streamer: '{}' failed to finish loading ({}); staying on its fallback model",
                      request.debug_name, stepped.error().type);

                return true;
            }

            if (!stepped->has_value()) {
                return false; // more GPU-upload work left for a later frame
            }

            auto installed = sink.install_model(request.handle, **stepped);

            if (!installed) {
                error("model_streamer: '{}' failed to install ({}); staying on its fallback model",
                      request.debug_name, installed.error().type);

                return true;
            }

            debug("model_streamer: '{}' uploaded to GPU", request.debug_name);

            request.installed = true;
        }

        // Geometry/materials installing doesn't mean this model is fully
        // loaded -- its textures stream in independently on thread_pool()
        // workers (see TextureStreamer) and can still be running well after
        // this point (as seen in practice: a Sponza-sized load's texture
        // log lines keep appearing for many frames after its "uploaded to
        // GPU" line). Keep this request around, still profiled, until every
        // texture it kicked off has finished, so the profile logged below
        // covers the whole load instead of whichever few textures happened
        // to race ahead of the others.
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

auto ModelStreamer::wait_all() -> void {
    for (auto &request: pending_) {
        if (!request.finalization.has_value() && !request.upload.has_value()) {
            request.future.wait();
        }
    }

    pending_.clear();
}
