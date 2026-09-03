#include "assets/texture_streamer.hxx"

#include <chrono>

#include "core/logger.hxx"
#include "core/thread_pool.hxx"

auto TextureStreamer::request(ImageStorage &images, std::filesystem::path source_path, TextureRole role,
                              ImageHandle fallback, std::string debug_name) -> ImageHandle {
    auto pending_handle = images.create_pending_image(fallback);

    if (!pending_handle) {
        warn("texture_streamer: could not reserve a slot for '{}', staying on its fallback texture", debug_name);
        return fallback;
    }

    auto &pool = thread_pool();

    auto future =
            pool.submit_task([path = std::move(source_path), role]() { return load_compressed_texture(path, role); });

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto TextureStreamer::request_from_memory(ImageStorage &images, std::vector<std::byte> encoded_bytes, TextureRole role,
                                          std::string cache_key, ImageHandle fallback, std::string debug_name)
        -> ImageHandle {
    auto pending_handle = images.create_pending_image(fallback);

    if (!pending_handle) {
        warn("texture_streamer: could not reserve a slot for '{}', staying on its fallback texture", debug_name);
        return fallback;
    }

    auto &pool = thread_pool();

    auto future = pool.submit_task([encoded = std::move(encoded_bytes), role, cache_key = std::move(cache_key)]() {
        return load_compressed_texture_from_encoded_memory(encoded, role, cache_key);
    });

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto TextureStreamer::process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index)
        -> void {
    ZoneScopedNC("ProcessReadyTextures", tracy::Color::Goldenrod);

    using namespace std::chrono_literals;

    if (frame_index >= retiring_staging_.size()) {
        retiring_staging_.resize(frame_index + 1);
    }

    // By the time this frame_index slot is reused, its fence from
    // frames_in_flight cycles ago has already been waited on (see
    // Swapchain's per-frame fence wait before handing back a frame), so
    // whatever was staged for it last time is now safe to free.
    for (auto &buffer: retiring_staging_[frame_index]) {
        buffer.destroy();
    }

    retiring_staging_[frame_index].clear();

    std::erase_if(pending_, [&](PendingRequest &request) {
        if (request.future.wait_for(0s) != std::future_status::ready) {
            return false;
        }

        auto result = request.future.get();

        if (!result) {
            error("texture_streamer: '{}' failed to load ({}); staying on its fallback texture", request.debug_name,
                  result.error().type);

            return true;
        }

        auto uploaded = images.upgrade_pending_image(request.handle, *result, command_buffer);

        if (!uploaded) {
            error("texture_streamer: '{}' failed to upload ({}); staying on its fallback texture", request.debug_name,
                  uploaded.error().type);

            return true;
        }

        debug("texture_streamer: '{}' uploaded to GPU", request.debug_name);

        retiring_staging_[frame_index].push_back(std::move(*uploaded));

        return true;
    });
}

auto TextureStreamer::wait_all() -> void {
    for (auto &request: pending_) {
        request.future.wait();
    }

    pending_.clear();

    for (auto &staging: retiring_staging_) {
        for (auto &buffer: staging) {
            buffer.destroy();
        }
    }

    retiring_staging_.clear();
}
