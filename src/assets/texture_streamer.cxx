#include "assets/texture_streamer.hxx"

#include <chrono>
#include <format>
#include <system_error>
#include <utility>

#include "core/logger.hxx"
#include "core/thread_pool.hxx"

auto TextureStreamer::request(ImageStorage &images, std::filesystem::path source_path, TextureRole role,
                              ImageHandle fallback, std::string debug_name, std::shared_ptr<ModelLoadProfile> profile)
        -> ImageHandle {
    std::error_code canonicalize_error;
    auto const canonical_path = std::filesystem::weakly_canonical(source_path, canonicalize_error);
    auto const path_key = std::format("{}|{}", (canonicalize_error ? source_path : canonical_path).generic_string(),
                                      std::to_underlying(role));

    if (auto const it = path_requests_.find(path_key); it != path_requests_.end() && images.contains(it->second)) {
        return it->second;
    }

    auto pending_handle = images.create_pending_image(fallback);

    if (!pending_handle) {
        warn("texture_streamer: could not reserve a slot for '{}', staying on its fallback texture", debug_name);
        return fallback;
    }

    if (profile != nullptr) {
        profile->expected_texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto &pool = thread_pool();

    auto recorded_path = canonicalize_error ? source_path : canonical_path;

    auto future = pool.submit_task([path = std::move(source_path), role, profile = std::move(profile)]() {
        return load_compressed_texture(path, role, default_texture_cache_directory(), profile);
    });

    path_requests_.insert_or_assign(path_key, *pending_handle);
    sources_.insert_or_assign(source_key(*pending_handle), Source{.path = std::move(recorded_path), .role = role});

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto TextureStreamer::request_from_memory(ImageStorage &images, std::vector<std::byte> encoded_bytes, TextureRole role,
                                          std::string cache_key, ImageHandle fallback, std::string debug_name,
                                          std::shared_ptr<ModelLoadProfile> profile) -> ImageHandle {
    auto pending_handle = images.create_pending_image(fallback);

    if (!pending_handle) {
        warn("texture_streamer: could not reserve a slot for '{}', staying on its fallback texture", debug_name);
        return fallback;
    }

    if (profile != nullptr) {
        profile->expected_texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto &pool = thread_pool();

    sources_.insert_or_assign(source_key(*pending_handle), Source{.cache_key = cache_key, .role = role});

    auto future = pool.submit_task([encoded = std::move(encoded_bytes), role, cache_key = std::move(cache_key),
                                    profile = std::move(profile)]() {
        return load_compressed_texture_from_encoded_memory(encoded, role, cache_key, default_texture_cache_directory(),
                                                           profile);
    });

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto TextureStreamer::request_cooked(ImageStorage &images,
                                     std::function<std::expected<CompressedTexture, TexturePipelineError>()> loader,
                                     std::string cache_key, ImageHandle fallback, std::string debug_name,
                                     std::shared_ptr<ModelLoadProfile> profile) -> ImageHandle {
    auto path_key = std::format("cooked|{}", cache_key);

    if (auto const it = path_requests_.find(path_key); it != path_requests_.end() && images.contains(it->second)) {
        return it->second;
    }

    auto pending_handle = images.create_pending_image(fallback);

    if (!pending_handle) {
        warn("texture_streamer: could not reserve a slot for '{}', staying on its fallback texture", debug_name);
        return fallback;
    }

    if (profile != nullptr) {
        profile->expected_texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto future = thread_pool().submit_task([loader = std::move(loader), profile = std::move(profile)]() {
        auto texture = loader();

        if (profile != nullptr) {
            profile->texture_count.fetch_add(1, std::memory_order_relaxed);
        }

        return texture;
    });

    path_requests_.insert_or_assign(std::move(path_key), *pending_handle);
    sources_.insert_or_assign(source_key(*pending_handle), Source{.cache_key = std::move(cache_key)});

    pending_.push_back(PendingRequest{
            .handle = *pending_handle,
            .debug_name = std::move(debug_name),
            .future = std::move(future),
    });

    return *pending_handle;
}

auto TextureStreamer::source_of(ImageHandle handle) const noexcept -> Source const * {
    auto const it = sources_.find(source_key(handle));
    return it != sources_.end() ? &it->second : nullptr;
}

auto TextureStreamer::set_source(ImageHandle handle, Source source) -> void {
    sources_.insert_or_assign(source_key(handle), std::move(source));
}

auto TextureStreamer::process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index)
        -> void {
    ZoneScopedNC("ProcessReadyTextures", tracy::Color::Goldenrod);

    using namespace std::chrono_literals;

    if (frame_index >= retiring_staging_.size()) {
        retiring_staging_.resize(frame_index + 1);
    }

    // This slot's fence was waited on before the frame began, so last use's staging buffers are free.
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

auto TextureStreamer::flush(ImageStorage &images, VulkanContext &context) -> std::size_t {
    ZoneScopedNC("FlushTextures", tracy::Color::Goldenrod);

    if (pending_.empty()) {
        return 0;
    }

    struct Ready {
        ImageHandle handle;
        std::string debug_name;
        CompressedTexture texture;
    };

    std::vector<Ready> ready;
    ready.reserve(pending_.size());

    for (auto &request: pending_) {
        auto result = request.future.get();

        if (!result) {
            error("texture_streamer: '{}' failed to load ({}); staying on its fallback texture", request.debug_name,
                  result.error().type);
            continue;
        }

        ready.push_back(Ready{
                .handle = request.handle, .debug_name = std::move(request.debug_name), .texture = std::move(*result)});
    }

    pending_.clear();

    std::vector<Buffer> staging;
    staging.reserve(ready.size());

    context.one_time_submit([&](VkCommandBuffer command_buffer) {
        for (auto const &entry: ready) {
            auto uploaded = images.upgrade_pending_image(entry.handle, entry.texture, command_buffer);

            if (!uploaded) {
                error("texture_streamer: '{}' failed to upload ({}); staying on its fallback texture", entry.debug_name,
                      uploaded.error().type);
                continue;
            }

            staging.push_back(std::move(*uploaded));
        }
    });

    // one_time_submit() waited for the copies.
    for (auto &buffer: staging) {
        buffer.destroy();
    }

    images.release_completed_uploads();

    return staging.size();
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
