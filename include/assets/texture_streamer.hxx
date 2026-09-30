#pragma once

#include <volk.h>

#include <expected>
#include <filesystem>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu/buffer.hxx"
#include "gpu/image_storage.hxx"
#include "assets/texture_pipeline.hxx"

// Loads textures on the thread pool and hands back a handle that samples a fallback until the real image is
// uploaded. Never blocks.
class TextureStreamer {
public:
    // `profile` gets this texture's share of the texture timings. Requesting a path again with the same role returns
    // the image already made for it, even one that failed to load and stays on its fallback.
    [[nodiscard]]
    auto request(ImageStorage &images, std::filesystem::path source_path, TextureRole role, ImageHandle fallback,
                std::string debug_name, std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    // request() for an image with no file, e.g. one embedded in a glTF. `encoded_bytes` is decoded on the
    // background thread. `cache_key` must be stable and unique per source image.
    [[nodiscard]]
    auto request_from_memory(ImageStorage &images, std::vector<std::byte> encoded_bytes, TextureRole role,
                             std::string cache_key, ImageHandle fallback, std::string debug_name,
                             std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    // Uploads every finished texture into its pending slot. Call once per frame with that frame's frame_index,
    // which decides when earlier staging buffers can be freed.
    auto process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

    // Blocks until background jobs finish, without uploading. Call before destroying the ImageStorage.
    auto wait_all() -> void;

    // Requests still loading in the background.
    [[nodiscard]] auto pending_count() const noexcept -> std::size_t { return pending_.size(); }

private:
    struct PendingRequest {
        ImageHandle handle;
        std::string debug_name;
        std::future<std::expected<CompressedTexture, TexturePipelineError>> future;
    };

    std::vector<PendingRequest> pending_;
    std::unordered_map<std::string, ImageHandle> path_requests_;
    std::vector<std::vector<Buffer>> retiring_staging_; // indexed by frame_index
};
