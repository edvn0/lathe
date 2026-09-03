#pragma once

#include <volk.h>

#include <expected>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include "gpu/buffer.hxx"
#include "gpu/image_storage.hxx"
#include "assets/texture_pipeline.hxx"

// Kicks off async decode+cache+compress work for a texture on
// Renderer::thread_pool(), handing back a handle that samples a fallback
// default image until the background work finishes and gets promoted to
// its real GPU image -- callers never block waiting for a texture to load.
class TextureStreamer {
public:
    // `profile`, when non-null, gets this texture's share of the texture
    // section of its timing breakdown added in once the background job
    // runs -- see ModelLoadProfile and load_compressed_texture().
    [[nodiscard]]
    auto request(ImageStorage &images, std::filesystem::path source_path, TextureRole role, ImageHandle fallback,
                std::string debug_name, std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    // Same as request(), for a texture with no file on disk to stream from
    // (e.g. a glTF image embedded in a bufferView/data URI): `encoded_bytes`
    // is the still-encoded (PNG/JPEG/etc.) image, decoded on the background
    // thread along with everything else. `cache_key` must be stable and
    // unique per distinct source image -- see load_compressed_texture_from_encoded_memory().
    [[nodiscard]]
    auto request_from_memory(ImageStorage &images, std::vector<std::byte> encoded_bytes, TextureRole role,
                             std::string cache_key, ImageHandle fallback, std::string debug_name,
                             std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    // Finalizes every request whose background CompressedTexture is ready:
    // records its GPU upload into `command_buffer` and swaps it into the
    // still-pending slot. Call once per frame, with that frame's rotating
    // frame_index (the same one passed to Renderer::prepare_frame) -- used
    // to know when a previous call's staging buffers are safe to free (by
    // the time a frame_index slot is reused, the caller has already waited
    // on its fence, per the existing double/triple-buffering discipline).
    auto process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

    // Blocks until every outstanding background job finishes, without
    // uploading anything. Call before destroying the ImageStorage this
    // streamer's requests point into.
    auto wait_all() -> void;

private:
    struct PendingRequest {
        ImageHandle handle;
        std::string debug_name;
        std::future<std::expected<CompressedTexture, TexturePipelineError>> future;
    };

    std::vector<PendingRequest> pending_;
    std::vector<std::vector<Buffer>> retiring_staging_; // indexed by frame_index
};
