#pragma once

#include <volk.h>

#include <expected>
#include <filesystem>
#include <functional>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include "assets/texture_pipeline.hxx"
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/image_storage.hxx"

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

    // request() for a texture that's already block-compressed, e.g. a TEXR chunk from an asset pack. `loader` runs on
    // the background thread and skips the texture pipeline. Requesting the same `cache_key` again returns the image
    // already made for it.
    [[nodiscard]]
    auto request_cooked(ImageStorage &images,
                        std::function<std::expected<CompressedTexture, TexturePipelineError>()> loader,
                        std::string cache_key, ImageHandle fallback, std::string debug_name,
                        std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    // Where a texture came from, for turning handles back into asset references when saving a scene.
    struct Source {
        std::filesystem::path path; // empty for cooked or embedded images
        std::string cache_key; // set for cooked and embedded images
        TextureRole role = TextureRole::colour;
    };

    // nullptr for handles this streamer didn't make (defaults, render targets) or whose slot was reused.
    [[nodiscard]]
    auto source_of(ImageHandle handle) const noexcept -> Source const *;

    // Records `source` for a cooked texture, so a scene loaded from a pack can be saved again with the original
    // file reference.
    auto set_source(ImageHandle handle, Source source) -> void;

    // Uploads every finished texture into its pending slot. Call once per frame with that frame's frame_index,
    // which decides when earlier staging buffers can be freed.
    auto process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

    // Blocks until every queued texture has loaded, then uploads them all in one submission and waits for it. For
    // loads that must be fully resident when they return (blocking scene loads, tools, tests). Render thread, outside
    // frame recording. Returns how many textures were installed.
    auto flush(ImageStorage &images, VulkanContext &context) -> std::size_t;

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
    [[nodiscard]] static auto source_key(ImageHandle handle) noexcept -> std::uint64_t {
        return (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;
    }

    std::unordered_map<std::string, ImageHandle> path_requests_;
    std::unordered_map<std::uint64_t, Source> sources_;
    std::vector<std::vector<Buffer>> retiring_staging_; // indexed by frame_index
};
