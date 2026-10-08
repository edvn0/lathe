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
#include "core/paths.hxx"
#include "core/fly_string.hxx"
#include "gpu/buffer.hxx"
#include "gpu/context.hxx"
#include "gpu/image_storage.hxx"

class TextureStreamer {
public:
    [[nodiscard]]
    auto request(ImageStorage &images, AssetPath source_path, TextureRole role, ImageHandle fallback,
                 FlyString debug_name, std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    [[nodiscard]]
    auto request_from_memory(ImageStorage &images, std::vector<std::byte> encoded_bytes, TextureRole role,
                             std::string cache_key, ImageHandle fallback, FlyString debug_name,
                             std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    [[nodiscard]]
    auto request_cooked(ImageStorage &images,
                        std::function<std::expected<CompressedTexture, TexturePipelineError>()> loader,
                        std::string cache_key, ImageHandle fallback, FlyString debug_name,
                        std::shared_ptr<ModelLoadProfile> profile = nullptr) -> ImageHandle;

    struct Source {
        std::optional<AssetPath> path;
        std::string cache_key;
        TextureRole role = TextureRole::colour;
    };

    [[nodiscard]]
    auto source_of(ImageHandle handle) const noexcept -> Source const *;

    auto set_source(ImageHandle handle, Source source) -> void;

    auto process_ready(ImageStorage &images, VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

    auto flush(ImageStorage &images, VulkanContext &context) -> std::size_t;

    auto wait_all() -> void;

    [[nodiscard]] auto pending_count() const noexcept -> std::size_t { return pending_.size(); }

private:
    struct PendingRequest {
        ImageHandle handle;
        FlyString debug_name;
        std::future<std::expected<CompressedTexture, TexturePipelineError>> future;
    };

    std::vector<PendingRequest> pending_;
    [[nodiscard]] static auto source_key(ImageHandle handle) noexcept -> std::uint64_t {
        return (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;
    }

    std::unordered_map<std::string, ImageHandle> path_requests_;
    std::unordered_map<std::uint64_t, Source> sources_;
    std::vector<std::vector<Buffer>> retiring_staging_;
};
