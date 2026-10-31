#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "assets/model_load_profile.hxx"
#include "assets/texture_cache.hxx"
#include "core/paths.hxx"
#include "core/error_context.hxx"
#include "gpu/compressed_texture.hxx"

enum class TexturePipelineErrorType : std::uint8_t {
    source_not_found,
    decode_failed,
    encode_failed,
    cache_io_failed,
    transcode_failed,
    // The process is shutting down; queued work gave up before it began.
    cancelled,
};

struct TexturePipelineError {
    TexturePipelineErrorType type = TexturePipelineErrorType::decode_failed;
    std::optional<ErrorCause> cause;
};

// Cache hits publish a low-resolution tail of the mip chain here before the full texture is read, so the streamer can
// show something while the large mips are still loading.
class TexturePreviewSlot {
public:
    auto publish(CompressedTexture texture) -> void {
        std::lock_guard const lock{mutex_};
        texture_ = std::move(texture);
    }

    [[nodiscard]]
    auto take() -> std::optional<CompressedTexture> {
        std::lock_guard const lock{mutex_};
        return std::exchange(texture_, std::nullopt);
    }

private:
    std::mutex mutex_;
    std::optional<CompressedTexture> texture_;
};

// Process-wide encoding knobs, set once at startup before any texture loads.
//
// RDO lambda: UASTC rate-distortion optimisation. 0 (the default) is off; larger values give smaller files and lower
// quality (about 0.25 to 10; 1 is a mild setting). It changes what is written, so entries are cached separately.
auto set_texture_rdo_lambda(float lambda) noexcept -> void;

// Largest side of a loaded texture, in texels: the top mips of anything larger are dropped as it loads (the cache keeps
// the full chain). 0 is no limit.
auto set_texture_max_extent(std::uint32_t extent) noexcept -> void;

[[nodiscard]]
auto load_compressed_texture(AssetPath const &source_path, TextureRole role,
                             TextureCache const &cache = TextureCache::global(),
                             std::shared_ptr<ModelLoadProfile> const &profile = nullptr,
                             std::shared_ptr<TexturePreviewSlot> const &preview = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

[[nodiscard]]
auto load_compressed_texture_from_memory(
        std::span<std::byte const> rgba_pixels, std::uint32_t width, std::uint32_t height, TextureRole role,
        std::string_view cache_key, TextureCache const &cache = TextureCache::global(),
        std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

[[nodiscard]]
auto load_compressed_texture_from_encoded_memory(
        std::span<std::byte const> encoded_bytes, TextureRole role, std::string_view cache_key,
        TextureCache const &cache = TextureCache::global(),
        std::shared_ptr<ModelLoadProfile> const &profile = nullptr,
        std::shared_ptr<TexturePreviewSlot> const &preview = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

template<>
struct std::formatter<TexturePipelineErrorType> : std::formatter<std::string_view> {
    constexpr auto format(TexturePipelineErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case TexturePipelineErrorType::source_not_found:
                    return "source_not_found";
                case TexturePipelineErrorType::decode_failed:
                    return "decode_failed";
                case TexturePipelineErrorType::encode_failed:
                    return "encode_failed";
                case TexturePipelineErrorType::cache_io_failed:
                    return "cache_io_failed";
                case TexturePipelineErrorType::transcode_failed:
                    return "transcode_failed";
                case TexturePipelineErrorType::cancelled:
                    return "cancelled";
            }

            return "unknown_texture_pipeline_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
