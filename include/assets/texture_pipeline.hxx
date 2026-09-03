#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/compressed_texture.hxx"
#include "assets/model_load_profile.hxx"
#include "core/error_context.hxx"

enum class TexturePipelineErrorType : std::uint8_t {
    source_not_found,
    decode_failed,
    encode_failed,
    cache_io_failed,
    transcode_failed,
};

struct TexturePipelineError {
    TexturePipelineErrorType type = TexturePipelineErrorType::decode_failed;
    std::optional<ErrorCause> cause;
};

// Resolves the default on-disk .ktx2 cache location: `$XDG_CACHE_HOME/ktx2`
// if set, otherwise `~/.cache/ktx2`. Shared across a user's checkouts/builds
// of this project rather than living under the (per-build-directory) cwd.
[[nodiscard]]
auto default_texture_cache_directory() -> std::filesystem::path;

// Loads `source_path` as a BC5/BC7 compressed, fully-mipped texture, using
// (and populating) an on-disk .ktx2 cache under `cache_directory`. Pure CPU
// work -- decoding, mip generation, Basis Universal UASTC encoding, and
// transcode to the target block format all happen here. Safe to call from
// any thread: touches only the filesystem and CPU memory, never a Vulkan
// handle. `profile`, when non-null, gets this call's share of the texture
// section of its timing breakdown added in (safe to do from multiple
// threads at once -- see ModelLoadProfile).
[[nodiscard]]
auto load_compressed_texture(std::filesystem::path const &source_path, TextureRole role,
                             std::filesystem::path const &cache_directory = default_texture_cache_directory(),
                             std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

// Same pipeline (mip-generate, UASTC-encode, disk-cache, BC5/BC7-transcode)
// for already-decoded pixels with no on-disk source (e.g. a glTF image
// embedded in a bufferView/data URI). `cache_key` stands in for the
// path+stat identity load_compressed_texture() above derives from a real
// file, and must be stable and unique per distinct source image.
[[nodiscard]]
auto load_compressed_texture_from_memory(std::span<std::byte const> rgba_pixels, std::uint32_t width,
                                         std::uint32_t height, TextureRole role, std::string_view cache_key,
                                         std::filesystem::path const &cache_directory = default_texture_cache_directory(),
                                         std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

// Same pipeline again, this time for an image that's still encoded (raw
// PNG/JPEG/etc. file bytes with no file on disk to read them from, e.g. a
// glTF image embedded in a bufferView or data URI) -- decodes `encoded_bytes`
// itself (via DecodedImage) before handing off to the same mip-generate /
// UASTC-encode / disk-cache / BC5-BC7-transcode pipeline. `cache_key` plays
// the same role as in load_compressed_texture_from_memory() above.
[[nodiscard]]
auto load_compressed_texture_from_encoded_memory(std::span<std::byte const> encoded_bytes, TextureRole role,
                                                  std::string_view cache_key,
                                                  std::filesystem::path const &cache_directory =
                                                          default_texture_cache_directory(),
                                                  std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
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
            }

            return "unknown_texture_pipeline_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
