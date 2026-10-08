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

#include "assets/model_load_profile.hxx"
#include "core/paths.hxx"
#include "core/error_context.hxx"
#include "gpu/compressed_texture.hxx"

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

[[nodiscard]]
auto default_texture_cache_directory() -> std::filesystem::path;

[[nodiscard]]
auto load_compressed_texture(AssetPath const &source_path, TextureRole role,
                             std::filesystem::path const &cache_directory = default_texture_cache_directory(),
                             std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

[[nodiscard]]
auto load_compressed_texture_from_memory(
        std::span<std::byte const> rgba_pixels, std::uint32_t width, std::uint32_t height, TextureRole role,
        std::string_view cache_key, std::filesystem::path const &cache_directory = default_texture_cache_directory(),
        std::shared_ptr<ModelLoadProfile> const &profile = nullptr)
        -> std::expected<CompressedTexture, TexturePipelineError>;

[[nodiscard]]
auto load_compressed_texture_from_encoded_memory(
        std::span<std::byte const> encoded_bytes, TextureRole role, std::string_view cache_key,
        std::filesystem::path const &cache_directory = default_texture_cache_directory(),
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
