#include "assets/texture_pipeline.hxx"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <glm/gtc/packing.hpp>
#include <ktx.h>
#include <stb_image_resize2.h>

#include "core/logger.hxx"
#include "gpu/image.hxx"

namespace {

    constexpr int texture_pipeline_encoder_version = 3;

    auto make_error(TexturePipelineErrorType type, std::string_view message = {},
                    std::source_location location = std::source_location::current()) -> TexturePipelineError {
        return TexturePipelineError{
                .type = type,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{message},
                        .location = location,
                }},
        };
    }

    struct KtxTextureDeleter {
        auto operator()(ktxTexture2 *texture) const noexcept -> void {
            if (texture != nullptr) {
                ktxTexture_Destroy(ktxTexture(texture));
            }
        }
    };

    using KtxTexturePtr = std::unique_ptr<ktxTexture2, KtxTextureDeleter>;

    [[nodiscard]]
    auto fnv1a(std::string_view data) noexcept -> std::uint64_t {
        auto hash = std::uint64_t{14695981039346656037ULL};

        for (auto const byte: data) {
            hash ^= static_cast<std::uint8_t>(byte);
            hash *= std::uint64_t{1099511628211ULL};
        }

        return hash;
    }

    [[nodiscard]]
    auto to_hex(std::uint64_t value) -> std::string {
        static constexpr std::string_view digits = "0123456789abcdef";

        std::string text(16, '0');

        for (int index = 15; index >= 0; --index) {
            text[static_cast<std::size_t>(index)] = digits[value & 0xF];
            value >>= 4;
        }

        return text;
    }

    [[nodiscard]]
    auto cache_path_for(std::string_view identity, TextureRole role, std::filesystem::path const &cache_directory,
                        std::string_view stem) -> std::filesystem::path {
        auto const key = std::format("{}|{}|{}", identity, static_cast<int>(role), texture_pipeline_encoder_version);

        return cache_directory / std::format("{}.{}.ktx2", stem, to_hex(fnv1a(key)));
    }

    [[nodiscard]]
    auto to_rgba8(DecodedImage const &decoded) -> std::vector<std::byte> {
        auto const span = decoded.span();

        if (decoded.format() != VK_FORMAT_R16G16B16A16_SFLOAT) {
            return std::vector<std::byte>{span.begin(), span.end()};
        }

        auto const *halfs = reinterpret_cast<std::uint16_t const *>(span.data());
        auto const texel_count = span.size_bytes() / (4 * sizeof(std::uint16_t));

        std::vector<std::byte> rgba8(texel_count * 4);
        auto *out = reinterpret_cast<std::uint8_t *>(rgba8.data());

        for (std::size_t texel = 0; texel < texel_count; ++texel) {
            for (std::size_t channel = 0; channel < 4; ++channel) {
                auto const value = glm::unpackHalf1x16(halfs[texel * 4 + channel]);
                auto const clamped = std::clamp(value, 0.0F, 1.0F);

                out[texel * 4 + channel] = static_cast<std::uint8_t>(std::lround(clamped * 255.0F));
            }
        }

        return rgba8;
    }

    struct RawMip {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::byte> pixels;
    };

    [[nodiscard]]
    auto generate_mip_chain(std::vector<std::byte> base_rgba8, std::uint32_t width, std::uint32_t height,
                            TextureRole role) -> std::vector<RawMip> {
        auto const mip_count = static_cast<std::uint32_t>(std::bit_width(std::max(width, height)));

        std::vector<RawMip> mips;
        mips.reserve(mip_count);
        mips.push_back(RawMip{.width = width, .height = height, .pixels = std::move(base_rgba8)});

        for (std::uint32_t level = 1; level < mip_count; ++level) {
            auto const &prev = mips.back();
            auto const next_width = prev.width > 1 ? prev.width / 2 : 1;
            auto const next_height = prev.height > 1 ? prev.height / 2 : 1;

            std::vector<std::byte> next(static_cast<std::size_t>(next_width) * next_height * 4);

            auto const *input = reinterpret_cast<unsigned char const *>(prev.pixels.data());
            auto *output = reinterpret_cast<unsigned char *>(next.data());

            if (role == TextureRole::colour) {
                stbir_resize_uint8_srgb(input, static_cast<int>(prev.width), static_cast<int>(prev.height), 0, output,
                                        static_cast<int>(next_width), static_cast<int>(next_height), 0, STBIR_RGBA);
            } else {
                stbir_resize_uint8_linear(input, static_cast<int>(prev.width), static_cast<int>(prev.height), 0, output,
                                          static_cast<int>(next_width), static_cast<int>(next_height), 0, STBIR_RGBA);
            }

            mips.push_back(RawMip{.width = next_width, .height = next_height, .pixels = std::move(next)});
        }

        return mips;
    }

    [[nodiscard]]
    auto encode_uastc(std::vector<RawMip> const &mips, TextureRole role)
            -> std::expected<KtxTexturePtr, TexturePipelineError> {
        ktxTextureCreateInfo create_info{};
        create_info.vkFormat = role == TextureRole::colour ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        create_info.baseWidth = mips.front().width;
        create_info.baseHeight = mips.front().height;
        create_info.baseDepth = 1;
        create_info.numDimensions = 2;
        create_info.numLevels = static_cast<std::uint32_t>(mips.size());
        create_info.numLayers = 1;
        create_info.numFaces = 1;
        create_info.isArray = KTX_FALSE;
        create_info.generateMipmaps = KTX_FALSE;

        ktxTexture2 *raw = nullptr;

        if (ktxTexture2_Create(&create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &raw) != KTX_SUCCESS) {
            return std::unexpected(make_error(TexturePipelineErrorType::encode_failed, "ktxTexture2_Create failed"));
        }

        KtxTexturePtr texture{raw};

        for (std::uint32_t level = 0; level < mips.size(); ++level) {
            auto const &mip = mips[level];

            auto const result = ktxTexture_SetImageFromMemory(ktxTexture(texture.get()), level, 0, 0,
                                                              reinterpret_cast<ktx_uint8_t const *>(mip.pixels.data()),
                                                              mip.pixels.size());

            if (result != KTX_SUCCESS) {
                return std::unexpected(
                        make_error(TexturePipelineErrorType::encode_failed, "ktxTexture_SetImageFromMemory failed"));
            }
        }

        ktxBasisParams params{};
        params.structSize = sizeof(params);
        params.uastc = KTX_TRUE;
        params.threadCount = 1;
        params.normalMap = role == TextureRole::normal_map ? KTX_TRUE : KTX_FALSE;
        params.uastcFlags = static_cast<ktx_pack_uastc_flags>(KTX_PACK_UASTC_LEVEL_FASTEST);

        if (ktxTexture2_CompressBasisEx(texture.get(), &params) != KTX_SUCCESS) {
            return std::unexpected(
                    make_error(TexturePipelineErrorType::encode_failed, "ktxTexture2_CompressBasisEx failed"));
        }

        return texture;
    }

    auto write_cache_atomic(ktxTexture2 *texture, std::filesystem::path const &cache_path) -> void {
        std::error_code ec;

        std::filesystem::create_directories(cache_path.parent_path(), ec);

        if (ec) {
            warn("texture_pipeline: could not create cache directory '{}': {}", cache_path.parent_path().string(),
                 ec.message());
            return;
        }

        static std::atomic<std::uint64_t> temp_counter{0};

        auto const temp_path = cache_path.string() +
                               std::format(".tmp-{:x}-{}", std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                           temp_counter.fetch_add(1, std::memory_order_relaxed));

        if (ktxTexture2_WriteToNamedFile(texture, temp_path.c_str()) != KTX_SUCCESS) {
            warn("texture_pipeline: failed to write cache file '{}'", temp_path);
            std::filesystem::remove(temp_path, ec);
            return;
        }

        std::filesystem::rename(temp_path, cache_path, ec);

        if (ec) {
            warn("texture_pipeline: failed to install cache file '{}': {}", cache_path.string(), ec.message());
            std::filesystem::remove(temp_path, ec);
        }
    }

    [[nodiscard]]
    auto extract_compressed_texture(ktxTexture2 *texture, FlyString debug_name) -> CompressedTexture {
        CompressedTexture result;
        result.format = static_cast<VkFormat>(texture->vkFormat);
        result.width = texture->baseWidth;
        result.height = texture->baseHeight;
        result.debug_name = debug_name;

        auto const *base_data = ktxTexture_GetData(ktxTexture(texture));
        auto const total_size = ktxTexture_GetDataSize(ktxTexture(texture));

        result.data.assign(reinterpret_cast<std::byte const *>(base_data),
                           reinterpret_cast<std::byte const *>(base_data) + total_size);

        auto width = texture->baseWidth;
        auto height = texture->baseHeight;

        result.mips.reserve(texture->numLevels);

        for (std::uint32_t level = 0; level < texture->numLevels; ++level) {
            ktx_size_t offset = 0;
            ktxTexture_GetImageOffset(ktxTexture(texture), level, 0, 0, &offset);

            auto const size = ktxTexture_GetImageSize(ktxTexture(texture), level);

            result.mips.push_back(CompressedMipLevel{
                    .width = width,
                    .height = height,
                    .byte_offset = static_cast<std::uint32_t>(offset),
                    .byte_length = static_cast<std::uint32_t>(size),
            });

            width = width > 1 ? width / 2 : 1;
            height = height > 1 ? height / 2 : 1;
        }

        return result;
    }

    [[nodiscard]]
    auto transcode_target(TextureRole role) noexcept -> ktx_transcode_fmt_e {
        return role == TextureRole::normal_map ? KTX_TTF_BC5_RG : KTX_TTF_BC7_RGBA;
    }

    // Cache files are written by libktx after transcoding, so they are plain 2D KTX2 containers with no
    // supercompression. Reading them directly lets each level range land in its final buffer with a single copy.
    constexpr std::uint32_t ktx2_header_bytes = 80;
    constexpr std::uint32_t ktx2_level_entry_bytes = 24;
    constexpr std::uint32_t ktx2_max_levels = 32;
    constexpr std::uint32_t preview_max_extent = 256;

    constexpr std::array<std::uint8_t, 12> ktx2_identifier{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                           0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

    struct Ktx2Level {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
    };

    class Ktx2File {
    public:
        // Missing files are an ordinary cache miss, so only an unreadable or unexpected layout is reported.
        [[nodiscard]]
        static auto open(std::filesystem::path const &path) -> std::optional<Ktx2File> {
            Ktx2File file;
            file.stream_.open(path, std::ios::binary);

            if (!file.stream_) {
                return std::nullopt;
            }

            std::error_code ec;
            auto const file_size = std::filesystem::file_size(path, ec);

            if (ec || !file.read_layout(file_size)) {
                warn("texture_pipeline: cache file '{}' has an unexpected layout, re-encoding", path.string());
                return std::nullopt;
            }

            return file;
        }

        // First level whose largest side fits in the preview budget, or 0 when the texture is already that small.
        [[nodiscard]]
        auto preview_level() const noexcept -> std::uint32_t {
            auto const largest = std::max(width_, height_);

            for (std::uint32_t level = 0; level + 1 < levels_.size(); ++level) {
                if ((largest >> level) <= preview_max_extent) {
                    return level;
                }
            }

            return 0;
        }

        // Reads levels [first_level, end) as a texture whose base is first_level. Smaller levels are stored first in
        // the file, so this is one contiguous range.
        [[nodiscard]]
        auto read(std::uint32_t first_level, FlyString debug_name) -> std::optional<CompressedTexture> {
            if (first_level >= levels_.size()) {
                return std::nullopt;
            }

            auto range_begin = std::numeric_limits<std::uint64_t>::max();
            std::uint64_t range_end = 0;

            for (auto level = first_level; level < levels_.size(); ++level) {
                range_begin = std::min(range_begin, levels_[level].offset);
                range_end = std::max(range_end, levels_[level].offset + levels_[level].length);
            }

            if (range_end - range_begin > std::numeric_limits<std::uint32_t>::max()) {
                return std::nullopt;
            }

            CompressedTexture result;
            result.format = format_;
            result.width = std::max<std::uint32_t>(width_ >> first_level, 1);
            result.height = std::max<std::uint32_t>(height_ >> first_level, 1);
            result.debug_name = debug_name;
            result.data.resize(range_end - range_begin);

            stream_.seekg(static_cast<std::streamoff>(range_begin));
            stream_.read(reinterpret_cast<char *>(result.data.data()), static_cast<std::streamsize>(result.data.size()));

            if (!stream_) {
                return std::nullopt;
            }

            result.mips.reserve(levels_.size() - first_level);

            for (auto level = first_level; level < levels_.size(); ++level) {
                result.mips.push_back(CompressedMipLevel{
                        .width = std::max<std::uint32_t>(width_ >> level, 1),
                        .height = std::max<std::uint32_t>(height_ >> level, 1),
                        .byte_offset = static_cast<std::uint32_t>(levels_[level].offset - range_begin),
                        .byte_length = static_cast<std::uint32_t>(levels_[level].length),
                });
            }

            return result;
        }

    private:
        [[nodiscard]]
        auto read_layout(std::uint64_t file_size) -> bool {
            std::array<std::byte, ktx2_header_bytes> header{};
            stream_.read(reinterpret_cast<char *>(header.data()), static_cast<std::streamsize>(header.size()));

            if (!stream_ || std::memcmp(header.data(), ktx2_identifier.data(), ktx2_identifier.size()) != 0) {
                return false;
            }

            auto const field = [&](std::size_t offset) {
                std::uint32_t value = 0;
                std::memcpy(&value, header.data() + offset, sizeof(value));
                return value;
            };

            auto const depth = field(28);
            auto const layers = field(32);
            auto const faces = field(36);
            auto const level_count = field(40);
            auto const supercompression = field(44);

            if (depth > 1 || layers > 1 || faces != 1 || supercompression != 0 || level_count == 0 ||
                level_count > ktx2_max_levels) {
                return false;
            }

            format_ = static_cast<VkFormat>(field(12));
            width_ = field(20);
            height_ = field(24);

            if (width_ == 0 || height_ == 0) {
                return false;
            }

            std::array<std::byte, ktx2_max_levels * ktx2_level_entry_bytes> index{};
            stream_.read(reinterpret_cast<char *>(index.data()),
                         static_cast<std::streamsize>(level_count * ktx2_level_entry_bytes));

            if (!stream_) {
                return false;
            }

            levels_.resize(level_count);

            for (std::uint32_t level = 0; level < level_count; ++level) {
                std::memcpy(&levels_[level].offset, index.data() + level * ktx2_level_entry_bytes, 8);
                std::memcpy(&levels_[level].length, index.data() + level * ktx2_level_entry_bytes + 8, 8);

                if (levels_[level].offset > file_size || levels_[level].length > file_size - levels_[level].offset) {
                    return false;
                }
            }

            return true;
        }

        std::ifstream stream_;
        VkFormat format_ = VK_FORMAT_UNDEFINED;
        std::uint32_t width_ = 0;
        std::uint32_t height_ = 0;
        std::vector<Ktx2Level> levels_;
    };

    [[nodiscard]]
    auto try_load_cached(std::filesystem::path const &cache_path, FlyString debug_name, ModelLoadProfile *profile,
                         TexturePreviewSlot *preview) -> std::optional<CompressedTexture> {
        ScopedProfileSample lookup_sample{profile != nullptr ? &profile->texture_cache_lookup_ns : nullptr};

        auto file = Ktx2File::open(cache_path);

        if (!file) {
            return std::nullopt;
        }

        if (preview != nullptr) {
            if (auto const level = file->preview_level(); level > 0) {
                if (auto tail = file->read(level, debug_name);
                    tail.has_value() && !validate_compressed_texture(*tail).has_value()) {
                    preview->publish(std::move(*tail));
                }
            }
        }

        auto texture = file->read(0, debug_name);

        lookup_sample.stop();

        if (!texture) {
            warn("texture_pipeline: cache file '{}' failed to load, re-encoding", cache_path.string());
            return std::nullopt;
        }

        debug("texture_pipeline: '{}' loaded from cache '{}'", debug_name, cache_path.string());

        if (profile != nullptr) {
            profile->texture_cache_hits.fetch_add(1, std::memory_order_relaxed);
        }

        if (auto const problem = validate_compressed_texture(*texture); problem.has_value()) {
            warn("texture_pipeline: cache file '{}' is invalid ({}), re-encoding", cache_path.string(), *problem);
            return std::nullopt;
        }

        return texture;
    }

    [[nodiscard]]
    auto encode_and_transcode(std::vector<std::byte> base_rgba8, std::uint32_t width, std::uint32_t height,
                              TextureRole role, std::filesystem::path const &cache_path, FlyString debug_name,
                              ModelLoadProfile *profile) -> std::expected<CompressedTexture, TexturePipelineError> {
        if (profile != nullptr) {
            profile->texture_cache_misses.fetch_add(1, std::memory_order_relaxed);
        }

        ScopedProfileSample mip_sample{profile != nullptr ? &profile->texture_mip_generation_ns : nullptr};

        auto mips = generate_mip_chain(std::move(base_rgba8), width, height, role);

        mip_sample.stop();

        ScopedProfileSample encode_sample{profile != nullptr ? &profile->texture_encode_ns : nullptr};

        auto encoded = encode_uastc(mips, role);

        encode_sample.stop();

        if (!encoded) {
            return std::unexpected(encoded.error());
        }

        auto texture = std::move(*encoded);

        ScopedProfileSample transcode_sample{profile != nullptr ? &profile->texture_transcode_ns : nullptr};

        if (ktxTexture2_TranscodeBasis(texture.get(), transcode_target(role), 0) != KTX_SUCCESS) {
            return std::unexpected(
                    make_error(TexturePipelineErrorType::transcode_failed, "ktxTexture2_TranscodeBasis failed"));
        }

        transcode_sample.stop();

        ScopedProfileSample const write_sample{profile != nullptr ? &profile->texture_cache_write_ns : nullptr};

        write_cache_atomic(texture.get(), cache_path);

        return extract_compressed_texture(texture.get(), debug_name);
    }

    [[nodiscard]]
    auto compress_decoded_image(DecodedImage const &decoded, TextureRole role, std::filesystem::path const &cache_path,
                                FlyString debug_name,
                                ModelLoadProfile *profile) -> std::expected<CompressedTexture, TexturePipelineError> {
        auto const width = decoded.width();
        auto const height = decoded.height();
        auto rgba8 = to_rgba8(decoded);

        return encode_and_transcode(std::move(rgba8), width, height, role, cache_path, debug_name, profile);
    }

}

auto default_texture_cache_directory() -> std::filesystem::path {
    if (auto const *xdg_cache = std::getenv("XDG_CACHE_HOME"); xdg_cache != nullptr && *xdg_cache != '\0') {
        return std::filesystem::path{xdg_cache} / "ktx2";
    }

    if (auto const *home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path{home} / ".cache" / "ktx2";
    }

    return cache_path("ktx2").absolute();
}

auto load_compressed_texture(AssetPath const &asset_path, TextureRole role,
                             std::filesystem::path const &cache_directory,
                             std::shared_ptr<ModelLoadProfile> const &profile,
                             std::shared_ptr<TexturePreviewSlot> const &preview)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    auto const &source_path = asset_path.absolute();
    auto *const profile_ptr = profile.get();

    if (profile_ptr != nullptr) {
        profile_ptr->texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    std::error_code ec;

    if (!std::filesystem::exists(source_path, ec) || ec) {
        return std::unexpected(make_error(TexturePipelineErrorType::source_not_found,
                                          std::format("'{}' does not exist", source_path.string())));
    }

    auto const file_size = std::filesystem::file_size(source_path, ec);

    if (ec) {
        return std::unexpected(make_error(TexturePipelineErrorType::source_not_found,
                                          std::format("could not stat '{}'", source_path.string())));
    }

    auto const write_time = std::filesystem::last_write_time(source_path, ec);

    if (ec) {
        return std::unexpected(make_error(TexturePipelineErrorType::source_not_found,
                                          std::format("could not stat '{}'", source_path.string())));
    }

    auto const absolute_path = std::filesystem::absolute(source_path, ec);
    auto const path_identity = ec ? source_path.string() : absolute_path.string();

    auto const identity = std::format("{}|{}|{}", path_identity, file_size, write_time.time_since_epoch().count());

    auto const stem = source_path.stem().string();
    auto const cache_path = cache_path_for(identity, role, cache_directory, stem);

    if (auto cached = try_load_cached(cache_path, FlyString{stem}, profile_ptr, preview.get()); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, decoding from source '{}'", stem, source_path.string());

    ScopedProfileSample decode_sample{profile_ptr != nullptr ? &profile_ptr->texture_decode_ns : nullptr};

    auto decoded = DecodedImage::load_from_file(
            source_path.string(), role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    decode_sample.stop();

    if (!decoded.has_value()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed,
                                          std::format("failed to decode '{}'", source_path.string())));
    }

    return compress_decoded_image(*decoded, role, cache_path, FlyString{stem}, profile_ptr);
}

auto load_compressed_texture_from_encoded_memory(std::span<std::byte const> encoded_bytes, TextureRole role,
                                                 std::string_view cache_key,
                                                 std::filesystem::path const &cache_directory,
                                                 std::shared_ptr<ModelLoadProfile> const &profile,
                                                 std::shared_ptr<TexturePreviewSlot> const &preview)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    auto *const profile_ptr = profile.get();

    if (profile_ptr != nullptr) {
        profile_ptr->texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (encoded_bytes.empty()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "empty encoded image buffer"));
    }

    auto const identity = std::format("encoded-memory|{}", cache_key);
    auto const cache_path = cache_path_for(identity, role, cache_directory, "embedded");

    if (auto cached = try_load_cached(cache_path, FlyString{cache_key}, profile_ptr, preview.get()); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, decoding from embedded memory", cache_key);

    ScopedProfileSample decode_sample{profile_ptr != nullptr ? &profile_ptr->texture_decode_ns : nullptr};

    auto decoded = DecodedImage::load_from_memory(
            encoded_bytes, role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    decode_sample.stop();

    if (!decoded.has_value()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "failed to decode embedded image"));
    }

    return compress_decoded_image(*decoded, role, cache_path, FlyString{cache_key}, profile_ptr);
}

auto load_compressed_texture_from_memory(std::span<std::byte const> rgba_pixels, std::uint32_t width,
                                         std::uint32_t height, TextureRole role, std::string_view cache_key,
                                         std::filesystem::path const &cache_directory,
                                         std::shared_ptr<ModelLoadProfile> const &profile)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    auto *const profile_ptr = profile.get();

    if (profile_ptr != nullptr) {
        profile_ptr->texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (width == 0 || height == 0 || rgba_pixels.size_bytes() != static_cast<std::size_t>(width) * height * 4) {
        return std::unexpected(
                make_error(TexturePipelineErrorType::decode_failed, "invalid in-memory image dimensions"));
    }

    auto const identity = std::format("memory|{}", cache_key);
    auto const cache_path = cache_path_for(identity, role, cache_directory, "embedded");

    if (auto cached = try_load_cached(cache_path, FlyString{cache_key}, profile_ptr, nullptr); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, encoding from raw pixel memory", cache_key);

    std::vector<std::byte> base{rgba_pixels.begin(), rgba_pixels.end()};

    return encode_and_transcode(std::move(base), width, height, role, cache_path, FlyString{cache_key}, profile_ptr);
}
