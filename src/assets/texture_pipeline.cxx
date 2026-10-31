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
#include <zstd.h>

#include "assets/gpu_texture_service.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "gpu/image.hxx"

namespace {

    // Levels of new cache files are zstd-compressed. The pixels are the same as before, so the cache version (and so
    // every existing cache entry) stays as it is; both kinds are read.
    constexpr std::uint32_t zstd_compression_level = 3;

    std::atomic<float> rdo_lambda{0.0F};
    std::atomic<std::uint32_t> max_extent{0};

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

    // Where a texture is cached: the key is the identity of the source plus what shapes the output; the label only
    // names the file, and the source (when it is a file) lets the entry be pruned once that is gone.
    struct CacheTarget {
        TextureCache const &cache;
        std::string key;
        std::string label;
        std::string source;
    };

    [[nodiscard]]
    auto make_cache_target(TextureCache const &cache, std::string_view identity, TextureRole role,
                           std::string_view label, std::string_view source) -> CacheTarget {
        auto key = std::format("{}|{}", identity, static_cast<int>(role));

        // Only when on, so entries made without RDO keep their keys.
        if (auto const lambda = rdo_lambda.load(std::memory_order_relaxed); lambda > 0.0F) {
            key += std::format("|rdo{}", lambda);
        }

        return CacheTarget{.cache = cache, .key = std::move(key), .label = std::string{label}, .source = std::string{source}};
    }

    // An embedded texture's cache key is "<model path>#<what in it>#<size>".
    [[nodiscard]]
    auto model_of_embedded_key(std::string_view cache_key) -> std::string {
        auto const hash = cache_key.find('#');

        return hash != std::string_view::npos && cache_key.starts_with('/') ? std::string{cache_key.substr(0, hash)}
                                                                            : std::string{};
    }

    [[nodiscard]]
    auto to_rgba8(DecodedImage &&decoded) -> std::vector<std::byte> {
        // 8-bit images already are RGBA8: take their pixels instead of copying tens of megabytes.
        if (decoded.format() != VK_FORMAT_R16G16B16A16_SFLOAT) {
            return std::move(decoded).release_pixels();
        }

        auto const span = decoded.span();

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
        // Textures are only ever transcoded to BC7 and BC5, never to ETC1, so the encoder is told not to spend its time
        // on the hints that make an ETC1 transcode good; they were most of the encode and do not touch the BC7 output.
        params.uastcFlags = static_cast<ktx_pack_uastc_flags>(
                KTX_PACK_UASTC_LEVEL_FASTEST | KTX_PACK_UASTC_ETC1_FASTEST_HINTS |
                KTX_PACK_UASTC__ETC1_DISABLE_FLIP_AND_INDIVIDUAL);

        if (auto const lambda = rdo_lambda.load(std::memory_order_relaxed); lambda > 0.0F) {
            params.uastcRDO = KTX_TRUE;
            // Normal maps want about half the strength of colour (libktx suggests 0.25 to 0.75 for them).
            params.uastcRDOQualityScalar = role == TextureRole::normal_map ? lambda * 0.5F : lambda;
        }

        if (ktxTexture2_CompressBasisEx(texture.get(), &params) != KTX_SUCCESS) {
            return std::unexpected(
                    make_error(TexturePipelineErrorType::encode_failed, "ktxTexture2_CompressBasisEx failed"));
        }

        return texture;
    }

    // Writes the texture where the cache wants it and installs it; a failure only costs the next load a re-encode.
    auto write_to_cache(ktxTexture2 *texture, CacheTarget const &target) -> void {
        auto pending = target.cache.begin(target.key, target.label);

        if (!pending) {
            return;
        }

        if (ktxTexture2_WriteToNamedFile(texture, pending->path().c_str()) != KTX_SUCCESS) {
            warn("texture_pipeline: failed to write cache file '{}'", pending->path().string());
            return;
        }

        static_cast<void>(target.cache.commit(std::move(*pending), target.source));
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

    // Drops the top mips of a texture larger than the configured limit.
    [[nodiscard]]
    auto apply_max_extent(CompressedTexture texture) -> CompressedTexture {
        auto const limit = max_extent.load(std::memory_order_relaxed);
        std::size_t skip = 0;

        while (limit != 0 && skip + 1 < texture.mips.size() &&
               std::max(texture.mips[skip].width, texture.mips[skip].height) > limit) {
            ++skip;
        }

        if (skip == 0) {
            return texture;
        }

        std::vector<std::byte> data;
        std::vector<CompressedMipLevel> mips;
        mips.reserve(texture.mips.size() - skip);

        for (auto level = skip; level < texture.mips.size(); ++level) {
            auto const &mip = texture.mips[level];
            auto const begin = texture.data.begin() + mip.byte_offset;

            mips.push_back(CompressedMipLevel{
                    .width = mip.width,
                    .height = mip.height,
                    .byte_offset = static_cast<std::uint32_t>(data.size()),
                    .byte_length = mip.byte_length,
            });
            data.insert(data.end(), begin, begin + mip.byte_length);
        }

        texture.width = mips.front().width;
        texture.height = mips.front().height;
        texture.data = std::move(data);
        texture.mips = std::move(mips);

        return texture;
    }

    [[nodiscard]]
    auto transcode_target(TextureRole role) noexcept -> ktx_transcode_fmt_e {
        return role == TextureRole::normal_map ? KTX_TTF_BC5_RG : KTX_TTF_BC7_RGBA;
    }

    // Cache files are written by libktx after transcoding: plain 2D KTX2 containers, either with no supercompression or
    // with per-level zstd. Reading them directly lets each level land in its final buffer with a single copy (or a
    // single inflate).
    constexpr std::uint32_t supercompression_none = 0;
    constexpr std::uint32_t supercompression_zstd = 2;
    constexpr std::uint32_t ktx2_header_bytes = 80;
    constexpr std::uint32_t ktx2_level_entry_bytes = 24;
    constexpr std::uint32_t ktx2_max_levels = 32;
    constexpr std::uint32_t preview_max_extent = 256;

    constexpr std::array<std::uint8_t, 12> ktx2_identifier{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                           0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

    struct Ktx2Level {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
        // Size once inflated; equals `length` without supercompression.
        std::uint64_t uncompressed_length = 0;
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

        // First level whose largest side fits in `limit`; the last level if none does. 0 (no limit) gives level 0.
        [[nodiscard]]
        auto level_for_extent(std::uint32_t limit) const noexcept -> std::uint32_t {
            auto const largest = std::max(width_, height_);
            std::uint32_t level = 0;

            while (limit != 0 && level + 1 < levels_.size() && (largest >> level) > limit) {
                ++level;
            }

            return level;
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

            if (supercompression_ == supercompression_zstd) {
                return read_zstd(first_level, debug_name);
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

            auto result = make_result(first_level, debug_name);
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
        auto make_result(std::uint32_t first_level, FlyString debug_name) const -> CompressedTexture {
            CompressedTexture result;
            result.format = format_;
            result.width = std::max<std::uint32_t>(width_ >> first_level, 1);
            result.height = std::max<std::uint32_t>(height_ >> first_level, 1);
            result.debug_name = debug_name;
            return result;
        }

        // Each level is its own zstd frame. They are inflated into one buffer in file order (smallest level first),
        // which is the layout the uncompressed path produces.
        [[nodiscard]]
        auto read_zstd(std::uint32_t first_level, FlyString debug_name) -> std::optional<CompressedTexture> {
            std::uint64_t total = 0;

            for (auto level = first_level; level < levels_.size(); ++level) {
                total += levels_[level].uncompressed_length;
            }

            if (total > std::numeric_limits<std::uint32_t>::max()) {
                return std::nullopt;
            }

            auto result = make_result(first_level, debug_name);
            result.data.resize(total);
            result.mips.resize(levels_.size() - first_level);

            std::vector<std::byte> compressed;
            std::uint64_t cursor = 0;

            for (auto level = static_cast<std::uint32_t>(levels_.size()); level-- > first_level;) {
                auto const &entry = levels_[level];

                compressed.resize(entry.length);
                stream_.seekg(static_cast<std::streamoff>(entry.offset));
                stream_.read(reinterpret_cast<char *>(compressed.data()), static_cast<std::streamsize>(entry.length));

                if (!stream_) {
                    return std::nullopt;
                }

                auto const inflated = ZSTD_decompress(result.data.data() + cursor, entry.uncompressed_length,
                                                      compressed.data(), compressed.size());

                if (ZSTD_isError(inflated) != 0 || inflated != entry.uncompressed_length) {
                    return std::nullopt;
                }

                result.mips[level - first_level] = CompressedMipLevel{
                        .width = std::max<std::uint32_t>(width_ >> level, 1),
                        .height = std::max<std::uint32_t>(height_ >> level, 1),
                        .byte_offset = static_cast<std::uint32_t>(cursor),
                        .byte_length = static_cast<std::uint32_t>(entry.uncompressed_length),
                };
                cursor += entry.uncompressed_length;
            }

            return result;
        }

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

            auto const known_supercompression =
                    supercompression == supercompression_none || supercompression == supercompression_zstd;

            if (depth > 1 || layers > 1 || faces != 1 || !known_supercompression || level_count == 0 ||
                level_count > ktx2_max_levels) {
                return false;
            }

            supercompression_ = supercompression;
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
                std::memcpy(&levels_[level].uncompressed_length, index.data() + level * ktx2_level_entry_bytes + 16, 8);

                if (levels_[level].offset > file_size || levels_[level].length > file_size - levels_[level].offset) {
                    return false;
                }
            }

            return true;
        }

        std::ifstream stream_;
        std::uint32_t supercompression_ = supercompression_none;
        VkFormat format_ = VK_FORMAT_UNDEFINED;
        std::uint32_t width_ = 0;
        std::uint32_t height_ = 0;
        std::vector<Ktx2Level> levels_;
    };

    [[nodiscard]]
    auto try_load_cached(CacheTarget const &target, FlyString debug_name, ModelLoadProfile *profile,
                         TexturePreviewSlot *preview) -> std::optional<CompressedTexture> {
        ScopedProfileSample lookup_sample{profile != nullptr ? &profile->texture_cache_lookup_ns : nullptr};

        auto const cache_path = target.cache.find(target.key);

        if (!cache_path) {
            return std::nullopt;
        }

        auto file = Ktx2File::open(*cache_path);

        if (!file) {
            target.cache.discard(target.key);
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

        auto texture = file->read(file->level_for_extent(max_extent.load(std::memory_order_relaxed)), debug_name);

        lookup_sample.stop();

        if (!texture) {
            warn("texture_pipeline: cache file '{}' failed to load, re-encoding", cache_path->string());
            target.cache.discard(target.key);
            return std::nullopt;
        }

        debug("texture_pipeline: '{}' loaded from cache '{}'", debug_name, cache_path->string());

        if (profile != nullptr) {
            profile->texture_cache_hits.fetch_add(1, std::memory_order_relaxed);
        }

        if (auto const problem = validate_compressed_texture(*texture); problem.has_value()) {
            warn("texture_pipeline: cache file '{}' is invalid ({}), re-encoding", cache_path->string(), *problem);
            target.cache.discard(target.key);
            return std::nullopt;
        }

        return texture;
    }

    // The cache-hit path shows a small tail of the mip chain while the big levels load; a miss would show nothing until
    // the whole texture is encoded. The tail is a few percent of the work, so encode it first and publish it.
    auto publish_cold_preview(std::vector<RawMip> const &mips, TextureRole role, FlyString debug_name,
                              TexturePreviewSlot &preview) -> void {
        auto first = std::size_t{0};

        while (first + 1 < mips.size() && std::max(mips[first].width, mips[first].height) > preview_max_extent) {
            ++first;
        }

        if (first == 0) {
            return;
        }

        auto const tail = std::vector<RawMip>(mips.begin() + static_cast<std::ptrdiff_t>(first), mips.end());
        auto encoded = encode_uastc(tail, role);

        if (!encoded || ktxTexture2_TranscodeBasis(encoded->get(), transcode_target(role), 0) != KTX_SUCCESS) {
            return;
        }

        auto texture = extract_compressed_texture(encoded->get(), debug_name);

        if (!validate_compressed_texture(texture).has_value()) {
            preview.publish(std::move(texture));
        }
    }

    // Compresses the mip chain on the GPU, when a service is installed: BC5 for normal maps, BC7 for the rest. Null on
    // any failure, and the caller compresses on the CPU.
    [[nodiscard]]
    auto compress_on_gpu(std::vector<RawMip> const &mips, TextureRole role) -> KtxTexturePtr {
        if (mips.empty()) {
            return nullptr;
        }

        auto const service = gpu_texture_service();

        if (service == nullptr) {
            return nullptr;
        }

        std::vector<GpuTextureLevel> levels;
        levels.reserve(mips.size());

        for (auto const &mip: mips) {
            levels.push_back(GpuTextureLevel{.rgba = mip.pixels, .width = mip.width, .height = mip.height});
        }

        auto const format = role == TextureRole::normal_map ? GpuBlockFormat::bc5 : GpuBlockFormat::bc7;
        auto outcome = service->submit(levels, format).get();

        if (!outcome) {
            debug("texture_pipeline: GPU compression unavailable ({}), using the CPU", outcome.error());

            return nullptr;
        }

        ktxTextureCreateInfo create_info{};
        create_info.vkFormat = format == GpuBlockFormat::bc5     ? VK_FORMAT_BC5_UNORM_BLOCK
                               : role == TextureRole::colour ? VK_FORMAT_BC7_SRGB_BLOCK
                                                             : VK_FORMAT_BC7_UNORM_BLOCK;
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
            return nullptr;
        }

        KtxTexturePtr texture{raw};

        for (std::uint32_t level = 0; level < mips.size(); ++level) {
            auto const &blocks = (*outcome)[level];

            if (blocks.size() != block_count_4x4(mips[level].width, mips[level].height) * 16 ||
                ktxTexture_SetImageFromMemory(ktxTexture(texture.get()), level, 0, 0,
                                              reinterpret_cast<ktx_uint8_t const *>(blocks.data()),
                                              blocks.size()) != KTX_SUCCESS) {
                warn("texture_pipeline: the GPU's blocks for level {} do not fit; using the CPU", level);

                return nullptr;
            }
        }

        return texture;
    }

    [[nodiscard]]
    auto encode_and_transcode(std::vector<std::byte> base_rgba8, std::uint32_t width, std::uint32_t height,
                              TextureRole role, CacheTarget const &target, FlyString debug_name,
                              ModelLoadProfile *profile, TexturePreviewSlot *preview = nullptr)
            -> std::expected<CompressedTexture, TexturePipelineError> {
        if (profile != nullptr) {
            profile->texture_cache_misses.fetch_add(1, std::memory_order_relaxed);
        }

        ScopedProfileSample mip_sample{profile != nullptr ? &profile->texture_mip_generation_ns : nullptr};

        auto mips = generate_mip_chain(std::move(base_rgba8), width, height, role);

        mip_sample.stop();

        // Skipped when prepare_texture_preview() already got one out ahead of this load.
        if (preview != nullptr && !preview->has_published()) {
            publish_cold_preview(mips, role, debug_name, *preview);
        }

        ScopedProfileSample encode_sample{profile != nullptr ? &profile->texture_encode_ns : nullptr};

        // The GPU writes BC7 or BC5 directly. The CPU encodes UASTC and transcodes it, which is where the time goes.
        auto texture = compress_on_gpu(mips, role);

        if (texture == nullptr) {
            // A GPU request that failed because we are shutting down must not turn into minutes of CPU encoding.
            if (thread_pool_shutting_down()) {
                return std::unexpected(make_error(TexturePipelineErrorType::cancelled, "shutting down"));
            }

            auto encoded = encode_uastc(mips, role);

            encode_sample.stop();

            if (!encoded) {
                return std::unexpected(encoded.error());
            }

            texture = std::move(*encoded);

            ScopedProfileSample transcode_sample{profile != nullptr ? &profile->texture_transcode_ns : nullptr};

            if (ktxTexture2_TranscodeBasis(texture.get(), transcode_target(role), 0) != KTX_SUCCESS) {
                return std::unexpected(
                        make_error(TexturePipelineErrorType::transcode_failed, "ktxTexture2_TranscodeBasis failed"));
            }

            transcode_sample.stop();
        }

        encode_sample.stop();

        ScopedProfileSample const write_sample{profile != nullptr ? &profile->texture_cache_write_ns : nullptr};

        // Taken before supercompression, which replaces the texture's data with the compressed levels.
        auto extracted = extract_compressed_texture(texture.get(), debug_name);

        if (auto const status = ktxTexture2_DeflateZstd(texture.get(), zstd_compression_level); status != KTX_SUCCESS) {
            warn("texture_pipeline: zstd supercompression failed for '{}' ({}); caching it uncompressed",
                 target.key, static_cast<int>(status));
        }

        write_to_cache(texture.get(), target);

        return apply_max_extent(std::move(extracted));
    }

    [[nodiscard]]
    auto compress_decoded_image(DecodedImage &&decoded, TextureRole role, CacheTarget const &target,
                                FlyString debug_name, ModelLoadProfile *profile,
                                TexturePreviewSlot *preview) -> std::expected<CompressedTexture, TexturePipelineError> {
        // Decoding is done and encoding is the slow part: give up here rather than start it.
        if (thread_pool_shutting_down()) {
            return std::unexpected(make_error(TexturePipelineErrorType::cancelled, "shutting down"));
        }

        auto const width = decoded.width();
        auto const height = decoded.height();
        auto rgba8 = to_rgba8(std::move(decoded));

        return encode_and_transcode(std::move(rgba8), width, height, role, target, debug_name, profile, preview);
    }

}

auto set_texture_rdo_lambda(float lambda) noexcept -> void { rdo_lambda.store(lambda, std::memory_order_relaxed); }

auto set_texture_max_extent(std::uint32_t extent) noexcept -> void { max_extent.store(extent, std::memory_order_relaxed); }

auto load_compressed_texture(AssetPath const &asset_path, TextureRole role,
                             TextureCache const &cache,
                             std::shared_ptr<ModelLoadProfile> const &profile,
                             std::shared_ptr<TexturePreviewSlot> const &preview)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    if (thread_pool_shutting_down()) {
        return std::unexpected(make_error(TexturePipelineErrorType::cancelled, "shutting down"));
    }

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
    auto const target = make_cache_target(cache, identity, role, stem, source_path.string());

    if (auto cached = try_load_cached(target, FlyString{stem}, profile_ptr, preview.get()); cached.has_value()) {
        return std::move(*cached);
    }

    // Another process may be encoding this very texture: wait for it, and take its result.
    auto const producing = target.cache.lock(target.key);

    if (producing) {
        if (auto cached = try_load_cached(target, FlyString{stem}, profile_ptr, preview.get()); cached.has_value()) {
            return std::move(*cached);
        }
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

    return compress_decoded_image(std::move(*decoded), role, target, FlyString{stem}, profile_ptr, preview.get());
}

auto load_compressed_texture_from_encoded_memory(std::span<std::byte const> encoded_bytes, TextureRole role,
                                                 std::string_view cache_key,
                                                 TextureCache const &cache,
                                                 std::shared_ptr<ModelLoadProfile> const &profile,
                                                 std::shared_ptr<TexturePreviewSlot> const &preview)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    if (thread_pool_shutting_down()) {
        return std::unexpected(make_error(TexturePipelineErrorType::cancelled, "shutting down"));
    }

    auto *const profile_ptr = profile.get();

    if (profile_ptr != nullptr) {
        profile_ptr->texture_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (encoded_bytes.empty()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "empty encoded image buffer"));
    }

    auto const identity = std::format("encoded-memory|{}", cache_key);
    auto const target = make_cache_target(cache, identity, role, "embedded", model_of_embedded_key(cache_key));

    if (auto cached = try_load_cached(target, FlyString{cache_key}, profile_ptr, preview.get()); cached.has_value()) {
        return std::move(*cached);
    }

    // Another process may be encoding this very texture: wait for it, and take its result.
    auto const producing = target.cache.lock(target.key);

    if (producing) {
        if (auto cached = try_load_cached(target, FlyString{cache_key}, profile_ptr, preview.get()); cached.has_value()) {
            return std::move(*cached);
        }
    }

    debug("texture_pipeline: '{}' not cached, decoding from embedded memory", cache_key);

    ScopedProfileSample decode_sample{profile_ptr != nullptr ? &profile_ptr->texture_decode_ns : nullptr};

    auto decoded = DecodedImage::load_from_memory(
            encoded_bytes, role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    decode_sample.stop();

    if (!decoded.has_value()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "failed to decode embedded image"));
    }

    return compress_decoded_image(std::move(*decoded), role, target, FlyString{cache_key}, profile_ptr, preview.get());
}

auto load_compressed_texture_from_memory(std::span<std::byte const> rgba_pixels, std::uint32_t width,
                                         std::uint32_t height, TextureRole role, std::string_view cache_key,
                                         TextureCache const &cache,
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
    auto const target = make_cache_target(cache, identity, role, "embedded", {});

    if (auto cached = try_load_cached(target, FlyString{cache_key}, profile_ptr, nullptr); cached.has_value()) {
        return std::move(*cached);
    }

    // Another process may be encoding this very texture: wait for it, and take its result.
    auto const producing = target.cache.lock(target.key);

    if (producing) {
        if (auto cached = try_load_cached(target, FlyString{cache_key}, profile_ptr, nullptr); cached.has_value()) {
            return std::move(*cached);
        }
    }

    debug("texture_pipeline: '{}' not cached, encoding from raw pixel memory", cache_key);

    std::vector<std::byte> base{rgba_pixels.begin(), rgba_pixels.end()};

    return encode_and_transcode(std::move(base), width, height, role, target, FlyString{cache_key}, profile_ptr);
}

auto prepare_texture_preview(AssetPath const &asset_path, TextureRole role, TextureCache const &cache,
                             std::shared_ptr<TexturePreviewSlot> const &preview) -> void {
    if (preview == nullptr || thread_pool_shutting_down()) {
        return;
    }

    auto const &source_path = asset_path.absolute();
    std::error_code ec;
    auto const file_size = std::filesystem::file_size(source_path, ec);

    if (ec) {
        return;
    }

    auto const write_time = std::filesystem::last_write_time(source_path, ec);

    if (ec) {
        return;
    }

    auto const absolute_path = std::filesystem::absolute(source_path, ec);
    auto const identity = std::format("{}|{}|{}", ec ? source_path.string() : absolute_path.string(), file_size,
                                      write_time.time_since_epoch().count());
    auto const stem = source_path.stem().string();
    auto const target = make_cache_target(cache, identity, role, stem, source_path.string());

    if (cache.find(target.key).has_value()) {
        return;
    }

    auto decoded = DecodedImage::load_from_file(
            source_path.string(), role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    if (!decoded.has_value()) {
        return;
    }

    auto const width = decoded->width();
    auto const height = decoded->height();
    auto const mips = generate_mip_chain(to_rgba8(std::move(*decoded)), width, height, role);

    publish_cold_preview(mips, role, FlyString{stem}, *preview);
}

auto prepare_texture_preview_from_encoded_memory(std::span<std::byte const> encoded_bytes, TextureRole role,
                                                 std::string_view cache_key, TextureCache const &cache,
                                                 std::shared_ptr<TexturePreviewSlot> const &preview) -> void {
    if (preview == nullptr || encoded_bytes.empty() || thread_pool_shutting_down()) {
        return;
    }

    auto const target = make_cache_target(cache, std::format("encoded-memory|{}", cache_key), role, "embedded",
                                          model_of_embedded_key(cache_key));

    if (cache.find(target.key).has_value()) {
        return;
    }

    auto decoded = DecodedImage::load_from_memory(
            encoded_bytes, role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    if (!decoded.has_value()) {
        return;
    }

    auto const width = decoded->width();
    auto const height = decoded->height();
    auto const mips = generate_mip_chain(to_rgba8(std::move(*decoded)), width, height, role);

    publish_cold_preview(mips, role, FlyString{cache_key}, *preview);
}
