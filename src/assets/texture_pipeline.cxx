#include "assets/texture_pipeline.hxx"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <source_location>
#include <system_error>
#include <thread>
#include <vector>

#include <glm/gtc/packing.hpp>
#include <ktx.h>
#include <stb_image_resize2.h>

#include "gpu/image.hxx"
#include "core/logger.hxx"

namespace {

    constexpr int texture_pipeline_encoder_version = 1;

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

    // FNV-1a. Cache keys only need to be stable and well-distributed on this
    // machine, not cryptographically strong or portable across processes.
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
        static constexpr char digits[] = "0123456789abcdef";

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

    // DecodedImage hands back either 8-bit RGBA (stb) or RGBA16F (EXR). UASTC
    // encoding always wants 8-bit input; this project's EXR usage is always
    // LDR-range PBR data (normals, roughness), never true HDR, so clamping to
    // [0,1] and rescaling loses nothing that mattered.
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

                out[texel * 4 + channel] = static_cast<std::uint8_t>(clamped * 255.0F + 0.5F);
            }
        }

        return rgba8;
    }

    struct RawMip {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::byte> pixels;
    };

    // Gamma-correct downsampling for colour data, linear for everything else
    // -- a plain box filter over sRGB-encoded texels darkens mips, which the
    // runtime GPU blit path this replaces never had to worry about.
    [[nodiscard]]
    auto generate_mip_chain(std::vector<std::byte> base_rgba8, std::uint32_t width, std::uint32_t height,
                            TextureRole role) -> std::vector<RawMip> {
        auto const mip_count = static_cast<std::uint32_t>(std::bit_width(std::max(width, height)));

        std::vector<RawMip> mips;
        mips.reserve(mip_count);
        mips.push_back(RawMip{width, height, std::move(base_rgba8)});

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
                stbir_resize_uint8_linear(input, static_cast<int>(prev.width), static_cast<int>(prev.height), 0,
                                          output, static_cast<int>(next_width), static_cast<int>(next_height), 0,
                                          STBIR_RGBA);
            }

            mips.push_back(RawMip{next_width, next_height, std::move(next)});
        }

        return mips;
    }

    [[nodiscard]]
    auto encode_uastc(std::vector<RawMip> const &mips, TextureRole role)
            -> std::expected<KtxTexturePtr, TexturePipelineError> {
        ktxTextureCreateInfo create_info{};
        create_info.vkFormat =
                role == TextureRole::colour ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
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
        // Each call here already runs as its own task on Renderer's shared
        // thread pool (see TextureStreamer::request), so letting this spin
        // up hardware_concurrency() internal threads per texture would
        // oversubscribe the CPU during a burst load of many textures at
        // once. A small bounded count instead gives a real speedup for the
        // common case (one or two textures streaming in) without turning a
        // multi-texture burst into thread-thrashing.
        params.threadCount = std::max(1U, std::min(4U, std::thread::hardware_concurrency()));
        params.normalMap = role == TextureRole::normal_map ? KTX_TRUE : KTX_FALSE;
        params.uastcFlags = static_cast<ktx_pack_uastc_flags>(KTX_PACK_UASTC_LEVEL_DEFAULT);

        if (ktxTexture2_CompressBasisEx(texture.get(), &params) != KTX_SUCCESS) {
            return std::unexpected(
                    make_error(TexturePipelineErrorType::encode_failed, "ktxTexture2_CompressBasisEx failed"));
        }

        return texture;
    }

    // Best-effort: a failed cache write just means the next load re-encodes
    // instead of hitting the cache, not a load failure.
    auto write_cache_atomic(ktxTexture2 *texture, std::filesystem::path const &cache_path) -> void {
        std::error_code ec;

        std::filesystem::create_directories(cache_path.parent_path(), ec);

        if (ec) {
            warn("texture_pipeline: could not create cache directory '{}': {}", cache_path.parent_path().string(),
                ec.message());
            return;
        }

        static std::atomic<std::uint64_t> temp_counter{0};

        auto const temp_path =
                cache_path.string() + std::format(".tmp-{:x}-{}", std::hash<std::thread::id>{}(std::this_thread::get_id()),
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
    auto extract_compressed_texture(ktxTexture2 *texture, std::string debug_name) -> CompressedTexture {
        CompressedTexture result;
        result.format = static_cast<VkFormat>(texture->vkFormat);
        result.width = texture->baseWidth;
        result.height = texture->baseHeight;
        result.debug_name = std::move(debug_name);

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

    // Cache hit path: load the cached UASTC container and transcode it.
    // Returns nullopt (not an error) on any miss/corruption so the caller
    // falls back to re-encoding from source -- a torn or stale cache file is
    // an ordinary condition, not a load failure.
    [[nodiscard]]
    auto try_load_cached(std::filesystem::path const &cache_path, TextureRole role, std::string debug_name)
            -> std::optional<CompressedTexture> {
        std::error_code ec;

        if (!std::filesystem::exists(cache_path, ec) || ec) {
            return std::nullopt;
        }

        ktxTexture2 *raw = nullptr;

        auto result = ktxTexture2_CreateFromNamedFile(cache_path.string().c_str(),
                                                       KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw);

        if (result != KTX_SUCCESS || raw == nullptr) {
            warn("texture_pipeline: cache file '{}' failed to load, re-encoding", cache_path.string());
            return std::nullopt;
        }

        KtxTexturePtr texture{raw};

        // UASTC-encoded textures (what this pipeline always writes) stay
        // supercompressionScheme == KTX_SS_NONE -- BASIS_LZ is only used by
        // the separate ETC1S codepath. TranscodeBasis itself is the real
        // arbiter of whether this file is transcodable; let it fail below
        // rather than second-guessing it here.
        result = ktxTexture2_TranscodeBasis(texture.get(), transcode_target(role), 0);

        if (result != KTX_SUCCESS) {
            warn("texture_pipeline: cache file '{}' failed to transcode, re-encoding", cache_path.string());
            return std::nullopt;
        }

        debug("texture_pipeline: '{}' loaded from cache '{}'", debug_name, cache_path.string());

        return extract_compressed_texture(texture.get(), std::move(debug_name));
    }

    [[nodiscard]]
    auto encode_and_transcode(std::vector<std::byte> base_rgba8, std::uint32_t width, std::uint32_t height,
                              TextureRole role, std::filesystem::path const &cache_path, std::string debug_name)
            -> std::expected<CompressedTexture, TexturePipelineError> {
        auto mips = generate_mip_chain(std::move(base_rgba8), width, height, role);

        auto encoded = encode_uastc(mips, role);

        if (!encoded) {
            return std::unexpected(encoded.error());
        }

        auto texture = std::move(*encoded);

        write_cache_atomic(texture.get(), cache_path);

        if (ktxTexture2_TranscodeBasis(texture.get(), transcode_target(role), 0) != KTX_SUCCESS) {
            return std::unexpected(
                    make_error(TexturePipelineErrorType::transcode_failed, "ktxTexture2_TranscodeBasis failed"));
        }

        return extract_compressed_texture(texture.get(), std::move(debug_name));
    }

    // Shared tail of load_compressed_texture() and
    // load_compressed_texture_from_encoded_memory(): both end up with a
    // decoded image and just need it turned into 8-bit RGBA and run through
    // the encoder.
    [[nodiscard]]
    auto compress_decoded_image(DecodedImage const &decoded, TextureRole role,
                                std::filesystem::path const &cache_path, std::string debug_name)
            -> std::expected<CompressedTexture, TexturePipelineError> {
        auto const width = decoded.width();
        auto const height = decoded.height();
        auto rgba8 = to_rgba8(decoded);

        return encode_and_transcode(std::move(rgba8), width, height, role, cache_path, std::move(debug_name));
    }

} // namespace

auto default_texture_cache_directory() -> std::filesystem::path {
    if (auto const *xdg_cache = std::getenv("XDG_CACHE_HOME"); xdg_cache != nullptr && *xdg_cache != '\0') {
        return std::filesystem::path{xdg_cache} / "ktx2";
    }

    if (auto const *home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path{home} / ".cache" / "ktx2";
    }

    return "cache/ktx2";
}

auto load_compressed_texture(std::filesystem::path const &source_path, TextureRole role,
                             std::filesystem::path const &cache_directory)
        -> std::expected<CompressedTexture, TexturePipelineError> {
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

    if (auto cached = try_load_cached(cache_path, role, stem); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, decoding from source '{}'", stem, source_path.string());

    auto decoded = DecodedImage::load_from_file(
            source_path.string(), role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    if (!decoded.has_value()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed,
                                          std::format("failed to decode '{}'", source_path.string())));
    }

    return compress_decoded_image(*decoded, role, cache_path, stem);
}

auto load_compressed_texture_from_encoded_memory(std::span<std::byte const> encoded_bytes, TextureRole role,
                                                  std::string_view cache_key,
                                                  std::filesystem::path const &cache_directory)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    if (encoded_bytes.empty()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "empty encoded image buffer"));
    }

    auto const identity = std::format("encoded-memory|{}", cache_key);
    auto const cache_path = cache_path_for(identity, role, cache_directory, "embedded");

    if (auto cached = try_load_cached(cache_path, role, std::string{cache_key}); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, decoding from embedded memory", cache_key);

    auto decoded = DecodedImage::load_from_memory(
            encoded_bytes, role == TextureRole::colour ? ImageColourSpace::srgb : ImageColourSpace::linear);

    if (!decoded.has_value()) {
        return std::unexpected(make_error(TexturePipelineErrorType::decode_failed, "failed to decode embedded image"));
    }

    return compress_decoded_image(*decoded, role, cache_path, std::string{cache_key});
}

auto load_compressed_texture_from_memory(std::span<std::byte const> rgba_pixels, std::uint32_t width,
                                         std::uint32_t height, TextureRole role, std::string_view cache_key,
                                         std::filesystem::path const &cache_directory)
        -> std::expected<CompressedTexture, TexturePipelineError> {
    if (width == 0 || height == 0 ||
        rgba_pixels.size_bytes() != static_cast<std::size_t>(width) * height * 4) {
        return std::unexpected(
                make_error(TexturePipelineErrorType::decode_failed, "invalid in-memory image dimensions"));
    }

    auto const identity = std::format("memory|{}", cache_key);
    auto const cache_path = cache_path_for(identity, role, cache_directory, "embedded");

    if (auto cached = try_load_cached(cache_path, role, std::string{cache_key}); cached.has_value()) {
        return std::move(*cached);
    }

    debug("texture_pipeline: '{}' not cached, encoding from raw pixel memory", cache_key);

    std::vector<std::byte> base{rgba_pixels.begin(), rgba_pixels.end()};

    return encode_and_transcode(std::move(base), width, height, role, cache_path, std::string{cache_key});
}
