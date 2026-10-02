#include "assets/hdr_image.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>

#include <glm/gtc/packing.hpp>
#include <ktx.h>
#include <stb_image.h>
#include <vulkan/vulkan_core.h>

#include "core/logger.hxx"
#include "gpu/image.hxx"

namespace {

    auto make_error(HdrImageErrorType type, std::string message) -> std::unexpected<HdrImageError> {
        return std::unexpected(HdrImageError{.type = type, .message = std::move(message)});
    }

    [[nodiscard]]
    auto lowercase_extension_of(std::string_view path) -> std::string {
        auto extension = std::filesystem::path{path}.extension().string();

        std::ranges::transform(extension, extension.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        return extension;
    }

    // libktx leaks its memstream when it rejects a buffer with a bad identifier, so screen those out first.
    constexpr std::array<unsigned char, 12> ktx2_identifier{0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, '\r', '\n', 0x1A, '\n'};
    constexpr std::size_t ktx2_header_size = 80;

    struct KtxTextureDeleter {
        auto operator()(ktxTexture2 *texture) const noexcept -> void {
            if (texture != nullptr) {
                ktxTexture_Destroy(ktxTexture(texture));
            }
        }
    };

    using KtxTexturePtr = std::unique_ptr<ktxTexture2, KtxTextureDeleter>;

    // Unsigned small float with `exponent_bits` and `mantissa_bits`, bias 15 (the B10G11R11 channels).
    [[nodiscard]]
    auto decode_small_float(std::uint32_t bits, std::uint32_t mantissa_bits) noexcept -> float {
        auto const mantissa = bits & ((1U << mantissa_bits) - 1U);
        auto const exponent = (bits >> mantissa_bits) & 0x1FU;
        auto const scale = static_cast<float>(1U << mantissa_bits);

        if (exponent == 0) {
            return std::ldexp(static_cast<float>(mantissa) / scale, -14);
        }

        if (exponent == 31) {
            return 0.0F; // inf / NaN
        }

        return std::ldexp(1.0F + (static_cast<float>(mantissa) / scale), static_cast<int>(exponent) - 15);
    }

    [[nodiscard]]
    auto bytes_per_texel(VkFormat format) noexcept -> std::uint32_t {
        switch (format) {
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8;
            case VK_FORMAT_R16G16B16_SFLOAT:
                return 6;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return 16;
            case VK_FORMAT_R32G32B32_SFLOAT:
                return 12;
            case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
                return 4;
            default:
                return 0;
        }
    }

    auto read_texel(VkFormat format, std::byte const *source, float (&rgba)[4]) noexcept -> void {
        rgba[3] = 1.0F;

        switch (format) {
            case VK_FORMAT_R16G16B16A16_SFLOAT:
            case VK_FORMAT_R16G16B16_SFLOAT: {
                auto const channels = format == VK_FORMAT_R16G16B16A16_SFLOAT ? 4U : 3U;
                for (std::uint32_t channel = 0; channel < channels; ++channel) {
                    std::uint16_t half = 0;
                    std::memcpy(&half, source + (channel * 2U), sizeof(half));
                    rgba[channel] = glm::unpackHalf1x16(half);
                }
                break;
            }
            case VK_FORMAT_R32G32B32A32_SFLOAT:
            case VK_FORMAT_R32G32B32_SFLOAT: {
                auto const channels = format == VK_FORMAT_R32G32B32A32_SFLOAT ? 4U : 3U;
                for (std::uint32_t channel = 0; channel < channels; ++channel) {
                    std::memcpy(&rgba[channel], source + (channel * 4U), sizeof(float));
                }
                break;
            }
            case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: {
                std::uint32_t packed = 0;
                std::memcpy(&packed, source, sizeof(packed));

                auto const exponent = static_cast<int>(packed >> 27) - 15 - 9;

                rgba[0] = std::ldexp(static_cast<float>(packed & 0x1FFU), exponent);
                rgba[1] = std::ldexp(static_cast<float>((packed >> 9) & 0x1FFU), exponent);
                rgba[2] = std::ldexp(static_cast<float>((packed >> 18) & 0x1FFU), exponent);
                break;
            }
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32: {
                std::uint32_t packed = 0;
                std::memcpy(&packed, source, sizeof(packed));

                rgba[0] = decode_small_float(packed & 0x7FFU, 6);
                rgba[1] = decode_small_float((packed >> 11) & 0x7FFU, 6);
                rgba[2] = decode_small_float((packed >> 22) & 0x3FFU, 5);
                break;
            }
            default:
                rgba[0] = rgba[1] = rgba[2] = 0.0F;
                break;
        }
    }

} // namespace

auto sanitize_to_half(float value) noexcept -> std::uint16_t {
    if (!std::isfinite(value)) {
        value = 0.0F;
    }

    return glm::packHalf1x16(std::clamp(value, 0.0F, hdr_max_radiance));
}

auto decode_radiance_hdr(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError> {
    if (encoded.empty() || encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return make_error(HdrImageErrorType::invalid_data, "empty or oversized .hdr buffer");
    }

    auto const *data = reinterpret_cast<stbi_uc const *>(encoded.data());
    auto const size = static_cast<int>(encoded.size());

    if (stbi_is_hdr_from_memory(data, size) == 0) {
        return make_error(HdrImageErrorType::unsupported_format, "not a Radiance .hdr file");
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    if (stbi_info_from_memory(data, size, &width, &height, &channels) == 0 || width <= 0 || height <= 0) {
        return make_error(HdrImageErrorType::invalid_data, std::format("could not read .hdr header: {}", stbi_failure_reason()));
    }

    if (std::cmp_greater(width, hdr_equirect_max_width) || std::cmp_greater(height, hdr_equirect_max_height)) {
        return make_error(HdrImageErrorType::too_large, std::format(".hdr is {}x{}, above the {}x{} limit", width, height,
                                                                   hdr_equirect_max_width, hdr_equirect_max_height));
    }

    if (width != 2 * height) {
        return make_error(HdrImageErrorType::invalid_data, std::format(".hdr is {}x{}, an equirect must be 2:1", width, height));
    }

    std::unique_ptr<float, decltype(&stbi_image_free)> decoded{stbi_loadf_from_memory(data, size, &width, &height, &channels, 4),
                                                              &stbi_image_free};

    if (!decoded) {
        return make_error(HdrImageErrorType::invalid_data, std::format("could not decode .hdr: {}", stbi_failure_reason()));
    }

    HdrImage image{
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .layers = 1,
            .pixels = std::vector<std::uint16_t>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U),
    };

    for (std::size_t index = 0; index < image.pixels.size(); ++index) {
        image.pixels[index] = sanitize_to_half(decoded.get()[index]);
    }

    return image;
}

auto decode_ktx2_float(std::span<std::byte const> encoded) -> std::expected<HdrImage, HdrImageError> {
    if (encoded.empty()) {
        return make_error(HdrImageErrorType::invalid_data, "empty .ktx2 buffer");
    }

    if (encoded.size() < ktx2_header_size || std::memcmp(encoded.data(), ktx2_identifier.data(), ktx2_identifier.size()) != 0) {
        return make_error(HdrImageErrorType::invalid_data, "not a .ktx2 file");
    }

    ktxTexture2 *raw = nullptr;

    auto const created = ktxTexture2_CreateFromMemory(reinterpret_cast<ktx_uint8_t const *>(encoded.data()), encoded.size(),
                                                      KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw);

    if (created != KTX_SUCCESS || raw == nullptr) {
        return make_error(HdrImageErrorType::invalid_data, std::format("could not parse .ktx2: {}", ktxErrorString(created)));
    }

    KtxTexturePtr texture{raw};

    if (ktxTexture2_NeedsTranscoding(texture.get())) {
        return make_error(HdrImageErrorType::unsupported_format, ".ktx2 is Basis Universal compressed, not a float format");
    }

    auto const format = static_cast<VkFormat>(texture->vkFormat);
    auto const texel_bytes = bytes_per_texel(format);

    if (texel_bytes == 0) {
        return make_error(HdrImageErrorType::unsupported_format,
                          std::format(".ktx2 VkFormat {} is not an uncompressed float format", texture->vkFormat));
    }

    if (texture->numDimensions != 2 || texture->baseDepth != 1 || texture->numLayers != 1 || texture->isArray) {
        return make_error(HdrImageErrorType::invalid_data, ".ktx2 must be a single 2D image or a cubemap");
    }

    auto const faces = texture->numFaces;

    if (faces != 1 && faces != 6) {
        return make_error(HdrImageErrorType::invalid_data, std::format(".ktx2 has {} faces", faces));
    }

    auto const width = texture->baseWidth;
    auto const height = texture->baseHeight;

    if (faces == 1) {
        if (width > hdr_equirect_max_width || height > hdr_equirect_max_height) {
            return make_error(HdrImageErrorType::too_large, std::format(".ktx2 equirect {}x{} is above the limit", width, height));
        }
        if (height == 0 || width != 2 * height) {
            return make_error(HdrImageErrorType::invalid_data, std::format(".ktx2 equirect {}x{} must be 2:1", width, height));
        }
    } else {
        if (width != height || width == 0 || !std::has_single_bit(width)) {
            return make_error(HdrImageErrorType::invalid_data,
                              std::format(".ktx2 cube faces {}x{} must be square powers of two", width, height));
        }
        if (width > hdr_cube_max_face_size) {
            return make_error(HdrImageErrorType::too_large, std::format(".ktx2 cube face {} is above the limit", width));
        }
    }

    auto const *base = ktxTexture_GetData(ktxTexture(texture.get()));
    auto const data_size = ktxTexture_GetDataSize(ktxTexture(texture.get()));

    HdrImage image{
            .width = width,
            .height = height,
            .layers = faces,
            .pixels = std::vector<std::uint16_t>(static_cast<std::size_t>(width) * height * faces * 4U),
    };

    auto const face_texels = static_cast<std::size_t>(width) * height;

    for (std::uint32_t face = 0; face < faces; ++face) {
        ktx_size_t offset = 0;

        if (ktxTexture_GetImageOffset(ktxTexture(texture.get()), 0, 0, face, &offset) != KTX_SUCCESS ||
            offset + (face_texels * texel_bytes) > data_size) {
            return make_error(HdrImageErrorType::invalid_data, ".ktx2 image data is truncated");
        }

        auto const *source = reinterpret_cast<std::byte const *>(base + offset);

        for (std::size_t texel = 0; texel < face_texels; ++texel) {
            float rgba[4]{};

            read_texel(format, source + (texel * texel_bytes), rgba);

            auto const destination = ((static_cast<std::size_t>(face) * face_texels) + texel) * 4U;

            for (std::size_t channel = 0; channel < 3; ++channel) {
                image.pixels[destination + channel] = sanitize_to_half(rgba[channel]);
            }

            image.pixels[destination + 3] = glm::packHalf1x16(1.0F);
        }
    }

    return image;
}

auto load_hdr_image(std::string_view path) -> std::expected<HdrImage, HdrImageError> {
    auto const extension = lowercase_extension_of(path);

    if (extension == ".exr") {
        auto decoded = DecodedImage::load_from_file(path);

        if (!decoded || decoded->format() != VK_FORMAT_R16G16B16A16_SFLOAT) {
            return make_error(HdrImageErrorType::unreadable, std::format("could not decode '{}'", path));
        }

        if (decoded->height() == 0 || decoded->width() != 2 * decoded->height() ||
            decoded->width() > hdr_equirect_max_width || decoded->height() > hdr_equirect_max_height) {
            return make_error(HdrImageErrorType::invalid_data,
                              std::format("'{}' is {}x{}; an equirect must be 2:1 and at most {}x{}", path, decoded->width(),
                                          decoded->height(), hdr_equirect_max_width, hdr_equirect_max_height));
        }

        HdrImage image{
                .width = decoded->width(),
                .height = decoded->height(),
                .layers = 1,
        };

        auto const bytes = decoded->span();

        image.pixels.resize(bytes.size() / sizeof(std::uint16_t));

        std::memcpy(image.pixels.data(), bytes.data(), bytes.size());

        for (auto &half: image.pixels) {
            half = sanitize_to_half(glm::unpackHalf1x16(half));
        }

        return image;
    }

    if (extension != ".hdr" && extension != ".ktx2") {
        return make_error(HdrImageErrorType::unsupported_format, std::format("'{}': expected .hdr, .exr or .ktx2", path));
    }

    std::ifstream file{std::filesystem::path{path}, std::ios::binary | std::ios::ate};

    if (!file) {
        return make_error(HdrImageErrorType::unreadable, std::format("could not open '{}'", path));
    }

    auto const size = file.tellg();

    if (size <= 0) {
        return make_error(HdrImageErrorType::unreadable, std::format("'{}' is empty", path));
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(size));

    file.seekg(0);
    file.read(reinterpret_cast<char *>(bytes.data()), size);

    if (!file) {
        return make_error(HdrImageErrorType::unreadable, std::format("could not read '{}'", path));
    }

    return extension == ".hdr" ? decode_radiance_hdr(bytes) : decode_ktx2_float(bytes);
}
