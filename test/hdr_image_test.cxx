#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <glm/gtc/packing.hpp>
#include <ktx.h>
#include <vulkan/vulkan_core.h>

#include "assets/hdr_image.hxx"

namespace {

    auto to_bytes(std::string const &text) -> std::vector<std::byte> {
        std::vector<std::byte> bytes(text.size());
        std::memcpy(bytes.data(), text.data(), text.size());
        return bytes;
    }

    auto make_radiance(std::uint32_t width, std::uint32_t height, std::array<std::uint8_t, 4> rgbe) -> std::vector<std::byte> {
        std::string text = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " + std::to_string(height) + " +X " + std::to_string(width) + "\n";

        for (std::uint32_t texel = 0; texel < width * height; ++texel) {
            for (auto const component: rgbe) {
                text.push_back(static_cast<char>(component));
            }
        }

        return to_bytes(text);
    }

    struct KtxBuffer {
        ktx_uint8_t *data = nullptr;
        ktx_size_t size = 0;

        KtxBuffer() = default;
        KtxBuffer(KtxBuffer const &) = delete;
        auto operator=(KtxBuffer const &) -> KtxBuffer & = delete;
        ~KtxBuffer() { std::free(data); }

        [[nodiscard]] auto span() const -> std::span<std::byte const> {
            return {reinterpret_cast<std::byte const *>(data), size};
        }
    };

    auto make_ktx2(VkFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t faces,
                   std::vector<std::byte> const &texel_bytes_per_face, KtxBuffer &out) -> bool {
        ktxTextureCreateInfo info{
                .glInternalformat = 0,
                .vkFormat = static_cast<ktx_uint32_t>(format),
                .pDfd = nullptr,
                .baseWidth = width,
                .baseHeight = height,
                .baseDepth = 1,
                .numDimensions = 2,
                .numLevels = 1,
                .numLayers = 1,
                .numFaces = faces,
                .isArray = KTX_FALSE,
                .generateMipmaps = KTX_FALSE,
        };

        ktxTexture2 *texture = nullptr;

        if (ktxTexture2_Create(&info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) != KTX_SUCCESS) {
            return false;
        }

        for (std::uint32_t face = 0; face < faces; ++face) {
            if (ktxTexture_SetImageFromMemory(ktxTexture(texture), 0, 0, face,
                                              reinterpret_cast<ktx_uint8_t const *>(texel_bytes_per_face.data()),
                                              texel_bytes_per_face.size()) != KTX_SUCCESS) {
                ktxTexture_Destroy(ktxTexture(texture));
                return false;
            }
        }

        auto const written = ktxTexture2_WriteToMemory(texture, &out.data, &out.size) == KTX_SUCCESS;

        ktxTexture_Destroy(ktxTexture(texture));

        return written;
    }

}

TEST_CASE("sanitize_to_half clamps, zeroes non-finite values and keeps small ones") {
    CHECK(glm::unpackHalf1x16(sanitize_to_half(1.0F)) == doctest::Approx(1.0F));
    CHECK(glm::unpackHalf1x16(sanitize_to_half(-3.0F)) == 0.0F);
    CHECK(glm::unpackHalf1x16(sanitize_to_half(1e9F)) == doctest::Approx(65000.0F).epsilon(1e-3));
    CHECK(glm::unpackHalf1x16(sanitize_to_half(std::numeric_limits<float>::infinity())) == 0.0F);
    CHECK(glm::unpackHalf1x16(sanitize_to_half(std::numeric_limits<float>::quiet_NaN())) == 0.0F);
}

TEST_CASE("decode_radiance_hdr reads RGBE to half floats with the clamp") {
    auto const white = decode_radiance_hdr(make_radiance(4, 2, {128, 128, 128, 129}));

    REQUIRE(white.has_value());
    CHECK(white->width == 4);
    CHECK(white->height == 2);
    CHECK(white->layers == 1);
    REQUIRE(white->pixels.size() == 4U * 2U * 4U);
    CHECK(glm::unpackHalf1x16(white->pixels[0]) == doctest::Approx(1.0F));
    CHECK(glm::unpackHalf1x16(white->pixels[3]) == doctest::Approx(1.0F));

    auto const bright = decode_radiance_hdr(make_radiance(4, 2, {255, 0, 0, 160}));

    REQUIRE(bright.has_value());
    CHECK(glm::unpackHalf1x16(bright->pixels[0]) == doctest::Approx(65000.0F).epsilon(1e-3));
    CHECK(glm::unpackHalf1x16(bright->pixels[1]) == 0.0F);
}

TEST_CASE("decode_radiance_hdr refuses non-equirects and garbage") {
    CHECK_FALSE(decode_radiance_hdr(make_radiance(4, 4, {128, 128, 128, 129})).has_value());
    CHECK_FALSE(decode_radiance_hdr(to_bytes("not an image")).has_value());
    CHECK_FALSE(decode_radiance_hdr({}).has_value());
}

TEST_CASE("decode_ktx2_float reads an RGBA32F equirect") {
    std::vector<std::byte> texels(4U * 2U * 4U * sizeof(float));

    for (std::size_t texel = 0; texel < 8; ++texel) {
        std::array<float, 4> const rgba{static_cast<float>(texel), 0.5F, 100000.0F, 0.25F};
        std::memcpy(texels.data() + (texel * sizeof(rgba)), rgba.data(), sizeof(rgba));
    }

    KtxBuffer file;
    REQUIRE(make_ktx2(VK_FORMAT_R32G32B32A32_SFLOAT, 4, 2, 1, texels, file));

    auto const image = decode_ktx2_float(file.span());

    REQUIRE(image.has_value());
    CHECK(image->width == 4);
    CHECK(image->height == 2);
    CHECK(image->layers == 1);
    CHECK(glm::unpackHalf1x16(image->pixels[(3 * 4) + 0]) == doctest::Approx(3.0F));
    CHECK(glm::unpackHalf1x16(image->pixels[(3 * 4) + 1]) == doctest::Approx(0.5F));
    CHECK(glm::unpackHalf1x16(image->pixels[(3 * 4) + 2]) == doctest::Approx(65000.0F).epsilon(1e-3));
    CHECK(glm::unpackHalf1x16(image->pixels[(3 * 4) + 3]) == doctest::Approx(1.0F));
}

TEST_CASE("decode_ktx2_float reads a shared-exponent cubemap") {
    constexpr std::uint32_t exponent = 24;
    constexpr std::uint32_t packed = (exponent << 27) | (8U << 18) | (4U << 9) | 2U;

    std::vector<std::byte> face(4U * 4U * sizeof(std::uint32_t));

    for (std::size_t texel = 0; texel < 16; ++texel) {
        std::memcpy(face.data() + (texel * sizeof(packed)), &packed, sizeof(packed));
    }

    KtxBuffer file;
    REQUIRE(make_ktx2(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 4, 4, 6, face, file));

    auto const image = decode_ktx2_float(file.span());

    REQUIRE(image.has_value());
    CHECK(image->width == 4);
    CHECK(image->height == 4);
    CHECK(image->layers == 6);
    REQUIRE(image->pixels.size() == 4U * 4U * 6U * 4U);

    for (std::size_t layer = 0; layer < 6; ++layer) {
        auto const base = layer * 16U * 4U;
        CHECK(glm::unpackHalf1x16(image->pixels[base + 0]) == doctest::Approx(2.0F));
        CHECK(glm::unpackHalf1x16(image->pixels[base + 1]) == doctest::Approx(4.0F));
        CHECK(glm::unpackHalf1x16(image->pixels[base + 2]) == doctest::Approx(8.0F));
    }
}

TEST_CASE("decode_ktx2_float refuses bad shapes and formats") {
    std::vector<std::byte> eight_bit(4U * 2U * 4U);

    KtxBuffer ldr;
    REQUIRE(make_ktx2(VK_FORMAT_R8G8B8A8_UNORM, 4, 2, 1, eight_bit, ldr));
    CHECK_FALSE(decode_ktx2_float(ldr.span()).has_value());

    std::vector<std::byte> square(4U * 4U * 8U);

    KtxBuffer not_two_to_one;
    REQUIRE(make_ktx2(VK_FORMAT_R16G16B16A16_SFLOAT, 4, 4, 1, square, not_two_to_one));
    CHECK_FALSE(decode_ktx2_float(not_two_to_one.span()).has_value());

    CHECK_FALSE(decode_ktx2_float(to_bytes("KTX garbage")).has_value());
    CHECK_FALSE(decode_ktx2_float({}).has_value());

}

TEST_CASE("the vendored Belfast Sunset cubemap loads through load_hdr_image") {
    auto const image = load_hdr_image(TEST_ASSETS_DIR "/assets/environments/belfast_sunset_puresky_512.ktx2");

    REQUIRE(image.has_value());
    CHECK(image->width == 512);
    CHECK(image->height == 512);
    CHECK(image->layers == 6);
    REQUIRE(image->pixels.size() == 512U * 512U * 6U * 4U);

    double sum = 0.0;
    for (std::size_t texel = 0; texel < image->pixels.size(); texel += 4) {
        auto const luminance = glm::unpackHalf1x16(image->pixels[texel + 1]);

        REQUIRE(luminance >= 0.0F);
        REQUIRE(luminance <= 65000.0F);

        sum += luminance;
    }

    auto const mean = sum / (512.0 * 512.0 * 6.0);
    CHECK(mean > 0.05);
    CHECK(mean < 50.0);
}
