// lathe-env-bake: converts an equirect HDR panorama (.hdr / .exr) into a KTX2 cubemap that the engine loads as an
// environment source (E5B9G9R9, Zstd supercompressed, one mip). Usage:
//
//   lathe-env-bake <input.hdr|.exr> <output.ktx2> [--size 512] [--samples 4] [--preview cross.png]
//
// --preview writes a tonemapped 4x3 cross of the six faces for eyeballing orientation.
// --samples is the per-axis supersampling of each cube texel, which keeps a 4K panorama from aliasing into a 512 face.
// Reuses the engine's cube map conventions (rendering/cube_map.hxx), so what it writes is what the engine reads.

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glm/gtc/packing.hpp>
#include <ktx.h>
#include <stb_image_write.h>

#include "assets/hdr_image.hxx"
#include "core/command_line.hxx"
#include "rendering/cube_map.hxx"

namespace {

    struct Options {
        std::string input;
        std::string output;
        std::uint32_t size = 512;
        std::uint32_t samples = 4;
        std::string preview;
    };

    // The error is what to tell the user.
    [[nodiscard]]
    auto validate(Options const &options) -> std::expected<void, std::string> {
        if (options.size == 0 || !std::has_single_bit(options.size) || options.size > hdr_cube_max_face_size) {
            return std::unexpected(std::format("--size must be a power of two up to {}", hdr_cube_max_face_size));
        }
        if (options.samples == 0 || options.samples > 16) {
            return std::unexpected(std::string{"--samples must be between 1 and 16"});
        }
        return {};
    }

    struct Bilinear {
        HdrImage const &image;

        // u wraps, v clamps.
        [[nodiscard]]
        auto sample(glm::vec2 uv) const -> glm::vec3 {
            auto const x = (uv.x * static_cast<float>(image.width)) - 0.5F;
            auto const y = (uv.y * static_cast<float>(image.height)) - 0.5F;

            auto const x0 = static_cast<std::int32_t>(std::floor(x));
            auto const y0 = static_cast<std::int32_t>(std::floor(y));

            auto const fx = x - static_cast<float>(x0);
            auto const fy = y - static_cast<float>(y0);

            auto const width = static_cast<std::int32_t>(image.width);
            auto const height = static_cast<std::int32_t>(image.height);

            auto const fetch = [&](std::int32_t px, std::int32_t py) {
                px = ((px % width) + width) % width;
                py = std::clamp(py, 0, height - 1);

                auto const *texel = &image.pixels[((static_cast<std::size_t>(py) * image.width) + static_cast<std::size_t>(px)) * 4U];

                return glm::vec3{glm::unpackHalf1x16(texel[0]), glm::unpackHalf1x16(texel[1]), glm::unpackHalf1x16(texel[2])};
            };

            return glm::mix(glm::mix(fetch(x0, y0), fetch(x0 + 1, y0), fx), glm::mix(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), fx), fy);
        }
    };

    // EXT_texture_shared_exponent reference encoder.
    [[nodiscard]]
    auto encode_rgb9e5(glm::vec3 rgb) -> std::uint32_t {
        constexpr int exponent_bias = 15;
        constexpr int mantissa_bits = 9;
        constexpr float max_value = 65408.0F;

        auto const clamped = glm::clamp(glm::vec3{std::isfinite(rgb.x) ? rgb.x : 0.0F, std::isfinite(rgb.y) ? rgb.y : 0.0F,
                                                  std::isfinite(rgb.z) ? rgb.z : 0.0F},
                                        glm::vec3{0.0F}, glm::vec3{max_value});

        auto const largest = std::max({clamped.x, clamped.y, clamped.z});

        auto exponent = std::max(-exponent_bias - 1, static_cast<int>(std::floor(std::log2(std::max(largest, 1e-30F))))) + 1 + exponent_bias;

        auto const scale = [&](int shared_exponent) { return std::exp2(static_cast<float>(shared_exponent - exponent_bias - mantissa_bits)); };

        if (static_cast<int>(std::floor((largest / scale(exponent)) + 0.5F)) == (1 << mantissa_bits)) {
            ++exponent;
        }

        auto const inverse = 1.0F / scale(exponent);

        auto const mantissa = [&](float channel) {
            return static_cast<std::uint32_t>(std::floor((channel * inverse) + 0.5F));
        };

        return (static_cast<std::uint32_t>(exponent) << 27) | (mantissa(clamped.z) << 18) | (mantissa(clamped.y) << 9) | mantissa(clamped.x);
    }

    [[nodiscard]]
    auto decode_rgb9e5(std::uint32_t packed) -> glm::vec3 {
        auto const scale = std::exp2(static_cast<float>(static_cast<int>(packed >> 27) - 15 - 9));

        return glm::vec3{static_cast<float>(packed & 0x1FFU), static_cast<float>((packed >> 9) & 0x1FFU),
                         static_cast<float>((packed >> 18) & 0x1FFU)} *
               scale;
    }

    // Faces laid out as the usual cross: +Y on top, -Y below, and -X +Z +X -Z across the middle row.
    auto write_preview(std::string_view path, std::vector<std::uint32_t> const &texels, std::uint32_t size) -> void {
        constexpr std::uint32_t columns = 4;
        constexpr std::uint32_t rows = 3;

        struct Placement {
            std::uint32_t face, column, row;
        };

        constexpr std::array<Placement, 6> placements{{{2, 1, 0}, {1, 0, 1}, {4, 1, 1}, {0, 2, 1}, {5, 3, 1}, {3, 1, 2}}};

        std::vector<std::uint8_t> image(static_cast<std::size_t>(columns) * size * rows * size * 3U, 0);

        for (auto const &placement: placements) {
            for (std::uint32_t y = 0; y < size; ++y) {
                for (std::uint32_t x = 0; x < size; ++x) {
                    auto colour = decode_rgb9e5(texels[(((static_cast<std::size_t>(placement.face) * size) + y) * size) + x]);

                    colour = glm::pow(colour / (glm::vec3{1.0F} + colour), glm::vec3{1.0F / 2.2F});

                    auto const out_x = (placement.column * size) + x;
                    auto const out_y = (placement.row * size) + y;
                    auto const offset = ((static_cast<std::size_t>(out_y) * columns * size) + out_x) * 3U;

                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        image[offset + channel] = static_cast<std::uint8_t>(std::clamp(colour[static_cast<int>(channel)], 0.0F, 1.0F) * 255.0F + 0.5F);
                    }
                }
            }
        }

        stbi_write_png(std::string{path}.c_str(), static_cast<int>(columns * size), static_cast<int>(rows * size), 3, image.data(),
                       static_cast<int>(columns * size * 3U));
    }

} // namespace

auto main(int argc, char **argv) -> int {
    Options parsed;
    CommandLine cli{"lathe-env-bake", "Converts an equirect HDR panorama (.hdr / .exr) into a KTX2 cubemap environment."};
    cli.positional("input", "Equirect panorama, .hdr or .exr", parsed.input);
    cli.positional("output", "Cubemap to write, .ktx2", parsed.output);

    auto bake = cli.group("Bake");
    bake.value("--size", "N", "Cube face size, a power of two (512)", parsed.size);
    bake.value("--samples", "N", "Per-axis supersampling of each cube texel, 1 to 16 (4)", parsed.samples);
    bake.value("--preview", "cross.png", "Also write a tonemapped 4x3 cross of the six faces, to eyeball orientation",
               parsed.preview);

    auto const outcome = cli.parse(argc, argv);

    if (!outcome) {
        std::fprintf(stderr, "lathe-env-bake: %s\nTry --help.\n", outcome.error().c_str());
        return 2;
    }
    if (*outcome == CommandLine::Outcome::help) {
        std::fputs(cli.help_text().c_str(), stdout);
        return 0;
    }
    if (auto const valid = validate(parsed); !valid) {
        std::fprintf(stderr, "lathe-env-bake: %s\n", valid.error().c_str());
        return 2;
    }

    auto const *const options = &parsed;

    auto const source = load_hdr_image(options->input);

    if (!source) {
        std::fprintf(stderr, "lathe-env-bake: %s\n", source.error().message.c_str());
        return 1;
    }

    if (source->layers != 1) {
        std::fputs("lathe-env-bake: input must be an equirect panorama\n", stderr);
        return 1;
    }

    auto const size = options->size;
    auto const samples = options->samples;

    std::vector<std::uint32_t> texels(static_cast<std::size_t>(size) * size * cube_face_count);

    Bilinear const bilinear{*source};

    auto const sample_weight = 1.0F / static_cast<float>(samples * samples);

    for (std::uint32_t face = 0; face < cube_face_count; ++face) {
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                glm::vec3 sum{0.0F};

                for (std::uint32_t sy = 0; sy < samples; ++sy) {
                    for (std::uint32_t sx = 0; sx < samples; ++sx) {
                        glm::vec2 const uv{
                                (static_cast<float>(x) + ((static_cast<float>(sx) + 0.5F) / static_cast<float>(samples))) / static_cast<float>(size),
                                (static_cast<float>(y) + ((static_cast<float>(sy) + 0.5F) / static_cast<float>(samples))) / static_cast<float>(size),
                        };

                        sum += bilinear.sample(direction_to_equirect_uv(cube_texel_direction(face, uv)));
                    }
                }

                texels[(((static_cast<std::size_t>(face) * size) + y) * size) + x] = encode_rgb9e5(sum * sample_weight);
            }
        }
    }

    if (!options->preview.empty()) {
        write_preview(options->preview, texels, size);
    }

    ktxTextureCreateInfo info{
            .glInternalformat = 0,
            .vkFormat = 123, // VK_FORMAT_E5B9G9R9_UFLOAT_PACK32
            .pDfd = nullptr,
            .baseWidth = size,
            .baseHeight = size,
            .baseDepth = 1,
            .numDimensions = 2,
            .numLevels = 1,
            .numLayers = 1,
            .numFaces = cube_face_count,
            .isArray = KTX_FALSE,
            .generateMipmaps = KTX_FALSE,
    };

    ktxTexture2 *texture = nullptr;

    if (ktxTexture2_Create(&info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) != KTX_SUCCESS) {
        std::fputs("lathe-env-bake: ktxTexture2_Create failed\n", stderr);
        return 1;
    }

    auto const face_bytes = static_cast<std::size_t>(size) * size * sizeof(std::uint32_t);

    for (std::uint32_t face = 0; face < cube_face_count; ++face) {
        auto const *data = reinterpret_cast<ktx_uint8_t const *>(texels.data() + (static_cast<std::size_t>(face) * size * size));

        if (ktxTexture_SetImageFromMemory(ktxTexture(texture), 0, 0, face, data, face_bytes) != KTX_SUCCESS) {
            std::fputs("lathe-env-bake: could not set face data\n", stderr);
            ktxTexture_Destroy(ktxTexture(texture));
            return 1;
        }
    }

    if (ktxTexture2_DeflateZstd(texture, 22) != KTX_SUCCESS) {
        std::fputs("lathe-env-bake: Zstd supercompression failed\n", stderr);
        ktxTexture_Destroy(ktxTexture(texture));
        return 1;
    }

    auto const written = ktxTexture_WriteToNamedFile(ktxTexture(texture), std::string{options->output}.c_str());

    ktxTexture_Destroy(ktxTexture(texture));

    if (written != KTX_SUCCESS) {
        std::fputs("lathe-env-bake: could not write the output file\n", stderr);
        return 1;
    }

    std::printf("wrote %s (%ux%u x 6 faces, %u x %u samples)\n", std::string{options->output}.c_str(), size, size, samples, samples);

    return 0;
}
