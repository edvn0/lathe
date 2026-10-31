#include <doctest/doctest.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu/image.hxx"
#include "gpu/png_decoder.hxx"

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {

    using Bytes = std::vector<std::uint8_t>;

    auto as_span(Bytes const &bytes) -> std::span<std::byte const> { return std::as_bytes(std::span{bytes}); }

    // ---- building PNG files by hand --------------------------------------------------------------------------

    auto crc32(std::uint8_t const *data, std::size_t size) -> std::uint32_t {
        static auto const table = [] {
            std::array<std::uint32_t, 256> entries{};

            for (std::uint32_t n = 0; n < 256; ++n) {
                auto c = n;

                for (auto k = 0; k < 8; ++k) {
                    c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1) : c >> 1;
                }

                entries[n] = c;
            }

            return entries;
        }();

        auto crc = 0xFFFFFFFFU;

        for (std::size_t i = 0; i < size; ++i) {
            crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8);
        }

        return crc ^ 0xFFFFFFFFU;
    }

    auto put_u32(Bytes &out, std::uint32_t value) -> void {
        out.push_back(static_cast<std::uint8_t>(value >> 24));
        out.push_back(static_cast<std::uint8_t>(value >> 16));
        out.push_back(static_cast<std::uint8_t>(value >> 8));
        out.push_back(static_cast<std::uint8_t>(value));
    }

    auto chunk(std::string_view type, Bytes const &data) -> Bytes {
        Bytes out;
        put_u32(out, static_cast<std::uint32_t>(data.size()));
        out.insert(out.end(), type.begin(), type.end());
        out.insert(out.end(), data.begin(), data.end());
        put_u32(out, crc32(out.data() + 4, out.size() - 4));

        return out;
    }

    auto adler32(Bytes const &data) -> std::uint32_t {
        std::uint32_t a = 1;
        std::uint32_t b = 0;

        for (auto const byte: data) {
            a = (a + byte) % 65521U;
            b = (b + a) % 65521U;
        }

        return (b << 16) | a;
    }

    // A zlib stream of stored (uncompressed) blocks.
    auto zlib_stored(Bytes const &raw) -> Bytes {
        Bytes out{0x78, 0x01};
        auto offset = std::size_t{0};

        do {
            auto const size = std::min<std::size_t>(raw.size() - offset, 65535);
            auto const last = offset + size == raw.size();

            out.push_back(last ? 1 : 0);
            out.push_back(static_cast<std::uint8_t>(size & 0xFF));
            out.push_back(static_cast<std::uint8_t>(size >> 8));
            out.push_back(static_cast<std::uint8_t>(~size & 0xFF));
            out.push_back(static_cast<std::uint8_t>((~size >> 8) & 0xFF));
            out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                       raw.begin() + static_cast<std::ptrdiff_t>(offset + size));
            offset += size;
        } while (offset < raw.size());

        put_u32(out, adler32(raw));

        return out;
    }

    struct PngSpec {
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint8_t bit_depth = 8;
        std::uint8_t colour = 6;
        std::uint8_t interlace = 0;
        Bytes raw; // filter bytes and filtered scanlines, as the zlib stream will carry them
        Bytes palette; // PLTE data, if any
        Bytes transparency; // tRNS data, if any
        std::vector<Bytes> extra_chunks_before_data;
    };

    auto make_png(PngSpec const &spec) -> Bytes {
        Bytes out{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
        Bytes header;
        put_u32(header, spec.width);
        put_u32(header, spec.height);
        header.insert(header.end(), {spec.bit_depth, spec.colour, 0, 0, spec.interlace});

        auto const append = [&out](Bytes const &piece) { out.insert(out.end(), piece.begin(), piece.end()); };

        append(chunk("IHDR", header));

        if (!spec.palette.empty()) {
            append(chunk("PLTE", spec.palette));
        }

        if (!spec.transparency.empty()) {
            append(chunk("tRNS", spec.transparency));
        }

        for (auto const &extra: spec.extra_chunks_before_data) {
            append(extra);
        }

        append(chunk("IDAT", zlib_stored(spec.raw)));
        append(chunk("IEND", {}));

        return out;
    }

    // ---- stb_image as the reference --------------------------------------------------------------------------

    struct Reference {
        std::vector<std::uint8_t> pixels;
        int width = 0;
        int height = 0;
        bool ok = false;
    };

    auto decode_with_stb(Bytes const &png) -> Reference {
        Reference reference;
        int channels = 0;
        auto *pixels = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &reference.width,
                                             &reference.height, &channels, STBI_rgb_alpha);

        if (pixels != nullptr) {
            reference.ok = true;
            reference.pixels.assign(pixels, pixels + static_cast<std::size_t>(reference.width) * reference.height * 4);
            stbi_image_free(pixels);
        }

        return reference;
    }

    auto same_as_stb(Bytes const &png, bool allow_simd) -> bool {
        auto const reference = decode_with_stb(png);
        auto const decoded = decode_png_fast(as_span(png), PngDecodeOptions{.allow_simd = allow_simd});

        if (!reference.ok || !decoded.has_value() || static_cast<int>(decoded->width) != reference.width ||
            static_cast<int>(decoded->height) != reference.height ||
            decoded->pixels.size() != reference.pixels.size()) {
            return false;
        }

        return std::memcmp(decoded->pixels.data(), reference.pixels.data(), reference.pixels.size()) == 0;
    }

    // ---- images --------------------------------------------------------------------------------------------------

    // Smooth with noise, so every filter has something to predict and something to get wrong.
    auto make_pixels(std::uint32_t width, std::uint32_t height, int channels, std::uint32_t seed) -> Bytes {
        std::mt19937 random{seed};
        Bytes pixels(static_cast<std::size_t>(width) * height * static_cast<std::size_t>(channels));

        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                for (auto channel = 0; channel < channels; ++channel) {
                    auto const smooth = (x * 3 + y * 5 + static_cast<std::uint32_t>(channel) * 40) % 256;
                    auto const noise = random() % 24;

                    pixels[(static_cast<std::size_t>(y) * width + x) * static_cast<std::size_t>(channels) +
                           static_cast<std::size_t>(channel)] = static_cast<std::uint8_t>((smooth + noise) & 0xFF);
                }
            }
        }

        return pixels;
    }

    // stb_image_write's PNG encoder, with its filter choice forced (-1: its own heuristic per row).
    auto write_png(Bytes const &pixels, std::uint32_t width, std::uint32_t height, int channels, int forced_filter)
            -> Bytes {
        auto const saved = stbi_write_force_png_filter;
        stbi_write_force_png_filter = forced_filter;

        Bytes png;
        auto const append = [](void *context, void *data, int size) {
            auto const *bytes = static_cast<std::uint8_t const *>(data);
            static_cast<Bytes *>(context)->insert(static_cast<Bytes *>(context)->end(), bytes, bytes + size);
        };

        auto const written = stbi_write_png_to_func(append, &png, static_cast<int>(width), static_cast<int>(height),
                                                    channels, pixels.data(), static_cast<int>(width) * channels);

        stbi_write_force_png_filter = saved;

        REQUIRE(written != 0);

        return png;
    }

    // Splits the IDAT chunk of 'png' into pieces of at most 'piece' bytes.
    auto split_idat(Bytes const &png, std::size_t piece) -> Bytes {
        Bytes out{png.begin(), png.begin() + 8};
        auto offset = std::size_t{8};

        while (offset + 12 <= png.size()) {
            auto const length = (static_cast<std::size_t>(png[offset]) << 24) |
                                (static_cast<std::size_t>(png[offset + 1]) << 16) |
                                (static_cast<std::size_t>(png[offset + 2]) << 8) | png[offset + 3];
            std::string const type{reinterpret_cast<char const *>(png.data() + offset + 4), 4};
            Bytes const data{png.begin() + static_cast<std::ptrdiff_t>(offset + 8),
                             png.begin() + static_cast<std::ptrdiff_t>(offset + 8 + length)};

            if (type == "IDAT") {
                for (std::size_t start = 0; start < data.size(); start += piece) {
                    auto const end = std::min(data.size(), start + piece);
                    auto const part = chunk("IDAT", Bytes{data.begin() + static_cast<std::ptrdiff_t>(start),
                                                          data.begin() + static_cast<std::ptrdiff_t>(end)});
                    out.insert(out.end(), part.begin(), part.end());
                }
            } else {
                auto const copy = chunk(type, data);
                out.insert(out.end(), copy.begin(), copy.end());
            }

            offset += 12 + length;
        }

        return out;
    }

    // Puts a chunk right before the first IDAT.
    auto insert_before_data(Bytes const &png, Bytes const &extra) -> Bytes {
        auto offset = std::size_t{8};

        while (offset + 12 <= png.size()) {
            auto const length = (static_cast<std::size_t>(png[offset]) << 24) |
                                (static_cast<std::size_t>(png[offset + 1]) << 16) |
                                (static_cast<std::size_t>(png[offset + 2]) << 8) | png[offset + 3];

            if (std::memcmp(png.data() + offset + 4, "IDAT", 4) == 0) {
                Bytes out{png.begin(), png.begin() + static_cast<std::ptrdiff_t>(offset)};
                out.insert(out.end(), extra.begin(), extra.end());
                out.insert(out.end(), png.begin() + static_cast<std::ptrdiff_t>(offset), png.end());

                return out;
            }

            offset += 12 + length;
        }

        return png;
    }

} // namespace

TEST_SUITE("unit") {

    TEST_CASE("png fast decoder: every colour type and filter matches stb_image, with and without SIMD") {
        // Widths that are not multiples of 4 or 16 test the SIMD tails; 1x1 and a single row test the edges.
        struct Size {
            std::uint32_t width;
            std::uint32_t height;
        };

        for (auto const size:
             {Size{1, 1}, Size{3, 5}, Size{16, 1}, Size{17, 9}, Size{33, 7}, Size{64, 40}, Size{101, 3}}) {
            for (auto channels = 1; channels <= 4; ++channels) {
                auto const pixels =
                        make_pixels(size.width, size.height, channels, 1234U + static_cast<unsigned>(channels));

                // 0..4 force each filter on every row; -1 lets the encoder mix them.
                for (auto filter = -1; filter <= 4; ++filter) {
                    auto const png = write_png(pixels, size.width, size.height, channels, filter);

                    CAPTURE(size.width);
                    CAPTURE(size.height);
                    CAPTURE(channels);
                    CAPTURE(filter);

                    CHECK(same_as_stb(png, true));
                    CHECK(same_as_stb(png, false));
                }
            }
        }
    }

    TEST_CASE("png fast decoder: random noise matches stb_image") {
        std::mt19937 random{99};

        for (auto round = 0; round < 12; ++round) {
            auto const width = static_cast<std::uint32_t>(1U + random() % 70);
            auto const height = static_cast<std::uint32_t>(1U + random() % 40);
            auto const channels = 1 + static_cast<int>(random() % 4);
            Bytes pixels(static_cast<std::size_t>(width) * height * static_cast<std::size_t>(channels));

            for (auto &byte: pixels) {
                byte = static_cast<std::uint8_t>(random());
            }

            auto const png = write_png(pixels, width, height, channels, -1);

            CHECK(same_as_stb(png, true));
            CHECK(same_as_stb(png, false));
        }
    }

    TEST_CASE("png fast decoder: an image in many IDAT chunks decodes the same") {
        auto const pixels = make_pixels(48, 32, 4, 7);
        auto const png = write_png(pixels, 48, 32, 4, -1);

        for (auto const piece: {std::size_t{1}, std::size_t{7}, std::size_t{100}, std::size_t{4096}}) {
            CHECK(same_as_stb(split_idat(png, piece), true));
        }
    }

    TEST_CASE("png fast decoder: palette images, with and without transparency, match stb_image") {
        // 4x3 pixels, one byte each, using filter 0 on every row.
        Bytes raw;
        std::array<std::uint8_t, 12> indices{0, 1, 2, 3, 3, 2, 1, 0, 1, 1, 2, 2};

        for (auto row = 0; row < 3; ++row) {
            raw.push_back(0);
            raw.insert(raw.end(), indices.begin() + row * 4, indices.begin() + row * 4 + 4);
        }

        PngSpec spec{.width = 4, .height = 3, .colour = 3, .raw = raw};
        spec.palette = {255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30};

        CHECK(same_as_stb(make_png(spec), true));

        // Alpha for only the first two entries: the rest are opaque.
        spec.transparency = {0, 128};

        CHECK(same_as_stb(make_png(spec), true));
        CHECK(same_as_stb(make_png(spec), false));
    }

    TEST_CASE("png fast decoder: a palette index past the palette is left to stb_image") {
        PngSpec spec{.width = 2, .height = 1, .colour = 3, .raw = {0, 0, 5}};
        spec.palette = {1, 2, 3, 4, 5, 6};

        CHECK_FALSE(decode_png_fast(as_span(make_png(spec))).has_value());
    }

    TEST_CASE("png fast decoder: a colour key makes matching pixels transparent, as in stb_image") {
        auto const grey = make_pixels(9, 6, 1, 3);
        auto const grey_png = write_png(grey, 9, 6, 1, -1);

        // Key on a grey value the image really holds.
        Bytes grey_key;
        grey_key.push_back(0);
        grey_key.push_back(grey[4]);

        CHECK(same_as_stb(insert_before_data(grey_png, chunk("tRNS", grey_key)), true));

        auto const rgb = make_pixels(9, 6, 3, 4);
        auto const rgb_png = write_png(rgb, 9, 6, 3, -1);
        Bytes rgb_key{0, rgb[12], 0, rgb[13], 0, rgb[14]};

        CHECK(same_as_stb(insert_before_data(rgb_png, chunk("tRNS", rgb_key)), true));
        CHECK(same_as_stb(insert_before_data(rgb_png, chunk("tRNS", rgb_key)), false));
    }

    TEST_CASE("png fast decoder: files it does not handle are declined, not mis-decoded") {
        auto const rgba = make_pixels(8, 8, 4, 5);
        auto const png = write_png(rgba, 8, 8, 4, -1);

        // 16 bits per channel.
        CHECK_FALSE(
                decode_png_fast(
                        as_span(make_png(PngSpec{
                                .width = 2, .height = 1, .bit_depth = 16, .colour = 6, .raw = Bytes(1 + 2 * 8, 0)})))
                        .has_value());

        // One bit per pixel.
        CHECK_FALSE(decode_png_fast(as_span(make_png(PngSpec{
                                            .width = 8, .height = 1, .bit_depth = 1, .colour = 0, .raw = Bytes(2, 0)})))
                            .has_value());

        // Interlaced.
        CHECK_FALSE(decode_png_fast(as_span(make_png(PngSpec{.width = 2,
                                                             .height = 1,
                                                             .bit_depth = 8,
                                                             .colour = 6,
                                                             .interlace = 1,
                                                             .raw = Bytes(1 + 8, 0)})))
                            .has_value());

        // Not a PNG at all, and empty.
        CHECK_FALSE(decode_png_fast(as_span(Bytes{1, 2, 3, 4, 5, 6, 7, 8, 9})).has_value());
        CHECK_FALSE(decode_png_fast({}).has_value());
        CHECK(has_png_signature(as_span(png)));
        CHECK_FALSE(has_png_signature(as_span(Bytes{0x89, 'P', 'N'})));
    }

    TEST_CASE("png fast decoder: damaged files are declined") {
        auto const pixels = make_pixels(20, 12, 4, 9);
        auto const png = write_png(pixels, 20, 12, 4, -1);

        // A filter byte PNG does not define.
        Bytes bad_filter(1 + 4, 0);
        bad_filter[0] = 9;
        CHECK_FALSE(decode_png_fast(as_span(make_png(PngSpec{.width = 1, .height = 1, .colour = 6, .raw = bad_filter})))
                            .has_value());

        // Fewer rows than the header says.
        CHECK_FALSE(decode_png_fast(
                            as_span(make_png(PngSpec{.width = 4, .height = 4, .colour = 6, .raw = Bytes(1 + 16, 0)})))
                            .has_value());

        // More data than the header says.
        CHECK_FALSE(decode_png_fast(as_span(make_png(
                                            PngSpec{.width = 1, .height = 1, .colour = 6, .raw = Bytes(1 + 4 + 5, 0)})))
                            .has_value());

        // Dimensions that are zero or absurd.
        CHECK_FALSE(
                decode_png_fast(as_span(make_png(PngSpec{.width = 0, .height = 1, .raw = Bytes(1, 0)}))).has_value());
        CHECK_FALSE(
                decode_png_fast(as_span(make_png(PngSpec{.width = (1U << 24) + 1, .height = 1, .raw = Bytes(1, 0)})))
                        .has_value());
        CHECK_FALSE(
                decode_png_fast(as_span(make_png(PngSpec{.width = 1U << 24, .height = 1U << 24, .raw = Bytes(1, 0)})))
                        .has_value());

        // Cut short at every length.
        for (std::size_t length = 0; length < png.size(); length += 3) {
            Bytes const cut{png.begin(), png.begin() + static_cast<std::ptrdiff_t>(length)};

            CHECK_FALSE(decode_png_fast(as_span(cut)).has_value());
        }
    }

    TEST_CASE("png fast decoder: flipped bytes never crash and never give a wrong-sized image") {
        auto const pixels = make_pixels(40, 30, 4, 11);
        auto const png = write_png(pixels, 40, 30, 4, -1);
        std::mt19937 random{2024};

        for (auto round = 0; round < 400; ++round) {
            auto damaged = png;
            auto const flips = 1 + random() % 4;

            for (std::uint32_t flip = 0; flip < flips; ++flip) {
                damaged[random() % damaged.size()] ^= static_cast<std::uint8_t>(1U << (random() % 8));
            }

            for (auto const simd: {true, false}) {
                auto const decoded = decode_png_fast(as_span(damaged), PngDecodeOptions{.allow_simd = simd});

                if (decoded.has_value()) {
                    CHECK(decoded->pixels.size() == static_cast<std::size_t>(decoded->width) * decoded->height * 4);
                    CHECK(decoded->width > 0);
                    CHECK(decoded->height > 0);
                }
            }
        }
    }

    TEST_CASE("DecodedImage: PNGs decode through the fast path to the pixels stb_image gives") {
        auto const pixels = make_pixels(37, 23, 4, 21);
        auto const png = write_png(pixels, 37, 23, 4, -1);
        auto const reference = decode_with_stb(png);

        auto const decoded = DecodedImage::load_from_memory(as_span(png), ImageColourSpace::srgb);

        REQUIRE(decoded.has_value());
        CHECK(decoded->format() == VK_FORMAT_R8G8B8A8_SRGB);
        REQUIRE(decoded->span().size() == reference.pixels.size());
        CHECK(std::memcmp(decoded->span().data(), reference.pixels.data(), reference.pixels.size()) == 0);

        // A 16-bit PNG is declined by the fast path and still decodes, through stb_image.
        Bytes sixteen(1 + 2 * 8, 0);
        auto const png16 = make_png(PngSpec{.width = 2, .height = 1, .bit_depth = 16, .colour = 6, .raw = sixteen});
        auto const via_stb = DecodedImage::load_from_memory(as_span(png16), ImageColourSpace::linear);
        CHECK(via_stb.has_value());
    }

    TEST_CASE("DecodedImage: a PNG file on disk decodes to the pixels stb_image gives") {
        auto const path = std::string{TEST_ASSETS_DIR} + "/assets/textures/terrain/terrain_roughness.png";
        int width = 0;
        int height = 0;
        int channels = 0;
        auto *reference = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

        if (reference == nullptr) {
            // No PNG among the test assets: the in-memory tests above cover the decoder.
            return;
        }

        auto const decoded = DecodedImage::load_from_file(path, ImageColourSpace::linear);

        REQUIRE(decoded.has_value());
        REQUIRE(decoded->width() == static_cast<std::uint32_t>(width));
        REQUIRE(decoded->height() == static_cast<std::uint32_t>(height));
        CHECK(std::memcmp(decoded->span().data(), reference, static_cast<std::size_t>(width) * height * 4) == 0);

        stbi_image_free(reference);
    }
}
