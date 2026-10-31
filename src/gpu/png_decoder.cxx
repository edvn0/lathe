#include "gpu/png_decoder.hxx"

#include <libdeflate.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {

    constexpr std::array<std::uint8_t, 8> png_signature{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

    // stb_image refuses anything wider or taller; so do we, and the byte counts below cannot overflow with it.
    constexpr std::uint32_t max_dimension = 1U << 24;
    // The size of the decoded image, of the scratch, or of the compressed data past which we leave it to stb_image.
    constexpr std::uint64_t max_bytes = 1ULL << 31;

    enum class ColourType : std::uint8_t {
        grey = 0,
        rgb = 2,
        palette = 3,
        grey_alpha = 4,
        rgba = 6,
    };

    struct Header {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        ColourType colour = ColourType::rgba;
    };

    // What the file asked for that the decoding needs, gathered from its chunks.
    struct Chunks {
        Header header;
        std::array<std::uint8_t, 768> palette{};
        std::size_t palette_entries = 0;
        std::array<std::uint8_t, 256> palette_alpha{};
        std::size_t palette_alpha_entries = 0;
        bool has_colour_key = false;
        std::array<std::uint8_t, 3> colour_key{};
        std::vector<std::span<std::uint8_t const>> data;
        std::size_t data_bytes = 0;
    };

    [[nodiscard]] auto read_u32(std::uint8_t const *bytes) noexcept -> std::uint32_t {
        return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
               (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
    }

    [[nodiscard]] auto channels_of(ColourType colour) noexcept -> std::size_t {
        switch (colour) {
            case ColourType::grey:
            case ColourType::palette:
                return 1;
            case ColourType::grey_alpha:
                return 2;
            case ColourType::rgb:
                return 3;
            case ColourType::rgba:
                return 4;
        }

        return 0;
    }

    // Reads the chunks the decoder needs. False for anything it does not handle or that is malformed, so the caller can
    // leave the file to stb_image.
    [[nodiscard]] auto parse_chunks(std::span<std::uint8_t const> file, Chunks &chunks) -> bool {
        if (file.size() < png_signature.size() ||
            !std::equal(png_signature.begin(), png_signature.end(), file.begin())) {
            return false;
        }

        auto offset = png_signature.size();
        auto seen_header = false;
        auto seen_data = false;
        auto data_ended = false;
        auto seen_end = false;

        while (!seen_end) {
            // length, type, then the data and the CRC (which stb_image does not check either).
            if (file.size() - offset < 12) {
                return false;
            }

            auto const length = static_cast<std::size_t>(read_u32(file.data() + offset));
            auto const type = file.subspan(offset + 4, 4);

            if (length > file.size() - offset - 12) {
                return false;
            }

            auto const data = file.subspan(offset + 8, length);
            offset += 12 + length;

            auto const is = [&type](char const(&name)[5]) { return std::memcmp(type.data(), name, 4) == 0; };

            if (!seen_header) {
                if (!is("IHDR") || length != 13) {
                    return false;
                }

                chunks.header.width = read_u32(data.data());
                chunks.header.height = read_u32(data.data() + 4);

                auto const bit_depth = data[8];
                auto const colour = data[9];
                auto const compression = data[10];
                auto const filter = data[11];
                auto const interlace = data[12];

                if (chunks.header.width == 0 || chunks.header.height == 0 || chunks.header.width > max_dimension ||
                    chunks.header.height > max_dimension || bit_depth != 8 || compression != 0 || filter != 0 ||
                    interlace != 0) {
                    return false;
                }

                if (colour != 0 && colour != 2 && colour != 3 && colour != 4 && colour != 6) {
                    return false;
                }

                chunks.header.colour = static_cast<ColourType>(colour);
                seen_header = true;

                continue;
            }

            if (is("IDAT")) {
                // All the IDAT chunks are one run: a gap makes it stb_image's problem.
                if (data_ended) {
                    return false;
                }

                seen_data = true;
                chunks.data.push_back(data);
                chunks.data_bytes += length;

                if (chunks.data_bytes > max_bytes) {
                    return false;
                }

                continue;
            }

            if (seen_data) {
                data_ended = true;
            }

            if (is("IEND")) {
                seen_end = true;
            } else if (is("PLTE")) {
                if (seen_data || length % 3 != 0 || length > chunks.palette.size() || chunks.palette_entries != 0) {
                    return false;
                }

                std::copy(data.begin(), data.end(), chunks.palette.begin());
                chunks.palette_entries = length / 3;
            } else if (is("tRNS")) {
                if (seen_data) {
                    return false;
                }

                switch (chunks.header.colour) {
                    case ColourType::palette:
                        if (length > chunks.palette_alpha.size()) {
                            return false;
                        }

                        std::copy(data.begin(), data.end(), chunks.palette_alpha.begin());
                        chunks.palette_alpha_entries = length;
                        break;
                    case ColourType::grey:
                        if (length != 2) {
                            return false;
                        }

                        // 16-bit samples; for 8-bit images the value is in the low byte (as stb_image reads it).
                        chunks.colour_key[0] = data[1];
                        chunks.has_colour_key = true;
                        break;
                    case ColourType::rgb:
                        if (length != 6) {
                            return false;
                        }

                        chunks.colour_key = {data[1], data[3], data[5]};
                        chunks.has_colour_key = true;
                        break;
                    default:
                        return false;
                }
            } else if ((type[0] & 0x20U) == 0) {
                // An unknown critical chunk.
                return false;
            }
        }

        return seen_data && !(chunks.header.colour == ColourType::palette && chunks.palette_entries == 0);
    }

    // ---- unfiltering --------------------------------------------------------------------------------------------

    [[nodiscard]] auto paeth_predictor(int a, int b, int c) noexcept -> int {
        auto const p = a + b - c;
        auto const pa = std::abs(p - a);
        auto const pb = std::abs(p - b);
        auto const pc = std::abs(p - c);

        if (pa <= pb && pa <= pc) {
            return a;
        }

        return pb <= pc ? b : c;
    }

    // Rows are unfiltered into place: row r's output lands at r * stride, its filtered bytes sit at
    // r * (stride + 1) + 1, which is never lower, so each byte is read before the write that could reach it. 'previous'
    // is the output of the row above (or zeros), 'source' may overlap 'destination' from above.
    auto unfilter_row_portable(std::uint8_t filter, std::uint8_t const *source, std::uint8_t *destination,
                               std::uint8_t const *previous, std::size_t stride, std::size_t bpp) noexcept -> void {
        switch (filter) {
            case 0:
                std::memmove(destination, source, stride);
                break;
            case 1:
                for (std::size_t i = 0; i < stride; ++i) {
                    auto const left = i >= bpp ? destination[i - bpp] : 0;
                    destination[i] = static_cast<std::uint8_t>(source[i] + left);
                }
                break;
            case 2:
                for (std::size_t i = 0; i < stride; ++i) {
                    destination[i] = static_cast<std::uint8_t>(source[i] + previous[i]);
                }
                break;
            case 3:
                for (std::size_t i = 0; i < stride; ++i) {
                    auto const left = i >= bpp ? destination[i - bpp] : 0;
                    destination[i] = static_cast<std::uint8_t>(source[i] + ((left + previous[i]) >> 1));
                }
                break;
            default:
                for (std::size_t i = 0; i < stride; ++i) {
                    auto const left = i >= bpp ? destination[i - bpp] : 0;
                    auto const up_left = i >= bpp ? previous[i - bpp] : 0;
                    destination[i] = static_cast<std::uint8_t>(source[i] + paeth_predictor(left, previous[i], up_left));
                }
                break;
        }
    }

#if defined(__SSE2__)

    // One pixel of Bpp (3 or 4) bytes in the low lanes of a register, zero extended to 16 bits per lane. 'Wide' reads
    // four bytes whatever Bpp is: the extra byte of a three-byte pixel lands in lane 3, which nothing uses, and one
    // four-byte access is far cheaper than the two a three-byte one compiles to (a store-buffer penalty on every
    // pixel). The caller makes sure the extra byte is inside the buffer.
    template<std::size_t Bpp, bool Wide>
    [[nodiscard]] auto load_pixel(std::uint8_t const *bytes) noexcept -> __m128i {
        std::uint32_t value = 0;

        std::memcpy(&value, bytes, Wide ? 4 : Bpp);

        return _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(value)), _mm_setzero_si128());
    }

    // The low bytes of the 16-bit lanes, each already in range 0..255. 'Wide' writes four bytes: the extra one is the
    // next pixel's first byte, which the next store overwrites.
    template<std::size_t Bpp, bool Wide>
    auto store_pixel(std::uint8_t *bytes, __m128i lanes) noexcept -> void {
        auto const packed = static_cast<std::uint32_t>(_mm_cvtsi128_si32(_mm_packus_epi16(lanes, lanes)));

        std::memcpy(bytes, &packed, Wide ? 4 : Bpp);
    }

    [[nodiscard]] auto abs16(__m128i value) noexcept -> __m128i {
        return _mm_max_epi16(value, _mm_sub_epi16(_mm_setzero_si128(), value));
    }

    // The pixels before the last use whole four-byte accesses; the last one is exact, so nothing is read or written
    // past the row's bytes.
    template<std::size_t Bpp, typename Step>
    auto for_each_pixel(std::size_t pixels, Step &&step) noexcept -> void {
        if (pixels == 0) {
            return;
        }

        for (std::size_t i = 0; i + 1 < pixels; ++i) {
            step.template operator()<true>(i);
        }

        step.template operator()<Bpp == 4>(pixels - 1);
    }

    template<std::size_t Bpp>
    auto unfilter_row_sse2(std::uint8_t filter, std::uint8_t const *source, std::uint8_t *destination,
                           std::uint8_t const *previous, std::size_t stride) noexcept -> void {
        auto const pixels = stride / Bpp;
        auto const low_byte = _mm_set1_epi16(0x00FF);
        auto const zero = _mm_setzero_si128();

        switch (filter) {
            case 0:
                std::memmove(destination, source, stride);
                break;
            case 1: {
                auto left = zero;

                for_each_pixel<Bpp>(pixels, [&]<bool Wide>(std::size_t i) {
                    left = _mm_and_si128(_mm_add_epi16(load_pixel<Bpp, Wide>(source + i * Bpp), left), low_byte);
                    store_pixel<Bpp, Wide>(destination + i * Bpp, left);
                });

                break;
            }
            case 2: {
                auto i = std::size_t{0};

                for (; i + 16 <= stride; i += 16) {
                    auto const x = _mm_loadu_si128(reinterpret_cast<__m128i const *>(source + i));
                    auto const up = _mm_loadu_si128(reinterpret_cast<__m128i const *>(previous + i));

                    _mm_storeu_si128(reinterpret_cast<__m128i *>(destination + i), _mm_add_epi8(x, up));
                }

                for (; i < stride; ++i) {
                    destination[i] = static_cast<std::uint8_t>(source[i] + previous[i]);
                }

                break;
            }
            case 3: {
                auto left = zero;

                for_each_pixel<Bpp>(pixels, [&]<bool Wide>(std::size_t i) {
                    auto const up = load_pixel<Bpp, Wide>(previous + i * Bpp);
                    auto const average = _mm_srli_epi16(_mm_add_epi16(left, up), 1);

                    left = _mm_and_si128(_mm_add_epi16(load_pixel<Bpp, Wide>(source + i * Bpp), average), low_byte);
                    store_pixel<Bpp, Wide>(destination + i * Bpp, left);
                });

                break;
            }
            default: {
                auto left = zero;
                auto up_left = zero;

                for_each_pixel<Bpp>(pixels, [&]<bool Wide>(std::size_t i) {
                    auto const up = load_pixel<Bpp, Wide>(previous + i * Bpp);
                    auto const pa_signed = _mm_sub_epi16(up, up_left);
                    auto const pb_signed = _mm_sub_epi16(left, up_left);
                    auto const pa = abs16(pa_signed);
                    auto const pb = abs16(pb_signed);
                    auto const pc = abs16(_mm_add_epi16(pa_signed, pb_signed));
                    auto const smallest = _mm_min_epi16(pa, _mm_min_epi16(pb, pc));

                    // The spec's order: the left pixel if it is closest, else the one above if it is, else the corner.
                    auto const take_left = _mm_cmpeq_epi16(pa, smallest);
                    auto const take_up = _mm_cmpeq_epi16(pb, smallest);
                    auto const up_or_corner =
                            _mm_or_si128(_mm_and_si128(take_up, up), _mm_andnot_si128(take_up, up_left));
                    auto const predictor =
                            _mm_or_si128(_mm_and_si128(take_left, left), _mm_andnot_si128(take_left, up_or_corner));

                    left = _mm_and_si128(_mm_add_epi16(load_pixel<Bpp, Wide>(source + i * Bpp), predictor), low_byte);
                    up_left = up;
                    store_pixel<Bpp, Wide>(destination + i * Bpp, left);
                });

                break;
            }
        }
    }

#endif

    // Unfilters every row, then hands it to 'on_row(row, output)' while it is still in cache. False for a filter byte
    // PNG does not define, or when 'on_row' says so.
    template<typename OnRow>
    [[nodiscard]] auto unfilter(std::uint8_t *raw, std::size_t stride, std::size_t rows, std::size_t bpp,
                                bool allow_simd, OnRow &&on_row) -> bool {
        std::vector<std::uint8_t> zeros(stride, 0);
        auto const *previous = zeros.data();

        for (std::size_t row = 0; row < rows; ++row) {
            auto const *record = raw + row * (stride + 1);
            auto const filter = record[0];
            auto const *source = record + 1;
            auto *destination = raw + row * stride;

            if (filter > 4) {
                return false;
            }

#if defined(__SSE2__)
            if (allow_simd && bpp == 4) {
                unfilter_row_sse2<4>(filter, source, destination, previous, stride);
            } else if (allow_simd && bpp == 3) {
                unfilter_row_sse2<3>(filter, source, destination, previous, stride);
            } else {
                unfilter_row_portable(filter, source, destination, previous, stride, bpp);
            }
#else
            static_cast<void>(allow_simd);
            unfilter_row_portable(filter, source, destination, previous, stride, bpp);
#endif

            if (!on_row(row, destination)) {
                return false;
            }

            previous = destination;
        }

        return true;
    }

    // ---- expansion to RGBA ----------------------------------------------------------------------------------------

    // One unfiltered row to RGBA, with the alpha stb_image gives each colour type. False for a palette index that is
    // past the palette.
    [[nodiscard]] auto expand_row_to_rgba(Chunks const &chunks, std::uint8_t const *in, std::uint8_t *out,
                                          std::size_t width) -> bool {
        switch (chunks.header.colour) {
            case ColourType::rgb:
                if (!chunks.has_colour_key) {
                    // A four-byte load and store per pixel: the load's fourth byte is the next pixel's red, ORed over
                    // with the opaque alpha. The last pixel is done by bytes so nothing is read past the row.
                    for (std::size_t x = 0; x + 1 < width; ++x) {
                        std::uint32_t pixel = 0;

                        std::memcpy(&pixel, in + x * 3, 4);
                        pixel |= 0xFF000000U;
                        std::memcpy(out + x * 4, &pixel, 4);
                    }

                    auto const last = width - 1;

                    out[last * 4] = in[last * 3];
                    out[last * 4 + 1] = in[last * 3 + 1];
                    out[last * 4 + 2] = in[last * 3 + 2];
                    out[last * 4 + 3] = 255;

                    return true;
                }

                for (std::size_t x = 0; x < width; ++x, in += 3, out += 4) {
                    auto const transparent = in[0] == chunks.colour_key[0] && in[1] == chunks.colour_key[1] &&
                                             in[2] == chunks.colour_key[2];

                    out[0] = in[0];
                    out[1] = in[1];
                    out[2] = in[2];
                    out[3] = transparent ? 0 : 255;
                }

                return true;
            case ColourType::grey:
                for (std::size_t x = 0; x < width; ++x, in += 1, out += 4) {
                    out[0] = out[1] = out[2] = in[0];
                    out[3] = chunks.has_colour_key && in[0] == chunks.colour_key[0] ? 0 : 255;
                }

                return true;
            case ColourType::grey_alpha:
                for (std::size_t x = 0; x < width; ++x, in += 2, out += 4) {
                    out[0] = out[1] = out[2] = in[0];
                    out[3] = in[1];
                }

                return true;
            case ColourType::palette:
                for (std::size_t x = 0; x < width; ++x, in += 1, out += 4) {
                    // An index past the palette is invalid; stb_image reads whatever is there, so leave it to it.
                    if (in[0] >= chunks.palette_entries) {
                        return false;
                    }

                    out[0] = chunks.palette[in[0] * 3U];
                    out[1] = chunks.palette[in[0] * 3U + 1];
                    out[2] = chunks.palette[in[0] * 3U + 2];
                    out[3] = in[0] < chunks.palette_alpha_entries ? chunks.palette_alpha[in[0]] : 255;
                }

                return true;
            case ColourType::rgba:
                return false;
        }

        return false;
    }

    struct DecompressorDeleter {
        auto operator()(libdeflate_decompressor *decompressor) const noexcept -> void {
            libdeflate_free_decompressor(decompressor);
        }
    };

    // One per thread: allocating one costs more than a small image's decode.
    auto thread_decompressor() -> libdeflate_decompressor * {
        thread_local std::unique_ptr<libdeflate_decompressor, DecompressorDeleter> decompressor{
                libdeflate_alloc_decompressor()};

        return decompressor.get();
    }

} // namespace

auto has_png_signature(std::span<std::byte const> encoded) noexcept -> bool {
    return encoded.size() >= png_signature.size() &&
           std::memcmp(encoded.data(), png_signature.data(), png_signature.size()) == 0;
}

auto decode_png_fast(std::span<std::byte const> encoded, PngDecodeOptions const &options) -> std::optional<PngRgba8> {
    auto const file = std::span{reinterpret_cast<std::uint8_t const *>(encoded.data()), encoded.size()};
    Chunks chunks;

    if (!parse_chunks(file, chunks)) {
        return std::nullopt;
    }

    auto const width = static_cast<std::size_t>(chunks.header.width);
    auto const height = static_cast<std::size_t>(chunks.header.height);
    auto const channels = channels_of(chunks.header.colour);
    auto const stride = width * channels;
    auto const raw_bytes = (stride + 1) * height;
    auto const rgba_bytes = width * height * 4;

    if (raw_bytes > max_bytes || rgba_bytes > max_bytes) {
        return std::nullopt;
    }

    // The zlib stream, in one piece for the decompressor.
    std::vector<std::uint8_t> joined;
    std::span<std::uint8_t const> compressed;

    if (chunks.data.size() == 1) {
        compressed = chunks.data.front();
    } else {
        joined.reserve(chunks.data_bytes);

        for (auto const &piece: chunks.data) {
            joined.insert(joined.end(), piece.begin(), piece.end());
        }

        compressed = joined;
    }

    // RGBA is decoded in the buffer that becomes the result; the other types need a scratch of their own.
    auto const in_place = chunks.header.colour == ColourType::rgba;
    std::vector<std::byte> pixels(in_place ? raw_bytes : 0);
    std::vector<std::uint8_t> scratch(in_place ? 0 : raw_bytes);
    auto *raw = in_place ? reinterpret_cast<std::uint8_t *>(pixels.data()) : scratch.data();

    std::size_t produced = 0;

    if (libdeflate_zlib_decompress_ex(thread_decompressor(), compressed.data(), compressed.size(), raw, raw_bytes,
                                      nullptr, &produced) != LIBDEFLATE_SUCCESS ||
        produced != raw_bytes) {
        return std::nullopt;
    }

    PngRgba8 result;

    result.width = chunks.header.width;
    result.height = chunks.header.height;

    if (in_place) {
        if (!unfilter(raw, stride, height, channels, options.allow_simd,
                      [](std::size_t, std::uint8_t const *) { return true; })) {
            return std::nullopt;
        }

        pixels.resize(rgba_bytes);
        result.pixels = std::move(pixels);

        return result;
    }

    // The other types are expanded to RGBA row by row, while each row is still in cache.
    result.pixels.resize(rgba_bytes);

    auto *const rgba = reinterpret_cast<std::uint8_t *>(result.pixels.data());

    if (!unfilter(raw, stride, height, channels, options.allow_simd,
                  [&](std::size_t row, std::uint8_t const *unfiltered) {
                      return expand_row_to_rgba(chunks, unfiltered, rgba + row * width * 4, width);
                  })) {
        return std::nullopt;
    }

    return result;
}
