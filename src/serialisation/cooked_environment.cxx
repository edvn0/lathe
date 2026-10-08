#include "serialisation/cooked_environment.hxx"

#include <bit>
#include <cstring>

#include "serialisation/byte_stream.hxx"

auto encode_cooked_environment(HdrImage const &image) -> std::vector<std::byte> {
    ByteWriter writer;

    writer.write(image.width);
    writer.write(image.height);
    writer.write(cooked_environment_format);
    writer.write(image.layers);

    auto const count = image.pixels.size();

    std::vector<std::byte> shuffled(count * sizeof(std::uint16_t));

    for (std::size_t index = 0; index < count; ++index) {
        shuffled[index] = static_cast<std::byte>(image.pixels[index] & 0xFFU);
        shuffled[count + index] = static_cast<std::byte>(image.pixels[index] >> 8U);
    }

    writer.write_span(std::span<std::byte const>{shuffled});

    return writer.take();
}

auto decode_cooked_environment(std::span<std::byte const> payload, std::uint16_t version)
        -> std::expected<HdrImage, LbfError> {
    if (version < cooked_environment_oldest_readable_version || version > cooked_environment_version) {
        return std::unexpected(LbfError{.type = LbfErrorType::unsupported_version});
    }

    auto const malformed = [](std::string_view what) {
        return std::unexpected(LbfError{
                .type = LbfErrorType::malformed_payload,
                .cause = ErrorCause{ErrorContext{.message = FlyString{what}}},
        });
    };

    ByteReader reader{payload};

    HdrImage image;
    image.width = reader.read<std::uint32_t>();
    image.height = reader.read<std::uint32_t>();

    auto const format = reader.read<std::uint32_t>();
    image.layers = reader.read<std::uint32_t>();

    if (reader.failed()) {
        return malformed("ENVM header is truncated");
    }

    if (format != cooked_environment_format) {
        return malformed("ENVM format is not R16G16B16A16_SFLOAT");
    }

    if (image.layers == 1) {
        if (image.width == 0 || image.height == 0 || image.width > hdr_equirect_max_width ||
            image.height > hdr_equirect_max_height || image.width != 2 * image.height) {
            return malformed("ENVM equirect must be 2:1 and at most 16384x8192");
        }
    } else if (image.layers == 6) {
        if (image.width == 0 || image.width != image.height || !std::has_single_bit(image.width) ||
            image.width > hdr_cube_max_face_size) {
            return malformed("ENVM cube faces must be square powers of two up to 2048");
        }
    } else {
        return malformed("ENVM layers must be 1 or 6");
    }

    auto const count = static_cast<std::size_t>(image.width) * image.height * image.layers * 4U;
    auto const expected_bytes = count * sizeof(std::uint16_t);

    if (reader.remaining() != expected_bytes) {
        return malformed("ENVM pixel data does not match its dimensions");
    }

    auto const bytes = reader.read_bytes(expected_bytes);

    image.pixels.resize(count);

    for (std::size_t index = 0; index < count; ++index) {
        image.pixels[index] = static_cast<std::uint16_t>(
                std::to_integer<std::uint16_t>(bytes[index]) |
                static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(bytes[count + index]) << 8U));
    }

    return image;
}
