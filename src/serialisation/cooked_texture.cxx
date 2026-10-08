#include "serialisation/cooked_texture.hxx"

#include <algorithm>
#include <utility>

#include "gpu/compressed_texture.hxx"
#include "serialisation/byte_stream.hxx"

auto encode_cooked_texture(CompressedTexture const &texture, TextureRole role) -> std::vector<std::byte> {
    ByteWriter writer;

    writer.write(static_cast<std::uint32_t>(texture.format));
    writer.write(texture.width);
    writer.write(texture.height);
    writer.write(std::to_underlying(role));
    writer.write_string(texture.debug_name.view());

    writer.write(static_cast<std::uint32_t>(texture.mips.size()));

    for (auto const &mip: texture.mips) {
        writer.write(mip.width);
        writer.write(mip.height);
        writer.write(mip.byte_offset);
        writer.write(mip.byte_length);
    }

    writer.write(static_cast<std::uint64_t>(texture.data.size()));
    writer.align(16);
    writer.write_span(std::span<std::byte const>{texture.data});

    return writer.take();
}

namespace {
    inline constexpr std::size_t max_debug_name_length = 256;
}

auto decode_cooked_texture(std::span<std::byte const> payload, std::uint16_t version)
        -> std::expected<CookedTexture, LbfError> {
    if (version < cooked_texture_oldest_readable_version || version > cooked_texture_version) {
        return std::unexpected(LbfError{.type = LbfErrorType::unsupported_version});
    }

    ByteReader reader{payload};
    CookedTexture cooked;
    auto &texture = cooked.texture;

    auto const format = reader.read<std::uint32_t>();

    if (compressed_block_bytes(format) == 0) {
        reader.fail();
    }

    texture.format = reader.ok() ? static_cast<VkFormat>(format) : VK_FORMAT_UNDEFINED;
    texture.width = reader.read<std::uint32_t>();
    texture.height = reader.read<std::uint32_t>();

    auto const role = reader.read<std::uint8_t>();

    if (role > std::to_underlying(TextureRole::normal_map)) {
        reader.fail();
    }

    cooked.role = static_cast<TextureRole>(role);

    auto debug_name = reader.read_string();
    debug_name.resize(std::min(debug_name.size(), max_debug_name_length));
    texture.debug_name = FlyString{debug_name};

    auto const mip_count = reader.read<std::uint32_t>();

    if (mip_count > 32) {
        reader.fail();
    }

    texture.mips.resize(reader.ok() ? mip_count : 0);

    for (auto &mip: texture.mips) {
        mip.width = reader.read<std::uint32_t>();
        mip.height = reader.read<std::uint32_t>();
        mip.byte_offset = reader.read<std::uint32_t>();
        mip.byte_length = reader.read<std::uint32_t>();
    }

    auto const data_size = reader.read<std::uint64_t>();
    reader.align(16);

    if (reader.failed() || data_size > reader.remaining()) {
        return std::unexpected(LbfError{.type = LbfErrorType::malformed_payload});
    }

    auto const data = reader.read_bytes(static_cast<std::size_t>(data_size));
    texture.data.assign(data.begin(), data.end());

    for (auto const &mip: texture.mips) {
        if (static_cast<std::uint64_t>(mip.byte_offset) + mip.byte_length > data_size) {
            return std::unexpected(LbfError{.type = LbfErrorType::malformed_payload});
        }
    }

    if (reader.failed() || texture.mips.empty() || texture.format == VK_FORMAT_UNDEFINED) {
        return std::unexpected(LbfError{.type = LbfErrorType::malformed_payload});
    }

    if (auto const problem = validate_compressed_texture(texture); problem.has_value()) {
        return std::unexpected(LbfError{
                .type = LbfErrorType::malformed_payload,
                .cause = ErrorCause{ErrorContext{.message = FlyString{*problem}}},
        });
    }

    return cooked;
}
