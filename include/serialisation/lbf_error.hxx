#pragma once

#include <cstdint>
#include <format>
#include <optional>
#include <string_view>

#include "core/error_context.hxx"

enum class LbfErrorType : std::uint8_t {
    file_not_found,
    io_failed,
    not_an_lbf_file,
    unsupported_version,
    wrong_file_kind,
    corrupt_header,
    corrupt_table_of_contents,
    checksum_mismatch,
    decompression_failed,
    compression_failed,
    chunk_not_found,
    malformed_payload,
    cook_failed,
    instantiate_failed,
};

struct LbfError {
    LbfErrorType type = LbfErrorType::malformed_payload;
    std::optional<ErrorCause> cause{std::nullopt};
};

template<>
struct std::formatter<LbfErrorType> : std::formatter<std::string_view> {
    constexpr auto format(LbfErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case LbfErrorType::file_not_found:
                    return "file_not_found";
                case LbfErrorType::io_failed:
                    return "io_failed";
                case LbfErrorType::not_an_lbf_file:
                    return "not_an_lbf_file";
                case LbfErrorType::unsupported_version:
                    return "unsupported_version";
                case LbfErrorType::wrong_file_kind:
                    return "wrong_file_kind";
                case LbfErrorType::corrupt_header:
                    return "corrupt_header";
                case LbfErrorType::corrupt_table_of_contents:
                    return "corrupt_table_of_contents";
                case LbfErrorType::checksum_mismatch:
                    return "checksum_mismatch";
                case LbfErrorType::decompression_failed:
                    return "decompression_failed";
                case LbfErrorType::compression_failed:
                    return "compression_failed";
                case LbfErrorType::chunk_not_found:
                    return "chunk_not_found";
                case LbfErrorType::malformed_payload:
                    return "malformed_payload";
                case LbfErrorType::cook_failed:
                    return "cook_failed";
                case LbfErrorType::instantiate_failed:
                    return "instantiate_failed";
            }

            return "unknown_lbf_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
