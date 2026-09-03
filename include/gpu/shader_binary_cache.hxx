#pragma once

#include <volk.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"

enum class ShaderBinaryCacheErrorType : std::uint8_t {
    invalid_argument,
    directory_creation_failed,
};

struct ShaderBinaryCacheError {
    ShaderBinaryCacheErrorType type = ShaderBinaryCacheErrorType::invalid_argument;
    std::optional<ErrorContext> context{std::nullopt};
};

class ShaderBinaryCache {
public:
    ShaderBinaryCache() = default;

    [[nodiscard]]
    static auto create(std::filesystem::path directory, std::array<std::uint8_t, VK_UUID_SIZE> binary_uuid,
                       std::uint32_t binary_version) -> std::expected<ShaderBinaryCache, ShaderBinaryCacheError>;

    [[nodiscard]]
    auto find(std::string_view cache_key) const -> std::optional<std::vector<std::byte>>;

    auto store(std::string_view cache_key, std::span<std::byte const> binary) const -> void;

private:
    [[nodiscard]]
    auto path_for(std::string_view cache_key) const -> std::filesystem::path;

    std::filesystem::path directory_;
};
