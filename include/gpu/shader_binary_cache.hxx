#pragma once

#include <volk.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "storage/cache_store.hxx"

enum class ShaderBinaryCacheErrorType : std::uint8_t {
    invalid_argument,
    store_failed,
};

struct ShaderBinaryCacheError {
    ShaderBinaryCacheErrorType type = ShaderBinaryCacheErrorType::invalid_argument;
    std::optional<CacheStoreError> cause{std::nullopt};
};

// The driver's compiled shader objects (vkGetShaderBinaryDataEXT), cached so the next run skips the driver's compile.
// Binaries are only valid for the driver that made them, so the driver's binary UUID and version are part of every key;
// the SPIR-V itself is too, so an edited shader never gets the old binary.
class ShaderBinaryCache {
public:
    static constexpr std::string_view namespace_name = "shader_binaries";

    ShaderBinaryCache() = default;

    [[nodiscard]]
    static auto create(std::shared_ptr<CacheStore> store, std::array<std::uint8_t, VK_UUID_SIZE> binary_uuid,
                       std::uint32_t binary_version) -> std::expected<ShaderBinaryCache, ShaderBinaryCacheError>;

    // 'cache_key' names the stage (path, entry point, stage, defines); 'spirv' is what it was compiled from.
    [[nodiscard]]
    auto find(std::string_view cache_key, std::span<std::uint32_t const> spirv) const
            -> std::optional<std::vector<std::byte>>;

    auto store(std::string_view cache_key, std::span<std::uint32_t const> spirv,
               std::span<std::byte const> binary) const -> void;

    // Forgets a binary the driver turned out to reject.
    auto discard(std::string_view cache_key, std::span<std::uint32_t const> spirv) const -> void;

private:
    [[nodiscard]]
    auto full_key(std::string_view cache_key, std::span<std::uint32_t const> spirv) const -> std::string;

    std::shared_ptr<CacheStore> store_;
    std::string driver_;
};
