#pragma once

#include <cstdint>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#include "assets/texture_pipeline.hxx"
#include "core/paths.hxx"

struct AssetId {
    std::uint64_t value = 0;

    [[nodiscard]] auto valid() const noexcept -> bool { return value != 0; }

    auto operator==(AssetId const &) const -> bool = default;
    auto operator<=>(AssetId const &) const = default;
};

struct AssetIdHash {
    auto operator()(AssetId id) const noexcept -> std::size_t { return static_cast<std::size_t>(id.value); }
};

inline constexpr std::string_view engine_asset_prefix = "engine://";

[[nodiscard]]
auto asset_id_from_key(std::string_view key) noexcept -> AssetId;

[[nodiscard]]
auto model_asset_key(AssetPath const &path) -> std::string;

[[nodiscard]]
auto texture_asset_key(AssetPath const &path, TextureRole role) -> std::string;

[[nodiscard]]
auto embedded_texture_asset_key(std::string_view cache_key, TextureRole role) -> std::string;

[[nodiscard]]
auto engine_asset_key(std::string_view name) -> std::string;

[[nodiscard]]
auto environment_asset_key(AssetPath const &path) -> std::string;

template<>
struct std::formatter<AssetId> : std::formatter<std::string_view> {
    auto format(AssetId id, std::format_context &context) const {
        return std::format_to(context.out(), "{:016x}", id.value);
    }
};
