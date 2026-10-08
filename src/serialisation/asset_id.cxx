#include "serialisation/asset_id.hxx"

#include <system_error>
#include <utility>

#include "serialisation/checksum.hxx"

auto asset_id_from_key(std::string_view key) noexcept -> AssetId {
    auto const hash = xxh64(key);
    return AssetId{.value = hash != 0 ? hash : 1};
}

auto model_asset_key(AssetPath const &path) -> std::string {
    return std::format("model:{}", path.key());
}

auto texture_asset_key(AssetPath const &path, TextureRole role) -> std::string {
    return std::format("texture:{}|{}", path.key(), std::to_underlying(role));
}

auto embedded_texture_asset_key(std::string_view cache_key, TextureRole role) -> std::string {
    return std::format("texture:{}|{}", cache_key, std::to_underlying(role));
}

auto engine_asset_key(std::string_view name) -> std::string { return std::format("{}{}", engine_asset_prefix, name); }

auto environment_asset_key(AssetPath const &path) -> std::string {
    return std::format("environment:{}", path.key());
}
