#include "serialisation/asset_id.hxx"

#include <system_error>
#include <utility>

#include "serialisation/checksum.hxx"

auto normalise_asset_path(std::filesystem::path const &path) -> std::string {
    std::error_code error;
    auto const absolute = std::filesystem::weakly_canonical(path, error);

    if (error) {
        return path.lexically_normal().generic_string();
    }

    auto const working_directory = std::filesystem::current_path(error);

    if (!error) {
        auto relative = absolute.lexically_relative(working_directory);

        if (!relative.empty() && *relative.begin() != "..") {
            return relative.generic_string();
        }
    }

    return absolute.generic_string();
}

auto asset_id_from_key(std::string_view key) noexcept -> AssetId {
    auto const hash = xxh64(key);
    // 0 is "no asset"; remap the one key that could hash to it.
    return AssetId{.value = hash != 0 ? hash : 1};
}

auto model_asset_key(std::filesystem::path const &path) -> std::string {
    return std::format("model:{}", normalise_asset_path(path));
}

auto texture_asset_key(std::filesystem::path const &path, TextureRole role) -> std::string {
    return std::format("texture:{}|{}", normalise_asset_path(path), std::to_underlying(role));
}

auto embedded_texture_asset_key(std::string_view cache_key, TextureRole role) -> std::string {
    return std::format("texture:{}|{}", cache_key, std::to_underlying(role));
}

auto engine_asset_key(std::string_view name) -> std::string { return std::format("{}{}", engine_asset_prefix, name); }
