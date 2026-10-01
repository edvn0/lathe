#pragma once

#include <cstdint>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

#include "assets/texture_pipeline.hxx"

// Stable identity of a cooked asset: xxh64 of a canonical key string. Scenes reference assets by AssetId and
// asset packs index their chunks by it, so a scene can find a model in any pack that has it.
//
// Keys:
//   model:   "model:<path>"                  e.g. "model:assets/models/sponza/Sponza.gltf"
//   texture: "texture:<path>|<role>"         a texture file, cooked for one TextureRole
//            "texture:<cache key>|<role>"    an image embedded in a glTF (see ModelCpuImageSource::cache_key)
//   engine:  "engine://<name>"               built-in procedural models (cube, sphere, ...); never cooked
//
// Paths are lexically normalised and stored with forward slashes, relative to the working directory when the
// file lives under it, so a scene saved on one machine still resolves on another.
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

// Normalised, generic-format, working-directory-relative where possible.
[[nodiscard]]
auto normalise_asset_path(std::filesystem::path const &path) -> std::string;

[[nodiscard]]
auto asset_id_from_key(std::string_view key) noexcept -> AssetId;

[[nodiscard]]
auto model_asset_key(std::filesystem::path const &path) -> std::string;

[[nodiscard]]
auto texture_asset_key(std::filesystem::path const &path, TextureRole role) -> std::string;

[[nodiscard]]
auto embedded_texture_asset_key(std::string_view cache_key, TextureRole role) -> std::string;

[[nodiscard]]
auto engine_asset_key(std::string_view name) -> std::string;

template<>
struct std::formatter<AssetId> : std::formatter<std::string_view> {
    auto format(AssetId id, std::format_context &context) const {
        return std::format_to(context.out(), "{:016x}", id.value);
    }
};
