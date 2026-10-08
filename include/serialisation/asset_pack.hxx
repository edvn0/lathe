#pragma once

#include "core/paths.hxx"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "assets/hdr_image.hxx"
#include "assets/load_model.hxx"
#include "assets/texture_pipeline.hxx"
#include "gpu/compressed_texture.hxx"
#include "gpu/sampler.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/lbf_container.hxx"

class SamplerStorage;

[[nodiscard]]
auto sampler_for(SamplerStorage const &sampler_storage, DefaultSampler sampler) noexcept -> SamplerHandle;

[[nodiscard]]
auto default_sampler_of(SamplerStorage const &sampler_storage, SamplerHandle handle) noexcept -> DefaultSampler;

class AssetPack : public std::enable_shared_from_this<AssetPack> {
public:
    [[nodiscard]]
    static auto open(std::filesystem::path const &path, LbfReadOptions const &options = {})
            -> std::expected<std::shared_ptr<AssetPack>, LbfError>;

    [[nodiscard]]
    static auto from_reader(LbfReader reader) -> std::shared_ptr<AssetPack>;

    [[nodiscard]] auto reader() const noexcept -> LbfReader const & { return reader_; }

    [[nodiscard]] auto has_model(AssetId id) const noexcept -> bool;
    [[nodiscard]] auto has_texture(AssetId id) const noexcept -> bool;
    [[nodiscard]] auto has_environment(AssetId id) const noexcept -> bool;

    [[nodiscard]]
    auto load_environment(AssetId id) const -> std::expected<HdrImage, LbfError>;

    [[nodiscard]]
    auto load_texture(AssetId id) const -> std::expected<CompressedTexture, LbfError>;

    [[nodiscard]]
    auto load_model(AssetId id, SamplerStorage const &sampler_storage) const -> std::expected<ModelCpuData, LbfError>;

    [[nodiscard]]
    auto texture_loader(AssetId id) const -> CookedTextureLoader;

    [[nodiscard]]
    static auto texture_cache_key(AssetId id) -> std::string;

private:
    explicit AssetPack(LbfReader reader) noexcept : reader_(std::move(reader)) {}

    LbfReader reader_;
};

struct AssetCookRequest {
    struct Texture {
        AssetPath path;
        TextureRole role = TextureRole::colour;
    };

    std::vector<AssetPath> models;
    std::vector<Texture> textures;

    std::vector<AssetPath> environments;
};

struct AssetCookReport {
    std::uint32_t models_cooked = 0;
    std::uint32_t models_copied = 0;
    std::uint32_t textures_cooked = 0;
    std::uint32_t textures_copied = 0;
    std::uint32_t environments_cooked = 0;
    std::uint32_t environments_copied = 0;

    std::vector<std::string> failures;
};

struct AssetCookOptions {
    std::vector<std::shared_ptr<AssetPack const>> source_packs;

    std::filesystem::path texture_cache_directory = default_texture_cache_directory();
};

[[nodiscard]]
auto cook_assets(AssetCookRequest const &request, SamplerStorage &sampler_storage, LbfWriter &writer,
                 AssetCookOptions const &options = {}) -> AssetCookReport;
