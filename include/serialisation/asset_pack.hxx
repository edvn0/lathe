#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/texture_pipeline.hxx"
#include "gpu/compressed_texture.hxx"
#include "gpu/sampler.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/lbf_container.hxx"

class SamplerStorage;

[[nodiscard]]
auto sampler_for(SamplerStorage const &sampler_storage, DefaultSampler sampler) noexcept -> SamplerHandle;

// linear_repeat for handles that aren't one of the defaults.
[[nodiscard]]
auto default_sampler_of(SamplerStorage const &sampler_storage, SamplerHandle handle) noexcept -> DefaultSampler;

// Read side of an .lbf file's MODL/TEXR chunks, whether it's a scene with embedded assets or a standalone pack.
// Shared, because the texture loaders it hands out keep it alive until the textures finish streaming.
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

    // Thread-safe.
    [[nodiscard]]
    auto load_texture(AssetId id) const -> std::expected<CompressedTexture, LbfError>;

    // A finalized ModelCpuData ready for record_model_gpu_upload()/start_model_gpu_upload(). Its image sources load
    // their TEXR chunks from this pack on the texture streamer's threads. Thread-safe.
    [[nodiscard]]
    auto load_model(AssetId id, SamplerStorage const &sampler_storage) const -> std::expected<ModelCpuData, LbfError>;

    // A loader for ModelCpuImageSource::cooked / TextureStreamer::request_cooked.
    [[nodiscard]]
    auto texture_loader(AssetId id) const -> CookedTextureLoader;

    // De-duplication key for TextureStreamer::request_cooked: identical across packs for the same asset.
    [[nodiscard]]
    static auto texture_cache_key(AssetId id) -> std::string;

private:
    explicit AssetPack(LbfReader reader) noexcept : reader_(std::move(reader)) {}

    LbfReader reader_;
};

// What to put in a pack.
struct AssetCookRequest {
    struct Texture {
        std::filesystem::path path;
        TextureRole role = TextureRole::colour;
    };

    std::vector<std::filesystem::path> models;
    std::vector<Texture> textures;
};

struct AssetCookReport {
    std::uint32_t models_cooked = 0;
    std::uint32_t models_copied = 0; // reused verbatim from a source pack
    std::uint32_t textures_cooked = 0;
    std::uint32_t textures_copied = 0;

    // Assets that couldn't be cooked; scenes keep referencing their source paths, so they still load from source.
    std::vector<std::string> failures;
};

struct AssetCookOptions {
    // Packs to copy already-cooked chunks from instead of cooking again, e.g. the scene file being re-saved.
    std::vector<std::shared_ptr<AssetPack const>> source_packs;

    std::filesystem::path texture_cache_directory = default_texture_cache_directory();
};

// Cooks every requested model (glTF parse, finalization, meshlets) and texture (BC5/BC7 through the .ktx2 cache)
// on thread_pool() and adds their MODL/TEXR chunks to `writer`. Textures referenced by several models are cooked
// once. Call from outside thread_pool().
[[nodiscard]]
auto cook_assets(AssetCookRequest const &request, SamplerStorage &sampler_storage, LbfWriter &writer,
                 AssetCookOptions const &options = {}) -> AssetCookReport;
