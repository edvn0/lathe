#include "serialisation/asset_pack.hxx"

#include <future>
#include <optional>
#include <source_location>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "gpu/sampler_storage.hxx"
#include "serialisation/cooked_model.hxx"
#include "serialisation/cooked_texture.hxx"

namespace {

    auto make_error(LbfErrorType type, std::string_view message,
                    std::source_location location = std::source_location::current()) -> LbfError {
        return LbfError{
                .type = type,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{message},
                        .location = location,
                }},
        };
    }

    [[nodiscard]] auto texture_role_for(ModelTextureSlot slot) noexcept -> TextureRole {
        switch (slot) {
            case ModelTextureSlot::base_colour:
            case ModelTextureSlot::emissive:
                return TextureRole::colour;
            case ModelTextureSlot::normal:
                return TextureRole::normal_map;
            case ModelTextureSlot::metallic_roughness:
            case ModelTextureSlot::occlusion:
                return TextureRole::generic;
        }

        return TextureRole::generic;
    }

    [[nodiscard]] auto image_asset_id(ModelCpuImageSource const &source) -> AssetId {
        auto const role = texture_role_for(source.slot);
        auto const key = source.path.empty() ? embedded_texture_asset_key(source.cache_key, role)
                                             : texture_asset_key(source.path, role);
        return asset_id_from_key(key);
    }

    [[nodiscard]] auto current_version(std::uint32_t type) noexcept -> std::uint16_t {
        return type == lbf_chunk::model ? cooked_model_version : cooked_texture_version;
    }

    // Finds `id` in the first source pack that has it at the current chunk version. Older versions still load, but
    // a save re-cooks them so files converge on the newest layout.
    [[nodiscard]]
    auto find_in_packs(std::span<std::shared_ptr<AssetPack const> const> packs, std::uint32_t type, AssetId id)
            -> std::pair<AssetPack const *, LbfChunkEntry const *> {
        for (auto const &pack: packs) {
            if (pack == nullptr) {
                continue;
            }

            if (auto const *entry = pack->reader().find(type, id.value);
                entry != nullptr && entry->version == current_version(type)) {
                return {pack.get(), entry};
            }
        }

        return {nullptr, nullptr};
    }

    struct TextureJob {
        AssetId id;
        std::filesystem::path path; // or
        std::vector<std::byte> encoded;
        std::string cache_key;
        TextureRole role = TextureRole::colour;
        std::string debug_name;
    };

    struct ParsedModel {
        AssetId id;
        std::filesystem::path source;
        ModelCpuData cpu_data;
    };

} // namespace

auto sampler_for(SamplerStorage const &sampler_storage, DefaultSampler sampler) noexcept -> SamplerHandle {
    switch (sampler) {
        case DefaultSampler::linear_repeat:
            return sampler_storage.linear_repeat();
        case DefaultSampler::linear_clamp:
            return sampler_storage.linear_clamp();
        case DefaultSampler::nearest_repeat:
            return sampler_storage.nearest_repeat();
        case DefaultSampler::nearest_clamp:
            return sampler_storage.nearest_clamp();
        case DefaultSampler::shadow_compare:
            return sampler_storage.shadow_compare();
    }

    return sampler_storage.linear_repeat();
}

auto default_sampler_of(SamplerStorage const &sampler_storage, SamplerHandle handle) noexcept -> DefaultSampler {
    for (auto const sampler:
         {DefaultSampler::linear_clamp, DefaultSampler::nearest_repeat, DefaultSampler::nearest_clamp}) {
        if (sampler_for(sampler_storage, sampler) == handle) {
            return sampler;
        }
    }

    return DefaultSampler::linear_repeat;
}

auto AssetPack::open(std::filesystem::path const &path, LbfReadOptions const &options)
        -> std::expected<std::shared_ptr<AssetPack>, LbfError> {
    auto reader = LbfReader::open(path, options);

    if (!reader) {
        return std::unexpected(reader.error());
    }

    return from_reader(std::move(*reader));
}

auto AssetPack::from_reader(LbfReader reader) -> std::shared_ptr<AssetPack> {
    // The constructor is private, so make_shared can't reach it.
    return std::shared_ptr<AssetPack>{new AssetPack{std::move(reader)}};
}

auto AssetPack::has_model(AssetId id) const noexcept -> bool {
    return reader_.find(lbf_chunk::model, id.value) != nullptr;
}

auto AssetPack::has_texture(AssetId id) const noexcept -> bool {
    return reader_.find(lbf_chunk::texture, id.value) != nullptr;
}

auto AssetPack::texture_cache_key(AssetId id) -> std::string { return std::format("lbf:{}", id); }

auto AssetPack::load_texture(AssetId id) const -> std::expected<CompressedTexture, LbfError> {
    auto const *entry = reader_.find(lbf_chunk::texture, id.value);

    if (entry == nullptr) {
        return std::unexpected(make_error(LbfErrorType::chunk_not_found, std::format("texture {}", id)));
    }

    auto payload = reader_.read_chunk(*entry);

    if (!payload) {
        return std::unexpected(payload.error());
    }

    auto cooked = decode_cooked_texture(*payload, entry->version);

    if (!cooked) {
        return std::unexpected(cooked.error());
    }

    return std::move(cooked->texture);
}

auto AssetPack::texture_loader(AssetId id) const -> CookedTextureLoader {
    return [pack = shared_from_this(), id]() -> std::expected<CompressedTexture, TexturePipelineError> {
        auto texture = pack->load_texture(id);

        if (!texture) {
            return std::unexpected(TexturePipelineError{
                    .type = TexturePipelineErrorType::cache_io_failed,
                    .cause = ErrorCause{Boxed<LbfError>{texture.error()}},
            });
        }

        return std::move(*texture);
    };
}

auto AssetPack::load_model(AssetId id, SamplerStorage const &sampler_storage) const
        -> std::expected<ModelCpuData, LbfError> {
    ZoneScopedNC("AssetPack::load_model", tracy::Color::Goldenrod);

    auto const *entry = reader_.find(lbf_chunk::model, id.value);

    if (entry == nullptr) {
        return std::unexpected(make_error(LbfErrorType::chunk_not_found, std::format("model {}", id)));
    }

    auto payload = reader_.read_chunk(*entry);

    if (!payload) {
        return std::unexpected(payload.error());
    }

    auto cooked = decode_cooked_model(*payload, entry->version);

    if (!cooked) {
        return std::unexpected(cooked.error());
    }

    auto &cpu_data = cooked->cpu_data;

    for (std::size_t index = 0; index < cpu_data.materials.size(); ++index) {
        cpu_data.materials[index].sampler = sampler_for(sampler_storage, cooked->material_samplers[index]);
    }

    for (std::size_t index = 0; index < cpu_data.image_sources.size(); ++index) {
        auto const texture_id = cooked->images[index].texture;
        auto &source = cpu_data.image_sources[index];

        // A texture missing from the pack (it failed to cook) leaves its slot on the default texture.
        if (!has_texture(texture_id)) {
            warn("AssetPack: model {} references texture {} ('{}'), which isn't in '{}'", id, texture_id,
                 source.debug_name, reader_.path().string());
        }

        source.cooked = texture_loader(texture_id);
        source.cache_key = texture_cache_key(texture_id);
    }

    return std::move(cpu_data);
}

auto cook_assets(AssetCookRequest const &request, SamplerStorage &sampler_storage, LbfWriter &writer,
                 AssetCookOptions const &options) -> AssetCookReport {
    ZoneScopedNC("cook_assets", tracy::Color::Goldenrod);

    AssetCookReport report;
    auto &pool = thread_pool();
    auto const &packs = options.source_packs;

    auto const copy_from_pack = [&](AssetPack const &pack, LbfChunkEntry const &entry) -> bool {
        auto stored = pack.reader().read_stored_chunk(entry);

        if (!stored) {
            return false;
        }

        writer.add_stored_chunk(entry, std::move(*stored));
        return true;
    };

    // Textures are keyed by AssetId so each is cooked or copied once, whoever references it.
    std::unordered_map<AssetId, TextureJob, AssetIdHash> texture_jobs;
    std::unordered_set<AssetId, AssetIdHash> handled_textures;

    auto const queue_texture = [&](TextureJob &&job) {
        if (handled_textures.contains(job.id) || writer.contains(lbf_chunk::texture, job.id.value)) {
            return;
        }

        if (auto const [pack, entry] = find_in_packs(packs, lbf_chunk::texture, job.id); pack != nullptr) {
            if (copy_from_pack(*pack, *entry)) {
                handled_textures.insert(job.id);
                ++report.textures_copied;
                return;
            }
        }

        texture_jobs.try_emplace(job.id, std::move(job));
    };

    // Phase 1: models. Copy what a source pack already has; parse the rest in parallel.
    std::vector<std::future<std::expected<ModelCpuData, ModelLoadError>>> parse_tasks;
    std::vector<ParsedModel> parsed;
    std::unordered_set<AssetId, AssetIdHash> seen_models;

    for (auto const &source: request.models) {
        auto const id = asset_id_from_key(model_asset_key(source));

        if (!seen_models.insert(id).second || writer.contains(lbf_chunk::model, id.value)) {
            continue;
        }

        if (auto const [pack, entry] = find_in_packs(packs, lbf_chunk::model, id); pack != nullptr) {
            auto payload = pack->reader().read_chunk(*entry);
            std::expected<CookedModel, LbfError> cooked = std::unexpected(LbfError{});

            if (payload) {
                cooked = decode_cooked_model(*payload, entry->version);
            }

            if (cooked && copy_from_pack(*pack, *entry)) {
                ++report.models_copied;

                // Its textures come along from whichever pack has them.
                for (auto const &image: cooked->images) {
                    queue_texture(TextureJob{.id = image.texture});
                }

                continue;
            }
        }

        parsed.push_back(ParsedModel{.id = id, .source = source});
        parse_tasks.push_back(
                pool.submit_task([source, &sampler_storage] { return load_model_cpu(source, sampler_storage); }));
    }

    std::vector<ParsedModel> ready_models;

    for (std::size_t index = 0; index < parse_tasks.size(); ++index) {
        auto result = parse_tasks[index].get();

        if (!result) {
            report.failures.push_back(
                    std::format("model '{}': {}", parsed[index].source.string(), describe(result.error())));
            continue;
        }

        parsed[index].cpu_data = std::move(*result);
        ready_models.push_back(std::move(parsed[index]));
    }

    for (auto const &model: ready_models) {
        for (auto const &source: model.cpu_data.image_sources) {
            queue_texture(TextureJob{
                    .id = image_asset_id(source),
                    .path = source.path,
                    .encoded = source.encoded,
                    .cache_key = source.cache_key,
                    .role = texture_role_for(source.slot),
                    .debug_name = source.debug_name,
            });
        }
    }

    for (auto const &texture: request.textures) {
        queue_texture(TextureJob{
                .id = asset_id_from_key(texture_asset_key(texture.path, texture.role)),
                .path = texture.path,
                .role = texture.role,
                .debug_name = texture.path.filename().string(),
        });
    }

    // Phase 2: textures, in parallel. Entries queued only as "copy from pack" that no pack had are skipped.
    struct TextureResult {
        AssetId id;
        std::string debug_name;
        std::expected<std::vector<std::byte>, std::string> payload;
    };

    std::vector<std::future<TextureResult>> texture_tasks;

    for (auto &[id, job]: texture_jobs) {
        if (job.path.empty() && job.encoded.empty()) {
            report.failures.push_back(std::format("texture {}: not in any source pack and no source to cook from", id));
            continue;
        }

        texture_tasks.push_back(pool.submit_task([job = std::move(job), directory = options.texture_cache_directory] {
            auto texture = job.path.empty() ? load_compressed_texture_from_encoded_memory(job.encoded, job.role,
                                                                                          job.cache_key, directory)
                                            : load_compressed_texture(job.path, job.role, directory);

            if (!texture) {
                return TextureResult{.id = job.id,
                                     .debug_name = job.debug_name,
                                     .payload = std::unexpected(describe(texture.error()))};
            }

            return TextureResult{
                    .id = job.id, .debug_name = job.debug_name, .payload = encode_cooked_texture(*texture, job.role)};
        }));
    }

    for (auto &task: texture_tasks) {
        auto result = task.get();

        if (!result.payload) {
            report.failures.push_back(std::format("texture '{}': {}", result.debug_name, result.payload.error()));
            continue;
        }

        writer.add_chunk(LbfChunkInput{
                .type = lbf_chunk::texture,
                .id = result.id.value,
                .version = cooked_texture_version,
                .payload = std::move(*result.payload),
        });
        ++report.textures_cooked;
    }

    // Phase 3: encode models. Cheap next to parsing, but meshopt's codecs still like the extra cores.
    std::vector<std::future<std::expected<std::vector<std::byte>, LbfError>>> encode_tasks;
    encode_tasks.reserve(ready_models.size());

    for (auto const &model: ready_models) {
        encode_tasks.push_back(pool.submit_task([&model, &sampler_storage] {
            std::vector<CookedImageRef> images;
            images.reserve(model.cpu_data.image_sources.size());

            for (auto const &source: model.cpu_data.image_sources) {
                images.push_back(CookedImageRef{
                        .texture = image_asset_id(source),
                        .slot = source.slot,
                        .debug_name = source.debug_name,
                });
            }

            std::vector<DefaultSampler> samplers;
            samplers.reserve(model.cpu_data.materials.size());

            for (auto const &material: model.cpu_data.materials) {
                samplers.push_back(default_sampler_of(sampler_storage, material.sampler));
            }

            return encode_cooked_model(model.cpu_data, images, samplers);
        }));
    }

    for (std::size_t index = 0; index < encode_tasks.size(); ++index) {
        auto payload = encode_tasks[index].get();

        if (!payload) {
            report.failures.push_back(
                    std::format("model '{}': {}", ready_models[index].source.string(), describe(payload.error())));
            continue;
        }

        writer.add_chunk(LbfChunkInput{
                .type = lbf_chunk::model,
                .id = ready_models[index].id.value,
                .version = cooked_model_version,
                .payload = std::move(*payload),
        });
        ++report.models_cooked;
    }

    return report;
}
