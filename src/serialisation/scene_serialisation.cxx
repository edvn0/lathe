#include "serialisation/scene_serialisation.hxx"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <future>
#include <optional>
#include <source_location>
#include <unordered_map>
#include <utility>

#include "assets/material_storage.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/renderer_error.hxx"
#include "core/thread_pool.hxx"
#include "gpu/image_storage.hxx"
#include "gpu/sampler_storage.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"
#include "serialisation/checksum.hxx"
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

    [[nodiscard]] auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    struct EngineModelName {
        std::string_view name;
        ModelHandle EngineModels::*member;
    };

    constexpr std::array engine_model_names{
            EngineModelName{.name = "cube", .member = &EngineModels::cube},
            EngineModelName{.name = "sphere", .member = &EngineModels::sphere},
            EngineModelName{.name = "grass_clump", .member = &EngineModels::grass_clump},
            EngineModelName{.name = "capsule", .member = &EngineModels::capsule},
    };

    // Texture slots in SceneMaterial::textures order.
    constexpr std::array material_texture_roles{TextureRole::colour, TextureRole::normal_map, TextureRole::generic,
                                                TextureRole::generic, TextureRole::colour};

    [[nodiscard]] auto default_texture_for_slot(ImageStorage const &images, std::size_t slot) -> ImageHandle {
        switch (slot) {
            case scene_material_texture::normal:
                return images.flat_normal();
            case scene_material_texture::metallic_roughness:
                return images.metallic_roughness();
            case scene_material_texture::occlusion:
                return images.occlusion();
            case scene_material_texture::emissive:
                return images.emissive();
            default:
                return images.white();
        }
    }

    // ---- capture -------------------------------------------------------------------------------------------------

    class SceneCapture {
    public:
        SceneCapture(Renderer &renderer, EngineModels const &engine_models, SceneCaptureReport *report) :
            renderer_(renderer), engine_models_(engine_models), report_(report) {}

        auto warn_once(std::string message) -> void {
            if (report_ != nullptr && std::ranges::find(report_->warnings, message) == report_->warnings.end()) {
                report_->warnings.push_back(std::move(message));
            }
        }

        // scene_no_index for models with no file behind them (procedural geometry made at runtime).
        auto model_index(ModelHandle handle) -> std::uint32_t {
            if (!handle.valid()) {
                return scene_no_index;
            }

            auto const key = (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;

            if (auto const it = model_indices_.find(key); it != model_indices_.end()) {
                return it->second;
            }

            std::optional<SceneAssetRef> reference;

            for (auto const &engine_model: engine_model_names) {
                if (engine_models_.*engine_model.member == handle) {
                    auto key_text = engine_asset_key(engine_model.name);
                    reference = SceneAssetRef{.id = asset_id_from_key(key_text), .source = std::move(key_text)};
                }
            }

            if (!reference.has_value()) {
                if (auto const *source = renderer_.model_source(handle); source != nullptr) {
                    reference = SceneAssetRef{
                            .id = asset_id_from_key(model_asset_key(*source)),
                            .source = normalise_asset_path(*source),
                    };
                }
            }

            auto index = scene_no_index;

            if (reference.has_value()) {
                index = static_cast<std::uint32_t>(description.models.size());
                description.models.push_back(std::move(*reference));
            } else {
                auto const name = renderer_.assets().models().name_of(handle);
                warn_once(std::format("model '{}' was generated at runtime and has no file; its entities are saved "
                                      "without a model",
                                      name.empty() ? std::string_view{"(unnamed)"} : name));
            }

            model_indices_.emplace(key, index);
            return index;
        }

        auto texture_index(ImageHandle handle, std::size_t slot) -> std::uint32_t {
            auto const &images = renderer_.image_storage();

            if (!handle.valid() || handle == default_texture_for_slot(images, slot) || handle == images.white()) {
                return scene_no_index;
            }

            auto const *source = renderer_.texture_streamer().source_of(handle);

            if (source == nullptr || source->path.empty()) {
                warn_once("a material uses a texture with no source file (embedded or generated); it's saved with "
                          "the default texture in that slot");
                return scene_no_index;
            }

            auto key = texture_asset_key(source->path, source->role);
            auto const id = asset_id_from_key(key);

            for (std::size_t index = 0; index < description.textures.size(); ++index) {
                if (description.textures[index].id == id) {
                    return static_cast<std::uint32_t>(index);
                }
            }

            description.textures.push_back(SceneTextureRef{
                    .id = id,
                    .source = normalise_asset_path(source->path),
                    .role = source->role,
            });

            return static_cast<std::uint32_t>(description.textures.size() - 1);
        }

        auto material_index(MaterialHandle handle) -> std::uint32_t {
            if (!handle.valid()) {
                return scene_no_index;
            }

            auto const key = (static_cast<std::uint64_t>(handle.generation) << 32U) | handle.index;

            if (auto const it = material_indices_.find(key); it != material_indices_.end()) {
                return it->second;
            }

            auto const *info = renderer_.material_storage().create_info(handle);

            if (info == nullptr) {
                warn_once("a material override references a destroyed material; it was dropped");
                material_indices_.emplace(key, scene_no_index);
                return scene_no_index;
            }

            SceneMaterial material{
                    .name = std::string{renderer_.assets().materials().name_of(handle)},
                    .base_colour_factor = info->base_colour_factor,
                    .emissive_factor = info->emissive_factor,
                    .emissive_strength = info->emissive_strength,
                    .metallic_factor = info->metallic_factor,
                    .roughness_factor = info->roughness_factor,
                    .normal_scale = info->normal_scale,
                    .occlusion_strength = info->occlusion_strength,
                    .alpha_cutoff = info->alpha_cutoff,
                    .wind_strength = info->wind_strength,
                    .max_shadow_cascade = info->max_shadow_cascade,
                    .alpha_mode = info->alpha_mode,
                    .sampler = default_sampler_of(renderer_.sampler_storage(), info->sampler),
                    .debug_meshlet_colours = info->debug_meshlet_colours,
            };

            std::array const textures{info->base_colour_texture, info->normal_texture, info->metallic_roughness_texture,
                                      info->occlusion_texture, info->emissive_texture};

            for (std::size_t slot = 0; slot < textures.size(); ++slot) {
                material.textures[slot] = texture_index(textures[slot], slot);
            }

            auto const index = static_cast<std::uint32_t>(description.materials.size());
            description.materials.push_back(std::move(material));
            material_indices_.emplace(key, index);

            return index;
        }

        SceneDescription description;

    private:
        Renderer &renderer_;
        EngineModels const &engine_models_;
        SceneCaptureReport *report_;

        std::unordered_map<std::uint64_t, std::uint32_t> model_indices_;
        std::unordered_map<std::uint64_t, std::uint32_t> material_indices_;
    };

    // ---- instantiate ---------------------------------------------------------------------------------------------

    struct ResolvedModel {
        ModelHandle handle{};
        bool owned = false; // we hold the creation reference
    };

    [[nodiscard]]
    auto find_pack(std::span<std::shared_ptr<AssetPack const> const> packs, std::uint32_t type, AssetId id)
            -> std::shared_ptr<AssetPack const> {
        for (auto const &pack: packs) {
            if (pack != nullptr && pack->reader().find(type, id.value) != nullptr) {
                return pack;
            }
        }

        return nullptr;
    }

} // namespace

auto capture_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                   SceneCaptureReport *report) -> SceneDescription {
    ZoneScopedNC("capture_scene", tracy::Color::Goldenrod);

    auto const &registry = scene.get_registry();
    SceneCapture capture{renderer, engine_models, report};
    auto &description = capture.description;

    description.physics_settings = scene.physics_settings;

    // Ascending entity id, so captures of an unchanged registry produce identical bytes.
    std::vector<entt::entity> entities;

    for (auto const entity: registry.view<entt::entity>()) {
        entities.push_back(entity);
    }

    std::ranges::sort(entities, {}, [](entt::entity entity) { return entt::to_integral(entity); });

    std::unordered_map<entt::entity, std::uint32_t> index_of;
    index_of.reserve(entities.size());

    for (std::size_t index = 0; index < entities.size(); ++index) {
        index_of.emplace(entities[index], static_cast<std::uint32_t>(index));
    }

    description.entities.reserve(entities.size());

    for (auto const entity: entities) {
        SceneEntity record;

        if (auto const *meta = registry.try_get<Components::Meta>(entity)) {
            record.name = std::string{meta->name.view()};
        } else if (auto const *generated = registry.try_get<Components::GeneratedMeta>(entity)) {
            record.name = generated->name;
            record.flags = record.flags | SceneEntityFlags::generated_name;
        }

        if (auto const *transform = registry.try_get<Components::Transform>(entity)) {
            record.transform = *transform;
        }

        if (auto const *parent = registry.try_get<Components::Parent>(entity)) {
            if (auto const it = index_of.find(parent->entity); it != index_of.end()) {
                record.parent = it->second;
            }
        }

        if (registry.all_of<Components::PlayerTag>(entity)) {
            record.flags = record.flags | SceneEntityFlags::player;
        }

        if (registry.all_of<Components::BulletTag>(entity)) {
            record.flags = record.flags | SceneEntityFlags::bullet;
        }

        if (registry.all_of<Components::StreamedModelTag>(entity)) {
            record.flags = record.flags | SceneEntityFlags::streamed_model;
        }

        description.entities.push_back(std::move(record));
    }

    for (std::size_t index = 0; index < entities.size(); ++index) {
        auto const entity = entities[index];
        auto const entity_index = static_cast<std::uint32_t>(index);

        ModelHandle entity_model{};

        if (auto const *model = registry.try_get<Components::Model>(entity)) {
            entity_model = model->model;

            if (auto const model_index = capture.model_index(model->model); model_index != scene_no_index) {
                description.model_components.push_back(
                        SceneModelComponent{.entity = entity_index, .model = model_index});
            }
        }

        if (auto const *material_override = registry.try_get<Components::MaterialOverride>(entity)) {
            SceneMaterialOverrideComponent component{
                    .entity = entity_index,
                    .material = capture.material_index(material_override->material),
            };

            if (!material_override->slots.empty()) {
                auto const model_materials = renderer.model_materials(entity_model);

                for (auto const &slot: material_override->slots) {
                    auto const source = std::ranges::find(model_materials, slot.source);
                    auto const material = capture.material_index(slot.material);

                    if (source == model_materials.end() || material == scene_no_index) {
                        capture.warn_once("a per-slot material override no longer matches its model; it was dropped");
                        continue;
                    }

                    component.slots.push_back(SceneMaterialOverrideComponent::Slot{
                            .source_slot = static_cast<std::uint32_t>(source - model_materials.begin()),
                            .material = material,
                    });
                }
            }

            if (component.material != scene_no_index || !component.slots.empty()) {
                description.material_overrides.push_back(std::move(component));
            }
        }

        if (auto const *instanced = registry.try_get<Components::InstancedModel>(entity)) {
            if (auto const model_index = capture.model_index(instanced->model); model_index != scene_no_index) {
                description.instanced_models.push_back(SceneInstancedModelComponent{
                        .entity = entity_index,
                        .model = model_index,
                        .material = capture.material_index(instanced->material_override),
                        .transforms = instanced->transforms,
                });
            }
        }

        if (auto const *light = registry.try_get<Components::PointLight>(entity)) {
            description.point_lights.push_back(ScenePointLightComponent{.entity = entity_index, .light = *light});
        }

        if (auto const *light = registry.try_get<Components::SpotLight>(entity)) {
            description.spot_lights.push_back(SceneSpotLightComponent{.entity = entity_index, .light = *light});
        }

        if (auto const *body = registry.try_get<Components::RigidBody>(entity)) {
            if (body->shape == Components::BodyShape::heightfield) {
                capture.warn_once("heightfield colliders come from terrain generation and aren't saved");
            } else {
                auto copy = *body;
                copy.heightfield = nullptr;
                description.rigid_bodies.push_back(
                        SceneRigidBodyComponent{.entity = entity_index, .body = std::move(copy)});
            }
        }

        if (auto const *script = registry.try_get<Components::Script>(entity)) {
            auto const name = renderer.assets().scripts().name_of(script->script);

            if (name.empty()) {
                capture.warn_once("a script without a registered name can't be saved; it was dropped");
            } else {
                description.scripts.push_back(
                        SceneScriptComponent{.entity = entity_index, .script = std::string{name}});
            }
        }

        if (auto const *lifetime = registry.try_get<Components::Lifetime>(entity)) {
            description.lifetimes.push_back(
                    SceneLifetimeComponent{.entity = entity_index, .remaining_seconds = lifetime->remaining_seconds});
        }
    }

    return std::move(description);
}

auto instantiate_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models,
                       SceneDescription const &description, SceneInstantiateOptions const &options)
        -> std::expected<SceneInstantiateReport, LbfError> {
    ZoneScopedNC("instantiate_scene", tracy::Color::Goldenrod);

    if (auto valid = validate_scene(description); !valid) {
        return std::unexpected(valid.error());
    }

    SceneInstantiateReport report;
    auto const &packs = options.packs;

    // ---- models: decode every cooked one in parallel, then upload on this thread.
    std::vector<ResolvedModel> models(description.models.size());

    struct PendingDecode {
        std::size_t index;
        std::future<std::expected<ModelCpuData, LbfError>> future;
    };

    std::vector<PendingDecode> decodes;

    for (std::size_t index = 0; index < description.models.size(); ++index) {
        auto const &reference = description.models[index];

        if (reference.source.starts_with(engine_asset_prefix)) {
            auto const name = std::string_view{reference.source}.substr(engine_asset_prefix.size());

            for (auto const &engine_model: engine_model_names) {
                if (engine_model.name == name) {
                    models[index].handle = engine_models.*engine_model.member;
                }
            }

            if (!models[index].handle.valid()) {
                report.warnings.push_back(std::format("unknown built-in model '{}'", reference.source));
            }

            continue;
        }

        // Already loaded (e.g. reloading the scene): reuse it rather than uploading a second copy.
        if (auto const cached = renderer.cached_model(reference.source); cached.valid()) {
            renderer.retain_model(cached);
            models[index] = ResolvedModel{.handle = cached, .owned = true};
            ++report.models_reused;
            continue;
        }

        auto pack = find_pack(packs, lbf_chunk::model, reference.id);
        auto const debug_name = std::filesystem::path{reference.source}.filename().string();

        if (options.stream_models) {
            ModelHandle handle{};

            if (pack != nullptr) {
                auto future = thread_pool().submit_task(
                        [pack = std::move(pack), id = reference.id,
                         &sampler_storage =
                                 renderer.sampler_storage()]() -> std::expected<ModelCpuData, ModelLoadError> {
                            auto cpu_data = pack->load_model(id, sampler_storage);

                            if (!cpu_data) {
                                return std::unexpected(ModelLoadError{
                                        .type = ModelLoadErrorType::parse_error,
                                        .cause = ErrorCause{Boxed<LbfError>{cpu_data.error()}},
                                });
                            }

                            return std::move(*cpu_data);
                        });

                handle = renderer.model_streamer().request_prepared(renderer, std::move(future), reference.source,
                                                                    engine_models.cube, debug_name);
                ++report.models_from_packs;
            } else {
                handle = renderer.model_streamer().request(renderer, reference.source, engine_models.cube, debug_name);
                ++report.models_from_source;
            }

            models[index] = ResolvedModel{.handle = handle, .owned = true};
            report.streaming_models.push_back(handle);
            continue;
        }

        if (pack != nullptr) {
            decodes.push_back(PendingDecode{
                    .index = index,
                    .future = thread_pool().submit_task([pack = std::move(pack), id = reference.id, &renderer] {
                        return pack->load_model(id, renderer.sampler_storage());
                    }),
            });
        }
    }

    for (auto &decode: decodes) {
        auto const &reference = description.models[decode.index];
        auto cpu_data = decode.future.get();

        if (!cpu_data) {
            report.warnings.push_back(std::format("cooked model '{}' is unusable ({}); loading its source",
                                                  reference.source, describe(cpu_data.error())));
            continue;
        }

        auto handle = renderer.create_model_from_cpu_data(*cpu_data);

        if (!handle) {
            report.warnings.push_back(
                    std::format("could not upload model '{}': {}", reference.source, describe(handle.error())));
            continue;
        }

        renderer.register_model_source(*handle, reference.source);
        renderer.register_model_name(*handle, std::filesystem::path{reference.source}.filename().string());

        models[decode.index] = ResolvedModel{.handle = *handle, .owned = true};
        ++report.models_from_packs;
    }

    for (std::size_t index = 0; index < description.models.size(); ++index) {
        auto &model = models[index];

        if (model.handle.valid() || description.models[index].source.starts_with(engine_asset_prefix)) {
            continue;
        }

        auto const &source = description.models[index].source;
        auto handle = renderer.load_model(source);

        if (!handle) {
            report.warnings.push_back(std::format("could not load model '{}' ({}); using the engine cube", source,
                                                  describe(handle.error())));
            model.handle = engine_models.cube;
            continue;
        }

        model = ResolvedModel{.handle = *handle, .owned = true};
        ++report.models_from_source;
    }

    // ---- textures and materials.
    auto &images = renderer.image_storage();
    std::vector<std::optional<ImageHandle>> textures(description.textures.size());

    auto const resolve_texture = [&](std::uint32_t texture_index, std::size_t slot) -> ImageHandle {
        auto const fallback = default_texture_for_slot(images, slot);

        if (texture_index == scene_no_index) {
            return fallback;
        }

        if (textures[texture_index].has_value()) {
            return *textures[texture_index];
        }

        auto const &reference = description.textures[texture_index];
        auto const debug_name = std::filesystem::path{reference.source}.filename().string();
        ImageHandle handle{};

        if (auto const pack = find_pack(packs, lbf_chunk::texture, reference.id)) {
            handle = renderer.texture_streamer().request_cooked(images, pack->texture_loader(reference.id),
                                                                AssetPack::texture_cache_key(reference.id), fallback,
                                                                debug_name);
            // So saving again references the source file rather than an anonymous cooked image.
            renderer.texture_streamer().set_source(handle, TextureStreamer::Source{
                                                                   .path = reference.source,
                                                                   .role = reference.role,
                                                           });
            static_cast<void>(renderer.assets().textures().register_asset(debug_name, handle));
            ++report.textures_from_packs;
        } else {
            handle = renderer.request_texture(reference.source, reference.role, fallback, debug_name);
            ++report.textures_from_source;
        }

        textures[texture_index] = handle;
        return handle;
    };

    std::vector<MaterialHandle> materials(description.materials.size());
    std::vector<MaterialHandle> anonymous_materials;

    for (std::size_t index = 0; index < description.materials.size(); ++index) {
        auto const &material = description.materials[index];

        MaterialCreateInfo info{
                .base_colour_factor = material.base_colour_factor,
                .emissive_factor = material.emissive_factor,
                .emissive_strength = material.emissive_strength,
                .metallic_factor = material.metallic_factor,
                .roughness_factor = material.roughness_factor,
                .normal_scale = material.normal_scale,
                .occlusion_strength = material.occlusion_strength,
                .alpha_cutoff = material.alpha_cutoff,
                .base_colour_texture = resolve_texture(material.textures[scene_material_texture::base_colour],
                                                       scene_material_texture::base_colour),
                .normal_texture = resolve_texture(material.textures[scene_material_texture::normal],
                                                  scene_material_texture::normal),
                .metallic_roughness_texture =
                        resolve_texture(material.textures[scene_material_texture::metallic_roughness],
                                        scene_material_texture::metallic_roughness),
                .occlusion_texture = resolve_texture(material.textures[scene_material_texture::occlusion],
                                                     scene_material_texture::occlusion),
                .emissive_texture = resolve_texture(material.textures[scene_material_texture::emissive],
                                                    scene_material_texture::emissive),
                .sampler = sampler_for(renderer.sampler_storage(), material.sampler),
                .alpha_mode = material.alpha_mode,
                .wind_strength = material.wind_strength,
                .max_shadow_cascade = material.max_shadow_cascade,
                .debug_meshlet_colours = material.debug_meshlet_colours,
        };

        // A named material is an asset: reloading a scene updates it in place instead of piling up copies.
        if (!material.name.empty()) {
            if (auto const existing = renderer.assets().materials().find(material.name); existing.valid()) {
                if (renderer.update_material(existing, info)) {
                    materials[index] = existing;
                    continue;
                }
            }
        }

        auto created = renderer.create_material(info, material.name);

        if (!created) {
            report.warnings.push_back(
                    std::format("could not create material '{}': {}", material.name, describe(created.error())));
            continue;
        }

        materials[index] = *created;

        if (material.name.empty()) {
            anonymous_materials.push_back(*created);
        }
    }

    auto const material_at = [&](std::uint32_t index) -> MaterialHandle {
        return index == scene_no_index ? MaterialHandle{} : materials[index];
    };

    // ---- entities.
    if (auto idle = renderer.wait_idle(); !idle) {
        return std::unexpected(make_error(LbfErrorType::instantiate_failed, describe(idle.error())));
    }

    auto &registry = scene.get_registry();
    registry.clear();
    scene.physics_settings = description.physics_settings;

    std::vector<entt::entity> entities;
    entities.reserve(description.entities.size());

    for (auto const &record: description.entities) {
        if (has_flag(record.flags, SceneEntityFlags::generated_name)) {
            entities.push_back(GeneratedEntity{&scene, std::string_view{record.name}});
        } else {
            entities.push_back(Entity{&scene, std::string_view{record.name}});
        }
    }

    for (std::size_t index = 0; index < description.entities.size(); ++index) {
        auto const &record = description.entities[index];
        auto const entity = entities[index];

        if (record.transform.has_value()) {
            registry.emplace<Components::Transform>(entity, *record.transform);
        }

        if (record.parent != scene_no_index) {
            registry.emplace<Components::Parent>(entity, Components::Parent{.entity = entities[record.parent]});
        }

        if (has_flag(record.flags, SceneEntityFlags::player)) {
            registry.emplace<Components::PlayerTag>(entity);
        }

        if (has_flag(record.flags, SceneEntityFlags::bullet)) {
            registry.emplace<Components::BulletTag>(entity);
        }
    }

    for (auto const &component: description.model_components) {
        auto const entity = entities[component.entity];
        auto const handle = models[component.model].handle;

        if (!handle.valid()) {
            continue;
        }

        registry.emplace<Components::Model>(entity, Components::Model{.model = handle});

        // StreamedModelTag entities own a model reference, which Scene releases when the entity or tag goes.
        if (has_flag(description.entities[component.entity].flags, SceneEntityFlags::streamed_model)) {
            renderer.retain_model(handle);
            registry.emplace<Components::StreamedModelTag>(entity);
        }
    }

    auto const model_streaming = [&](ModelHandle handle) {
        return options.stream_models && renderer.model_streamer().state(handle) == ModelRequestState::loading;
    };

    for (auto const &component: description.material_overrides) {
        Components::MaterialOverride material_override{.material = material_at(component.material)};
        auto const *entity_model = registry.try_get<Components::Model>(entities[component.entity]);

        // The model's own materials don't exist until it installs; finish this override then.
        if (!component.slots.empty() && entity_model != nullptr && model_streaming(entity_model->model)) {
            DeferredSlotOverride deferred{.entity = entities[component.entity], .model = entity_model->model};

            for (auto const &slot: component.slots) {
                auto const material = material_at(slot.material);
                renderer.retain_material(material);
                deferred.slots.emplace_back(slot.source_slot, material);
            }

            report.deferred_slot_overrides.push_back(std::move(deferred));
        } else if (!component.slots.empty()) {
            auto const model_materials = entity_model != nullptr ? renderer.model_materials(entity_model->model)
                                                                 : std::vector<MaterialHandle>{};

            for (auto const &slot: component.slots) {
                if (slot.source_slot >= model_materials.size()) {
                    report.warnings.emplace_back("a per-slot material override doesn't match its model; skipped");
                    continue;
                }

                material_override.slots.push_back(MaterialSlotOverride{
                        .source = model_materials[slot.source_slot],
                        .material = material_at(slot.material),
                });
            }
        }

        scene.set_material_override(entities[component.entity], std::move(material_override));
    }

    for (auto const &component: description.instanced_models) {
        if (auto const handle = models[component.model].handle; handle.valid()) {
            registry.emplace<Components::InstancedModel>(entities[component.entity],
                                                         Components::InstancedModel{
                                                                 .model = handle,
                                                                 .material_override = material_at(component.material),
                                                                 .transforms = component.transforms,
                                                         });
        }
    }

    for (auto const &component: description.point_lights) {
        registry.emplace<Components::PointLight>(entities[component.entity], component.light);
    }

    for (auto const &component: description.spot_lights) {
        registry.emplace<Components::SpotLight>(entities[component.entity], component.light);
    }

    for (auto const &component: description.rigid_bodies) {
        registry.emplace<Components::RigidBody>(entities[component.entity], component.body);
    }

    for (auto const &component: description.scripts) {
        auto const handle = renderer.assets().scripts().find(component.script);

        if (!handle.valid()) {
            report.warnings.push_back(
                    std::format("script '{}' isn't registered in this build; skipped", component.script));
            continue;
        }

        registry.emplace<Components::Script>(entities[component.entity], Components::Script{.script = handle});
    }

    for (auto const &component: description.lifetimes) {
        registry.emplace<Components::Lifetime>(entities[component.entity],
                                               Components::Lifetime{.remaining_seconds = component.remaining_seconds});
    }

    // Overrides now hold their own references; drop the creation reference on unnamed materials so they die with
    // the last entity using them. Named ones stay, owned by their name.
    for (auto const material: anonymous_materials) {
        renderer.release_material(material);
    }

    return report;
}

auto apply_deferred_slot_overrides(Scene &scene, Renderer &renderer, std::vector<DeferredSlotOverride> &pending)
        -> bool {
    auto &registry = scene.get_registry();

    std::erase_if(pending, [&](DeferredSlotOverride &deferred) {
        if (renderer.model_streamer().state(deferred.model) == ModelRequestState::loading) {
            return false;
        }

        auto const *model =
                registry.valid(deferred.entity) ? registry.try_get<Components::Model>(deferred.entity) : nullptr;

        // Skipped if the entity went away or was given another model in the meantime.
        if (model != nullptr && model->model == deferred.model) {
            auto const model_materials = renderer.model_materials(deferred.model);
            auto material_override = registry.all_of<Components::MaterialOverride>(deferred.entity)
                                             ? registry.get<Components::MaterialOverride>(deferred.entity)
                                             : Components::MaterialOverride{};

            for (auto const &[source_slot, material]: deferred.slots) {
                if (source_slot < model_materials.size()) {
                    material_override.slots.push_back(MaterialSlotOverride{
                            .source = model_materials[source_slot],
                            .material = material,
                    });
                } else {
                    warn("load_scene: a per-slot material override doesn't match its model; skipped");
                }
            }

            scene.set_material_override(deferred.entity, std::move(material_override));
        }

        for (auto const &slot: deferred.slots) {
            renderer.release_material(slot.second);
        }

        return true;
    });

    return pending.empty();
}

namespace {

    // The part of a save that doesn't touch the Renderer beyond its SamplerStorage's fixed default handles, so it can
    // run on a background thread.
    auto write_scene_file(SceneDescription const &description, std::vector<std::string> warnings,
                          SamplerStorage &sampler_storage, std::filesystem::path const &path,
                          SceneSaveOptions const &options, std::chrono::steady_clock::time_point start)
            -> std::expected<SceneSaveResult, LbfError>;

} // namespace

auto save_scene(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                std::filesystem::path const &path, SceneSaveOptions const &options)
        -> std::expected<SceneSaveResult, LbfError> {
    ZoneScopedNC("save_scene", tracy::Color::Goldenrod);

    auto const start = std::chrono::steady_clock::now();

    SceneCaptureReport capture_report;
    auto const description = capture_scene(scene, renderer, engine_models, &capture_report);

    return write_scene_file(description, std::move(capture_report.warnings), renderer.sampler_storage(), path, options,
                            start);
}

auto SceneSaveJob::start(Scene const &scene, Renderer &renderer, EngineModels const &engine_models,
                         std::filesystem::path path, SceneSaveOptions options) -> SceneSaveJob {
    auto const started_at = std::chrono::steady_clock::now();

    SceneCaptureReport capture_report;
    auto description = capture_scene(scene, renderer, engine_models, &capture_report);

    SceneSaveJob job;
    job.path_ = path;
    job.fingerprint_ = scene_fingerprint(description);

    // std::async rather than thread_pool(): cooking fans out to the pool and waits on it, which would deadlock from
    // inside a pool worker.
    job.future_ = std::async(std::launch::async, [description = std::move(description),
                                                  warnings = std::move(capture_report.warnings),
                                                  &sampler_storage = renderer.sampler_storage(), path = std::move(path),
                                                  options = std::move(options), started_at]() mutable {
        return write_scene_file(description, std::move(warnings), sampler_storage, path, options, started_at);
    });

    return job;
}

auto SceneSaveJob::ready() const -> bool {
    return future_.valid() && future_.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

auto SceneSaveJob::take() -> std::expected<SceneSaveResult, LbfError> {
    if (!future_.valid()) {
        return std::unexpected(make_error(LbfErrorType::io_failed, "save job already taken"));
    }

    return future_.get();
}

namespace {

    auto write_scene_file(SceneDescription const &description, std::vector<std::string> warnings,
                          SamplerStorage &sampler_storage, std::filesystem::path const &path,
                          SceneSaveOptions const &options, std::chrono::steady_clock::time_point start)
            -> std::expected<SceneSaveResult, LbfError> {
        ZoneScopedNC("write_scene_file", tracy::Color::Goldenrod);

        SceneSaveResult result;
        result.warnings = std::move(warnings);
        result.fingerprint = scene_fingerprint(description);

        LbfWriter writer{LbfFileKind::scene};

        writer.add_chunk(LbfChunkInput{
                .type = lbf_chunk::scene,
                .version = scene_chunk_version,
                .payload = encode_scene(description),
        });

        // Which engine and format versions wrote this file; informational, never needed to read it.
        {
            auto const metadata = std::format("writer=lathe;lbf={}.{};scene={};section={};model={};texture={}",
                                              lbf_version_major, lbf_version_minor, scene_chunk_version,
                                              scene_section_version, cooked_model_version, cooked_texture_version);
            auto const bytes = std::as_bytes(std::span<char const>{metadata.data(), metadata.size()});

            writer.add_chunk(LbfChunkInput{
                    .type = lbf_chunk::metadata,
                    .version = 1,
                    .payload = {bytes.begin(), bytes.end()},
                    .compress = false,
            });
        }

        if (options.embed_assets) {
            AssetCookRequest request;

            for (auto const &model: description.models) {
                if (!model.source.starts_with(engine_asset_prefix)) {
                    request.models.emplace_back(model.source);
                }
            }

            for (auto const &texture: description.textures) {
                request.textures.push_back(AssetCookRequest::Texture{.path = texture.source, .role = texture.role});
            }

            result.cook = cook_assets(request, sampler_storage, writer,
                                      AssetCookOptions{.source_packs = options.source_packs});
        }

        auto written = writer.write_file(path, options.write);

        if (!written) {
            return std::unexpected(written.error());
        }

        result.file_size = *written;
        result.seconds = seconds_since(start);

        info("save_scene: wrote '{}' ({:.1f} MiB, {} entities, {} models cooked / {} copied, {} textures cooked / {} "
             "copied) in {:.2f}s",
             path.string(), static_cast<double>(result.file_size) / (1024.0 * 1024.0), description.entities.size(),
             result.cook.models_cooked, result.cook.models_copied, result.cook.textures_cooked,
             result.cook.textures_copied, result.seconds);

        for (auto const &warning: result.warnings) {
            warn("save_scene: {}", warning);
        }

        for (auto const &failure: result.cook.failures) {
            warn("save_scene: not embedded, will load from source: {}", failure);
        }

        return result;
    }

} // namespace

namespace {

    struct ReadScene {
        std::shared_ptr<AssetPack> pack;
        SceneDescription description;
        SceneDecodeReport decode;
    };

    // File and CPU work only, so it can run off the render thread.
    auto read_scene_file(std::filesystem::path const &path) -> std::expected<ReadScene, LbfError> {
        ZoneScopedNC("read_scene_file", tracy::Color::Goldenrod);

        auto pack = AssetPack::open(path);

        if (!pack) {
            return std::unexpected(pack.error());
        }

        auto const &reader = (*pack)->reader();

        if (reader.header().kind != LbfFileKind::scene) {
            return std::unexpected(
                    make_error(LbfErrorType::wrong_file_kind, "this .lbf is an asset pack, not a scene"));
        }

        auto const *entry = reader.find(lbf_chunk::scene);

        if (entry == nullptr) {
            return std::unexpected(make_error(LbfErrorType::chunk_not_found, "no SCEN chunk"));
        }

        auto payload = reader.read_chunk(*entry);

        if (!payload) {
            return std::unexpected(payload.error());
        }

        ReadScene read{.pack = std::move(*pack)};
        auto description = decode_scene(*payload, entry->version, &read.decode);

        if (!description) {
            return std::unexpected(description.error());
        }

        read.description = std::move(*description);
        return read;
    }

    auto log_load(std::filesystem::path const &path, SceneLoadResult const &result, std::size_t entity_count) -> void {
        info("load_scene: '{}' -> {} entities in {:.2f}s (models: {} cooked, {} from source, {} reused; textures: {} "
             "cooked, {} from source)",
             path.string(), entity_count, result.seconds, result.instantiate.models_from_packs,
             result.instantiate.models_from_source, result.instantiate.models_reused,
             result.instantiate.textures_from_packs, result.instantiate.textures_from_source);

        if (result.decode.skipped_sections > 0) {
            warn("load_scene: skipped {} section(s) written by a newer engine", result.decode.skipped_sections);
        }

        for (auto const &warning: result.instantiate.warnings) {
            warn("load_scene: {}", warning);
        }
    }

} // namespace

auto SceneLoadJob::start(std::filesystem::path path) -> SceneLoadJob {
    SceneLoadJob job;
    job.path_ = path;
    job.started_at_ = std::chrono::steady_clock::now();
    job.future_ = std::async(std::launch::async, [path = std::move(path)]() -> std::expected<Prepared, LbfError> {
        auto read = read_scene_file(path);

        if (!read) {
            return std::unexpected(read.error());
        }

        return Prepared{
                .pack = std::move(read->pack), .description = std::move(read->description), .decode = read->decode};
    });

    return job;
}

auto SceneLoadJob::models_remaining(Renderer &renderer) const -> std::size_t {
    return static_cast<std::size_t>(std::ranges::count_if(streaming_, [&](ModelHandle handle) {
        return renderer.model_streamer().state(handle) == ModelRequestState::loading;
    }));
}

auto SceneLoadJob::step(Scene &scene, Renderer &renderer, EngineModels const &engine_models)
        -> std::optional<std::expected<SceneLoadResult, LbfError>> {
    if (phase_ == Phase::done) {
        return std::nullopt;
    }

    if (phase_ == Phase::reading) {
        if (future_.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
            return std::nullopt;
        }

        auto prepared = future_.get();

        if (!prepared) {
            phase_ = Phase::done;
            return std::unexpected(prepared.error());
        }

        auto instantiated =
                instantiate_scene(scene, renderer, engine_models, prepared->description,
                                  SceneInstantiateOptions{.packs = {prepared->pack}, .stream_models = true});

        if (!instantiated) {
            phase_ = Phase::done;
            return std::unexpected(instantiated.error());
        }

        streaming_ = std::move(instantiated->streaming_models);
        deferred_ = std::move(instantiated->deferred_slot_overrides);

        result_ = SceneLoadResult{
                .pack = std::move(prepared->pack),
                .instantiate = std::move(*instantiated),
                .decode = prepared->decode,
        };

        log_load(path_, *result_, prepared->description.entities.size());
        phase_ = Phase::streaming;
    }

    auto const overrides_done = apply_deferred_slot_overrides(scene, renderer, deferred_);

    if (!overrides_done || models_remaining(renderer) > 0) {
        return std::nullopt;
    }

    phase_ = Phase::done;
    result_->seconds = seconds_since(started_at_);
    info("load_scene: '{}' finished streaming in {:.2f}s", path_.string(), result_->seconds);

    return std::move(*result_);
}

auto load_scene(Scene &scene, Renderer &renderer, EngineModels const &engine_models, std::filesystem::path const &path,
                SceneLoadOptions const &options) -> std::expected<SceneLoadResult, LbfError> {
    ZoneScopedNC("load_scene", tracy::Color::Goldenrod);

    auto const start = std::chrono::steady_clock::now();
    auto read = read_scene_file(path);

    if (!read) {
        return std::unexpected(read.error());
    }

    auto instantiated = instantiate_scene(scene, renderer, engine_models, read->description,
                                          SceneInstantiateOptions{.packs = {read->pack}});

    if (!instantiated) {
        return std::unexpected(instantiated.error());
    }

    if (options.wait_for_textures) {
        auto const installed = renderer.texture_streamer().flush(renderer.image_storage(), renderer.context());
        debug("load_scene: uploaded {} textures before returning", installed);
    }

    SceneLoadResult result{
            .pack = std::move(read->pack),
            .instantiate = std::move(*instantiated),
            .decode = read->decode,
            .seconds = seconds_since(start),
    };

    log_load(path, result, read->description.entities.size());

    return result;
}

auto scene_fingerprint(SceneDescription const &description) -> std::uint64_t {
    ZoneScopedNC("scene_fingerprint", tracy::Color::Goldenrod);

    // Each entity is hashed as a one-entity scene holding its own components and copies of what they reference (and
    // its parent's own data), so the result doesn't depend on entity, model or material order. The sorted per-entity
    // hashes are then hashed together.
    auto const entity_count = description.entities.size();
    std::vector<SceneDescription> singles(entity_count);

    auto const add_model = [&](SceneDescription &single, std::uint32_t model) {
        single.models.push_back(description.models[model]);
        return static_cast<std::uint32_t>(single.models.size() - 1);
    };

    auto const add_material = [&](SceneDescription &single, std::uint32_t material) {
        if (material == scene_no_index) {
            return scene_no_index;
        }

        auto copy = description.materials[material];

        for (auto &texture: copy.textures) {
            if (texture != scene_no_index) {
                single.textures.push_back(description.textures[texture]);
                texture = static_cast<std::uint32_t>(single.textures.size() - 1);
            }
        }

        single.materials.push_back(std::move(copy));
        return static_cast<std::uint32_t>(single.materials.size() - 1);
    };

    for (std::size_t index = 0; index < entity_count; ++index) {
        auto entity = description.entities[index];
        auto &single = singles[index];

        if (entity.parent != scene_no_index) {
            auto parent = description.entities[entity.parent];
            parent.parent = scene_no_index;
            entity.parent = 1;
            single.entities.push_back(std::move(entity));
            single.entities.push_back(std::move(parent));
        } else {
            single.entities.push_back(std::move(entity));
        }
    }

    for (auto component: description.model_components) {
        auto &single = singles[component.entity];
        component.model = add_model(single, component.model);
        component.entity = 0;
        single.model_components.push_back(component);
    }

    for (auto component: description.material_overrides) {
        auto &single = singles[component.entity];
        component.material = add_material(single, component.material);

        for (auto &slot: component.slots) {
            slot.material = add_material(single, slot.material);
        }

        component.entity = 0;
        single.material_overrides.push_back(std::move(component));
    }

    for (auto component: description.instanced_models) {
        auto &single = singles[component.entity];
        component.model = add_model(single, component.model);
        component.material = add_material(single, component.material);
        component.entity = 0;
        single.instanced_models.push_back(std::move(component));
    }

    auto const add_plain = [&](auto const &components, auto member) {
        for (auto component: components) {
            auto &single = singles[component.entity];
            component.entity = 0;
            (single.*member).push_back(std::move(component));
        }
    };

    add_plain(description.point_lights, &SceneDescription::point_lights);
    add_plain(description.spot_lights, &SceneDescription::spot_lights);
    add_plain(description.rigid_bodies, &SceneDescription::rigid_bodies);
    add_plain(description.scripts, &SceneDescription::scripts);
    add_plain(description.lifetimes, &SceneDescription::lifetimes);

    std::vector<std::uint64_t> hashes;
    hashes.reserve(entity_count + 1);

    for (auto const &single: singles) {
        hashes.push_back(xxh64(encode_scene(single)));
    }

    std::ranges::sort(hashes);

    SceneDescription settings_only;
    settings_only.physics_settings = description.physics_settings;
    hashes.push_back(xxh64(encode_scene(settings_only)));

    return xxh64(std::as_bytes(std::span<std::uint64_t const>{hashes}));
}
