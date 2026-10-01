#include "serialisation/scene_codec.hxx"

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <source_location>
#include <utility>

#include "serialisation/byte_stream.hxx"

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

    // ---- field helpers -------------------------------------------------------------------------------------------

    auto write_vec3(ByteWriter &writer, glm::vec3 const &value) -> void {
        writer.write(value.x);
        writer.write(value.y);
        writer.write(value.z);
    }

    auto write_vec4(ByteWriter &writer, glm::vec4 const &value) -> void {
        writer.write(value.x);
        writer.write(value.y);
        writer.write(value.z);
        writer.write(value.w);
    }

    [[nodiscard]] auto read_vec3(ByteReader &reader) -> glm::vec3 {
        glm::vec3 value{};
        reader.read(value.x);
        reader.read(value.y);
        reader.read(value.z);
        return value;
    }

    [[nodiscard]] auto read_vec4(ByteReader &reader) -> glm::vec4 {
        glm::vec4 value{};
        reader.read(value.x);
        reader.read(value.y);
        reader.read(value.z);
        reader.read(value.w);
        return value;
    }

    auto write_transform(ByteWriter &writer, Components::Transform const &transform) -> void {
        write_vec3(writer, transform.position);
        writer.write(transform.rotation.w);
        writer.write(transform.rotation.x);
        writer.write(transform.rotation.y);
        writer.write(transform.rotation.z);
        write_vec3(writer, transform.scale);
    }

    [[nodiscard]] auto read_transform(ByteReader &reader) -> Components::Transform {
        Components::Transform transform;
        transform.position = read_vec3(reader);
        reader.read(transform.rotation.w);
        reader.read(transform.rotation.x);
        reader.read(transform.rotation.y);
        reader.read(transform.rotation.z);
        transform.scale = read_vec3(reader);
        return transform;
    }

    auto write_bool(ByteWriter &writer, bool value) -> void { writer.write(static_cast<std::uint8_t>(value ? 1 : 0)); }

    [[nodiscard]] auto read_bool(ByteReader &reader) -> bool { return reader.read<std::uint8_t>() != 0; }

    // A record count, bounded by the bytes left so a corrupt count can't trigger a huge allocation.
    [[nodiscard]] auto read_count(ByteReader &reader, std::size_t minimum_record_size) -> std::uint32_t {
        auto const count = reader.read<std::uint32_t>();

        if (reader.failed() || count > reader.remaining() / minimum_record_size) {
            reader.fail();
            return 0;
        }

        return count;
    }

    // ---- sections ------------------------------------------------------------------------------------------------
    //
    // Each section's writer emits the current version; its reader takes the version it was written with. When a
    // layout changes: bump scene_section_version (or give that section its own constant), branch on `version` in the
    // reader, and keep the old branch.

    auto write_settings(ByteWriter &writer, SceneDescription const &scene) -> void {
        write_vec3(writer, scene.physics_settings.gravity);
        writer.write(scene.physics_settings.ground_y);
    }

    auto read_settings(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.physics_settings.gravity = read_vec3(reader);
        reader.read(scene.physics_settings.ground_y);
    }

    auto write_models(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.models.size()));

        for (auto const &model: scene.models) {
            writer.write(model.id.value);
            writer.write_string(model.source);
        }
    }

    auto read_models(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.models.resize(read_count(reader, 12));

        for (auto &model: scene.models) {
            model.id.value = reader.read<std::uint64_t>();
            reader.read_string(model.source);
        }
    }

    auto write_textures(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.textures.size()));

        for (auto const &texture: scene.textures) {
            writer.write(texture.id.value);
            writer.write_string(texture.source);
            writer.write(std::to_underlying(texture.role));
        }
    }

    auto read_textures(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.textures.resize(read_count(reader, 13));

        for (auto &texture: scene.textures) {
            texture.id.value = reader.read<std::uint64_t>();
            reader.read_string(texture.source);

            auto const role = reader.read<std::uint8_t>();

            if (role > std::to_underlying(TextureRole::normal_map)) {
                reader.fail();
            }

            texture.role = static_cast<TextureRole>(role);
        }
    }

    auto write_materials(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.materials.size()));

        for (auto const &material: scene.materials) {
            writer.write_string(material.name);
            write_vec4(writer, material.base_colour_factor);
            write_vec3(writer, material.emissive_factor);
            writer.write(material.emissive_strength);
            writer.write(material.metallic_factor);
            writer.write(material.roughness_factor);
            writer.write(material.normal_scale);
            writer.write(material.occlusion_strength);
            writer.write(material.alpha_cutoff);
            writer.write(material.wind_strength);
            writer.write(material.max_shadow_cascade);
            writer.write(std::to_underlying(material.alpha_mode));
            writer.write(std::to_underlying(material.sampler));
            write_bool(writer, material.debug_meshlet_colours);

            for (auto const texture: material.textures) {
                writer.write(texture);
            }
        }
    }

    auto read_materials(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.materials.resize(read_count(reader, 64));

        for (auto &material: scene.materials) {
            reader.read_string(material.name);
            material.base_colour_factor = read_vec4(reader);
            material.emissive_factor = read_vec3(reader);
            reader.read(material.emissive_strength);
            reader.read(material.metallic_factor);
            reader.read(material.roughness_factor);
            reader.read(material.normal_scale);
            reader.read(material.occlusion_strength);
            reader.read(material.alpha_cutoff);
            reader.read(material.wind_strength);
            reader.read(material.max_shadow_cascade);

            auto const alpha_mode = reader.read<std::uint32_t>();
            auto const sampler = reader.read<std::uint8_t>();

            if (alpha_mode > std::to_underlying(AlphaMode::blend) ||
                sampler > std::to_underlying(DefaultSampler::nearest_clamp)) {
                reader.fail();
            }

            material.alpha_mode = static_cast<AlphaMode>(alpha_mode);
            material.sampler = static_cast<DefaultSampler>(sampler);
            material.debug_meshlet_colours = read_bool(reader);

            for (auto &texture: material.textures) {
                reader.read(texture);
            }
        }
    }

    auto write_entities(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.entities.size()));

        for (auto const &entity: scene.entities) {
            writer.write_string(entity.name);
            writer.write(entity.parent);
            writer.write(static_cast<std::uint32_t>(entity.flags));
            write_bool(writer, entity.transform.has_value());

            if (entity.transform.has_value()) {
                write_transform(writer, *entity.transform);
            }
        }
    }

    auto read_entities(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.entities.resize(read_count(reader, 13));

        for (auto &entity: scene.entities) {
            reader.read_string(entity.name);
            reader.read(entity.parent);
            entity.flags = static_cast<SceneEntityFlags>(reader.read<std::uint32_t>());

            if (read_bool(reader)) {
                entity.transform = read_transform(reader);
            }
        }
    }

    auto write_model_components(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.model_components.size()));

        for (auto const &component: scene.model_components) {
            writer.write(component.entity);
            writer.write(component.model);
        }
    }

    auto read_model_components(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.model_components.resize(read_count(reader, 8));

        for (auto &component: scene.model_components) {
            reader.read(component.entity);
            reader.read(component.model);
        }
    }

    auto write_material_overrides(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.material_overrides.size()));

        for (auto const &component: scene.material_overrides) {
            writer.write(component.entity);
            writer.write(component.material);
            writer.write(static_cast<std::uint32_t>(component.slots.size()));

            for (auto const &slot: component.slots) {
                writer.write(slot.source_slot);
                writer.write(slot.material);
            }
        }
    }

    auto read_material_overrides(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.material_overrides.resize(read_count(reader, 12));

        for (auto &component: scene.material_overrides) {
            reader.read(component.entity);
            reader.read(component.material);
            component.slots.resize(read_count(reader, 8));

            for (auto &slot: component.slots) {
                reader.read(slot.source_slot);
                reader.read(slot.material);
            }
        }
    }

    auto write_instanced_models(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.instanced_models.size()));

        for (auto const &component: scene.instanced_models) {
            writer.write(component.entity);
            writer.write(component.model);
            writer.write(component.material);
            writer.write(static_cast<std::uint32_t>(component.transforms.size()));

            for (auto const &transform: component.transforms) {
                writer.write_span(std::span<float const>{&transform[0][0], 16});
            }
        }
    }

    auto read_instanced_models(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.instanced_models.resize(read_count(reader, 16));

        for (auto &component: scene.instanced_models) {
            reader.read(component.entity);
            reader.read(component.model);
            reader.read(component.material);
            component.transforms.resize(read_count(reader, sizeof(glm::mat4)));

            for (auto &transform: component.transforms) {
                reader.read_span(std::span<float>{&transform[0][0], 16});
            }
        }
    }

    auto write_point_lights(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.point_lights.size()));

        for (auto const &component: scene.point_lights) {
            writer.write(component.entity);
            write_vec3(writer, component.light.colour);
            writer.write(component.light.intensity);
            writer.write(component.light.range);
        }
    }

    auto read_point_lights(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.point_lights.resize(read_count(reader, 24));

        for (auto &component: scene.point_lights) {
            reader.read(component.entity);
            component.light.colour = read_vec3(reader);
            reader.read(component.light.intensity);
            reader.read(component.light.range);
        }
    }

    auto write_spot_lights(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.spot_lights.size()));

        for (auto const &component: scene.spot_lights) {
            writer.write(component.entity);
            write_vec3(writer, component.light.colour);
            writer.write(component.light.intensity);
            writer.write(component.light.range);
            writer.write(component.light.inner_cone_degrees);
            writer.write(component.light.outer_cone_degrees);
        }
    }

    auto read_spot_lights(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.spot_lights.resize(read_count(reader, 32));

        for (auto &component: scene.spot_lights) {
            reader.read(component.entity);
            component.light.colour = read_vec3(reader);
            reader.read(component.light.intensity);
            reader.read(component.light.range);
            reader.read(component.light.inner_cone_degrees);
            reader.read(component.light.outer_cone_degrees);
        }
    }

    auto write_rigid_bodies(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.rigid_bodies.size()));

        for (auto const &component: scene.rigid_bodies) {
            auto const &body = component.body;

            writer.write(component.entity);
            write_vec3(writer, body.velocity);
            write_vec3(writer, body.half_extents);
            writer.write(body.capsule_radius);
            writer.write(body.capsule_height);
            writer.write(body.restitution);
            writer.write(body.mass);
            write_bool(writer, body.is_static);
            write_bool(writer, body.lock_rotation);
            writer.write(std::to_underlying(body.shape));

            auto const box_count = body.compound_boxes != nullptr ? body.compound_boxes->size() : 0;
            writer.write(static_cast<std::uint32_t>(box_count));

            for (std::size_t index = 0; index < box_count; ++index) {
                write_vec3(writer, (*body.compound_boxes)[index].local_centre);
                write_vec3(writer, (*body.compound_boxes)[index].half_extents);
            }
        }
    }

    auto read_rigid_bodies(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.rigid_bodies.resize(read_count(reader, 48));

        for (auto &component: scene.rigid_bodies) {
            auto &body = component.body;

            reader.read(component.entity);
            body.velocity = read_vec3(reader);
            body.half_extents = read_vec3(reader);
            reader.read(body.capsule_radius);
            reader.read(body.capsule_height);
            reader.read(body.restitution);
            reader.read(body.mass);
            body.is_static = read_bool(reader);
            body.lock_rotation = read_bool(reader);

            auto const shape = reader.read<std::uint8_t>();

            // Heightfields are never written; see SceneRigidBodyComponent.
            if (shape > std::to_underlying(Components::BodyShape::compound) ||
                shape == std::to_underlying(Components::BodyShape::heightfield)) {
                reader.fail();
            }

            body.shape = static_cast<Components::BodyShape>(shape);

            auto const box_count = read_count(reader, 24);

            if (box_count > 0) {
                auto boxes = std::make_shared<std::vector<Components::CompoundBoxChild>>(box_count);

                for (auto &box: *boxes) {
                    box.local_centre = read_vec3(reader);
                    box.half_extents = read_vec3(reader);
                }

                body.compound_boxes = std::move(boxes);
            }
        }
    }

    auto write_scripts(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.scripts.size()));

        for (auto const &component: scene.scripts) {
            writer.write(component.entity);
            writer.write_string(component.script);
        }
    }

    auto read_scripts(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.scripts.resize(read_count(reader, 8));

        for (auto &component: scene.scripts) {
            reader.read(component.entity);
            reader.read_string(component.script);
        }
    }

    auto write_lifetimes(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.lifetimes.size()));

        for (auto const &component: scene.lifetimes) {
            writer.write(component.entity);
            writer.write(component.remaining_seconds);
        }
    }

    auto read_lifetimes(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        scene.lifetimes.resize(read_count(reader, 8));

        for (auto &component: scene.lifetimes) {
            reader.read(component.entity);
            reader.read(component.remaining_seconds);
        }
    }

    // ---- section table -------------------------------------------------------------------------------------------

    using SectionWriter = void (*)(ByteWriter &, SceneDescription const &);
    using SectionReader = void (*)(ByteReader &, std::uint16_t, SceneDescription &);

    struct SectionCodec {
        std::uint32_t type;
        std::uint16_t version; // written
        std::uint16_t oldest_readable;
        SectionWriter write;
        SectionReader read;
    };

    // Order matters only for writing; readers accept sections in any order.
    constexpr std::array section_codecs{
            SectionCodec{.type = scene_section::settings,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_settings,
                         .read = read_settings},
            SectionCodec{.type = scene_section::models,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_models,
                         .read = read_models},
            SectionCodec{.type = scene_section::textures,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_textures,
                         .read = read_textures},
            SectionCodec{.type = scene_section::materials,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_materials,
                         .read = read_materials},
            SectionCodec{.type = scene_section::entities,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_entities,
                         .read = read_entities},
            SectionCodec{.type = scene_section::model_components,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_model_components,
                         .read = read_model_components},
            SectionCodec{.type = scene_section::material_overrides,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_material_overrides,
                         .read = read_material_overrides},
            SectionCodec{.type = scene_section::instanced_models,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_instanced_models,
                         .read = read_instanced_models},
            SectionCodec{.type = scene_section::point_lights,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_point_lights,
                         .read = read_point_lights},
            SectionCodec{.type = scene_section::spot_lights,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_spot_lights,
                         .read = read_spot_lights},
            SectionCodec{.type = scene_section::rigid_bodies,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_rigid_bodies,
                         .read = read_rigid_bodies},
            SectionCodec{.type = scene_section::scripts,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_scripts,
                         .read = read_scripts},
            SectionCodec{.type = scene_section::lifetimes,
                         .version = scene_section_version,
                         .oldest_readable = 1,
                         .write = write_lifetimes,
                         .read = read_lifetimes},
    };

    [[nodiscard]] auto find_codec(std::uint32_t type) noexcept -> SectionCodec const * {
        for (auto const &codec: section_codecs) {
            if (codec.type == type) {
                return &codec;
            }
        }

        return nullptr;
    }

    [[nodiscard]] auto entity_in_range(SceneDescription const &scene, std::uint32_t entity) noexcept -> bool {
        return entity < scene.entities.size();
    }

    [[nodiscard]] auto material_in_range(SceneDescription const &scene, std::uint32_t material) noexcept -> bool {
        return material == scene_no_index || material < scene.materials.size();
    }

} // namespace

auto encode_scene(SceneDescription const &scene) -> std::vector<std::byte> {
    ZoneScopedNC("encode_scene", tracy::Color::Goldenrod);

    ByteWriter writer;

    for (auto const &codec: section_codecs) {
        ByteWriter section;
        codec.write(section, scene);

        writer.write(codec.type);
        writer.write(codec.version);
        writer.write(std::uint16_t{0});
        writer.write(static_cast<std::uint64_t>(section.size()));
        writer.write_span(section.bytes());
    }

    return writer.take();
}

auto decode_scene(std::span<std::byte const> payload, std::uint16_t chunk_version, SceneDecodeReport *report)
        -> std::expected<SceneDescription, LbfError> {
    ZoneScopedNC("decode_scene", tracy::Color::Goldenrod);

    if (chunk_version == 0 || chunk_version > scene_chunk_version) {
        return std::unexpected(
                make_error(LbfErrorType::unsupported_version,
                           std::format("SCEN v{}, this build reads v1-v{}", chunk_version, scene_chunk_version)));
    }

    SceneDescription scene;
    ByteReader reader{payload};

    while (reader.remaining() > 0) {
        auto const type = reader.read<std::uint32_t>();
        auto const version = reader.read<std::uint16_t>();
        static_cast<void>(reader.read<std::uint16_t>());
        auto const size = reader.read<std::uint64_t>();
        auto const bytes = reader.read_bytes(static_cast<std::size_t>(size));

        if (reader.failed()) {
            return std::unexpected(make_error(LbfErrorType::malformed_payload, "truncated SCEN section header"));
        }

        auto const *codec = find_codec(type);

        if (codec == nullptr || version > codec->version) {
            // Written by a newer engine: unknown component, or a layout this build can't read. Keep the rest.
            if (report != nullptr) {
                ++report->skipped_sections;
            }
            continue;
        }

        if (version < codec->oldest_readable) {
            return std::unexpected(make_error(LbfErrorType::unsupported_version,
                                              std::format("SCEN section {} v{} is too old", type, version)));
        }

        ByteReader section{bytes};
        codec->read(section, version, scene);

        if (section.failed()) {
            return std::unexpected(
                    make_error(LbfErrorType::malformed_payload, std::format("SCEN section {} is malformed", type)));
        }
    }

    if (auto valid = validate_scene(scene); !valid) {
        return std::unexpected(valid.error());
    }

    return scene;
}

auto validate_scene(SceneDescription const &scene) -> std::expected<void, LbfError> {
    auto const fail = [](std::string_view what) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, what));
    };

    auto const entity_count = scene.entities.size();

    for (std::size_t index = 0; index < entity_count; ++index) {
        auto const parent = scene.entities[index].parent;

        if (parent != scene_no_index && parent >= entity_count) {
            return fail("entity parent out of range");
        }

        // Walking up more than entity_count links means a cycle.
        auto cursor = parent;

        for (std::size_t steps = 0; cursor != scene_no_index; ++steps) {
            if (steps > entity_count || cursor == index) {
                return fail("entity parent chain has a cycle");
            }

            cursor = scene.entities[cursor].parent;
        }
    }

    for (auto const &material: scene.materials) {
        for (auto const texture: material.textures) {
            if (texture != scene_no_index && texture >= scene.textures.size()) {
                return fail("material texture out of range");
            }
        }
    }

    for (auto const &component: scene.model_components) {
        if (!entity_in_range(scene, component.entity) || component.model >= scene.models.size()) {
            return fail("model component out of range");
        }
    }

    for (auto const &component: scene.material_overrides) {
        if (!entity_in_range(scene, component.entity) || !material_in_range(scene, component.material)) {
            return fail("material override out of range");
        }

        for (auto const &slot: component.slots) {
            if (slot.material == scene_no_index || !material_in_range(scene, slot.material)) {
                return fail("material slot override out of range");
            }
        }
    }

    for (auto const &component: scene.instanced_models) {
        if (!entity_in_range(scene, component.entity) || component.model >= scene.models.size() ||
            !material_in_range(scene, component.material)) {
            return fail("instanced model out of range");
        }
    }

    auto const entities_valid = [&](auto const &components) {
        return std::ranges::all_of(components,
                                   [&](auto const &component) { return entity_in_range(scene, component.entity); });
    };

    if (!entities_valid(scene.point_lights) || !entities_valid(scene.spot_lights) ||
        !entities_valid(scene.rigid_bodies) || !entities_valid(scene.scripts) || !entities_valid(scene.lifetimes)) {
        return fail("component entity out of range");
    }

    return {};
}
