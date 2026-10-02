#include "serialisation/scene_codec.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vector_relational.hpp>

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

        // Saved rotations are unit length; renormalise so a slightly-off or hand-edited one doesn't skew the matrix.
        // A zero or non-finite one is left alone for validate_scene() to reject.
        auto const length = glm::length(transform.rotation);

        if (std::isfinite(length) && length > 1e-6F) {
            transform.rotation = transform.rotation / length;
        }

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

    // ---- instance transforms -------------------------------------------------------------------------------------
    //
    // v2 writes each component's instances as float columns, one per field, every column byte-shuffled: all the
    // values' first bytes, then all their second bytes, and so on. Neighbouring instances share sign, exponent and
    // high mantissa bytes, so the shuffled planes are long runs zstd compresses well. When every instance decomposes
    // into translation, rotation and scale the columns are those 10 floats instead of the 16 matrix floats.
    //
    // On a 14,892-blade grass field this is 288 KiB compressed against 464 KiB for v1's interleaved matrices.

    enum class InstanceEncoding : std::uint8_t {
        matrices = 0,
        trs = 1,
    };

    inline constexpr std::size_t trs_column_count = 10; // translation xyz, rotation wxyz, scale xyz
    inline constexpr std::size_t matrix_column_count = 16;

    struct InstanceTrs {
        glm::vec3 translation{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        glm::vec3 scale{1.0F};
    };

    [[nodiscard]] auto compose(InstanceTrs const &trs) -> glm::mat4 {
        auto matrix = glm::mat4_cast(trs.rotation);
        matrix[0] *= trs.scale.x;
        matrix[1] *= trs.scale.y;
        matrix[2] *= trs.scale.z;
        matrix[3] = glm::vec4{trs.translation, 1.0F};
        return matrix;
    }

    // nullopt unless `matrix` is affine, without shear, and recomposes from its TRS to within float rounding.
    [[nodiscard]] auto decompose(glm::mat4 const &matrix) -> std::optional<InstanceTrs> {
        if (matrix[0][3] != 0.0F || matrix[1][3] != 0.0F || matrix[2][3] != 0.0F || matrix[3][3] != 1.0F) {
            return std::nullopt;
        }

        glm::vec3 scale{glm::length(glm::vec3{matrix[0]}), glm::length(glm::vec3{matrix[1]}),
                        glm::length(glm::vec3{matrix[2]})};
        if (scale.x == 0.0F || scale.y == 0.0F || scale.z == 0.0F) {
            return std::nullopt;
        }

        glm::mat3 basis{glm::vec3{matrix[0]} / scale.x, glm::vec3{matrix[1]} / scale.y,
                        glm::vec3{matrix[2]} / scale.z};

        // A mirroring matrix becomes a proper rotation and a negative X scale.
        if (glm::determinant(basis) < 0.0F) {
            scale.x = -scale.x;
            basis[0] = -basis[0];
        }

        InstanceTrs const trs{
                .translation = glm::vec3{matrix[3]},
                .rotation = glm::normalize(glm::quat_cast(basis)),
                .scale = scale,
        };

        // Shear survives the steps above as a wrong rotation, which the recomposition exposes.
        auto const recomposed = compose(trs);
        float largest = 1.0F;
        float worst_error = 0.0F;
        for (glm::length_t column = 0; column < 4; ++column) {
            for (glm::length_t row = 0; row < 4; ++row) {
                largest = std::max(largest, std::abs(matrix[column][row]));
                worst_error = std::max(worst_error, std::abs(recomposed[column][row] - matrix[column][row]));
            }
        }

        if (worst_error > 1e-5F * largest) {
            return std::nullopt;
        }

        return trs;
    }

    // `columns` holds column_count columns of `count` floats each, column after column.
    auto write_shuffled_columns(ByteWriter &writer, std::span<float const> columns, std::size_t count) -> void {
        std::vector<std::byte> shuffled(columns.size_bytes());
        auto const *source = reinterpret_cast<std::byte const *>(columns.data());
        auto const column_count = count == 0 ? 0 : columns.size() / count;

        for (std::size_t column = 0; column < column_count; ++column) {
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                auto *plane = shuffled.data() + (((column * sizeof(float)) + byte) * count);
                for (std::size_t value = 0; value < count; ++value) {
                    plane[value] = source[(((column * count) + value) * sizeof(float)) + byte];
                }
            }
        }

        writer.write_span(std::span<std::byte const>{shuffled});
    }

    [[nodiscard]] auto read_shuffled_columns(ByteReader &reader, std::size_t column_count, std::size_t count)
            -> std::vector<float> {
        std::vector<std::byte> shuffled(column_count * count * sizeof(float));
        std::vector<float> columns(column_count * count);

        if (!reader.read_span(std::span<std::byte>{shuffled})) {
            return columns;
        }

        auto *destination = reinterpret_cast<std::byte *>(columns.data());
        for (std::size_t column = 0; column < column_count; ++column) {
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                auto const *plane = shuffled.data() + (((column * sizeof(float)) + byte) * count);
                for (std::size_t value = 0; value < count; ++value) {
                    destination[(((column * count) + value) * sizeof(float)) + byte] = plane[value];
                }
            }
        }

        return columns;
    }

    auto write_instance_transforms(ByteWriter &writer, std::vector<glm::mat4> const &transforms) -> void {
        auto const count = transforms.size();

        std::vector<InstanceTrs> decomposed;
        decomposed.reserve(count);
        for (auto const &transform: transforms) {
            auto trs = decompose(transform);
            if (!trs) {
                break;
            }
            decomposed.push_back(*trs);
        }

        if (decomposed.size() == count) {
            writer.write(InstanceEncoding::trs);

            std::vector<float> columns(trs_column_count * count);

            for (std::size_t i = 0; i < count; ++i) {
                auto const &trs = decomposed[i];
                std::array const values{trs.translation.x, trs.translation.y, trs.translation.z,
                                        trs.rotation.w,    trs.rotation.x,    trs.rotation.y,
                                        trs.rotation.z,    trs.scale.x,       trs.scale.y,
                                        trs.scale.z};
                for (std::size_t field = 0; field < trs_column_count; ++field) {
                    columns[(field * count) + i] = values[field];
                }
            }

            write_shuffled_columns(writer, columns, count);
            return;
        }

        // Any instance that isn't a plain TRS (shear, projection) keeps the whole component as exact matrices.
        writer.write(InstanceEncoding::matrices);

        std::vector<float> columns(matrix_column_count * count);
        for (std::size_t i = 0; i < count; ++i) {
            auto const *values = &transforms[i][0][0];
            for (std::size_t field = 0; field < matrix_column_count; ++field) {
                columns[(field * count) + i] = values[field];
            }
        }

        write_shuffled_columns(writer, columns, count);
    }

    auto read_instance_transforms(ByteReader &reader, std::vector<glm::mat4> &transforms) -> void {
        auto const count = transforms.size();
        auto const encoding = reader.read<InstanceEncoding>();

        if (encoding == InstanceEncoding::trs) {
            auto const columns = read_shuffled_columns(reader, trs_column_count, count);
            auto const value = [&](std::size_t field, std::size_t i) { return columns[(field * count) + i]; };

            for (std::size_t i = 0; i < count; ++i) {
                transforms[i] = compose(InstanceTrs{
                        .translation = {value(0, i), value(1, i), value(2, i)},
                        .rotation = glm::quat{value(3, i), value(4, i), value(5, i), value(6, i)},
                        .scale = {value(7, i), value(8, i), value(9, i)},
                });
            }
        } else if (encoding == InstanceEncoding::matrices) {
            auto const columns = read_shuffled_columns(reader, matrix_column_count, count);

            for (std::size_t i = 0; i < count; ++i) {
                auto *values = &transforms[i][0][0];
                for (std::size_t field = 0; field < matrix_column_count; ++field) {
                    values[field] = columns[(field * count) + i];
                }
            }
        } else {
            reader.fail();
        }
    }

    auto write_instanced_models(ByteWriter &writer, SceneDescription const &scene) -> void {
        writer.write(static_cast<std::uint32_t>(scene.instanced_models.size()));

        for (auto const &component: scene.instanced_models) {
            writer.write(component.entity);
            writer.write(component.model);
            writer.write(component.material);
            writer.write(static_cast<std::uint32_t>(component.transforms.size()));
            write_instance_transforms(writer, component.transforms);
        }
    }

    auto read_instanced_models(ByteReader &reader, std::uint16_t version, SceneDescription &scene) -> void {
        scene.instanced_models.resize(read_count(reader, 16));

        for (auto &component: scene.instanced_models) {
            reader.read(component.entity);
            reader.read(component.model);
            reader.read(component.material);

            if (version == 1) {
                // v1: interleaved column-major matrices.
                component.transforms.resize(read_count(reader, sizeof(glm::mat4)));

                for (auto &transform: component.transforms) {
                    reader.read_span(std::span<float>{&transform[0][0], 16});
                }
            } else {
                component.transforms.resize(read_count(reader, trs_column_count * sizeof(float)));
                read_instance_transforms(reader, component.transforms);
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

    namespace environment_flag_bits {
        constexpr std::uint8_t draw_skybox = 1U << 0U;
        constexpr std::uint8_t fog_sky = 1U << 1U;
        constexpr std::uint8_t sun_drives_light = 1U << 2U;
        constexpr std::uint8_t derive_sun_colour = 1U << 3U;
        constexpr std::uint8_t multi_scatter = 1U << 4U;
        constexpr std::uint8_t fog_enabled = 1U << 5U;
        constexpr std::uint8_t fog_from_environment = 1U << 6U;
    } // namespace environment_flag_bits

    auto write_environment(ByteWriter &writer, SceneDescription const &scene) -> void {
        namespace bits = environment_flag_bits;

        auto const &environment = scene.environment;

        std::uint8_t flags = 0;
        flags |= environment.draw_skybox ? bits::draw_skybox : 0U;
        flags |= environment.fog_sky ? bits::fog_sky : 0U;
        flags |= environment.sun_drives_directional_light ? bits::sun_drives_light : 0U;
        flags |= environment.sun.derive_colour_from_sky ? bits::derive_sun_colour : 0U;
        flags |= environment.multi_scatter ? bits::multi_scatter : 0U;
        flags |= environment.fog.enabled ? bits::fog_enabled : 0U;
        flags |= environment.fog.from_environment ? bits::fog_from_environment : 0U;

        writer.write(std::to_underlying(environment.source));
        writer.write(flags);
        writer.write(std::uint16_t{0});

        writer.write(environment.ambient_intensity);
        writer.write(environment.rotation_degrees);
        writer.write(environment.exposure_ev);
        writer.write(environment.diffuse_intensity);
        writer.write(environment.specular_intensity);
        writer.write(environment.specular_occlusion);
        writer.write(environment.sky_intensity);
        writer.write(environment.hdr_cube_size);

        writer.write(scene.environment_id.value);
        writer.write_string(environment.hdr_source);

        auto const &sun = environment.sun;

        writer.write(sun.azimuth_degrees);
        writer.write(sun.elevation_degrees);
        writer.write(sun.turbidity);
        write_vec3(writer, sun.ground_albedo);
        writer.write(sun.angular_radius_degrees);
        write_vec3(writer, sun.colour);
        writer.write(sun.intensity);

        write_vec3(writer, environment.fog.colour);
        writer.write(environment.fog.extinction);
        writer.write(environment.fog.inscattering);
    }

    auto read_environment(ByteReader &reader, std::uint16_t, SceneDescription &scene) -> void {
        namespace bits = environment_flag_bits;

        auto &environment = scene.environment;

        auto const source = reader.read<std::uint8_t>();

        if (source > std::to_underlying(EnvironmentSource::hdr_image)) {
            reader.fail();
        }

        environment.source = static_cast<EnvironmentSource>(source);

        auto const flags = reader.read<std::uint8_t>();
        static_cast<void>(reader.read<std::uint16_t>());

        environment.draw_skybox = (flags & bits::draw_skybox) != 0;
        environment.fog_sky = (flags & bits::fog_sky) != 0;
        environment.sun_drives_directional_light = (flags & bits::sun_drives_light) != 0;
        environment.sun.derive_colour_from_sky = (flags & bits::derive_sun_colour) != 0;
        environment.multi_scatter = (flags & bits::multi_scatter) != 0;
        environment.fog.enabled = (flags & bits::fog_enabled) != 0;
        environment.fog.from_environment = (flags & bits::fog_from_environment) != 0;

        reader.read(environment.ambient_intensity);
        reader.read(environment.rotation_degrees);
        reader.read(environment.exposure_ev);
        reader.read(environment.diffuse_intensity);
        reader.read(environment.specular_intensity);
        reader.read(environment.specular_occlusion);
        reader.read(environment.sky_intensity);
        reader.read(environment.hdr_cube_size);

        scene.environment_id.value = reader.read<std::uint64_t>();
        reader.read_string(environment.hdr_source);

        auto &sun = environment.sun;

        reader.read(sun.azimuth_degrees);
        reader.read(sun.elevation_degrees);
        reader.read(sun.turbidity);
        sun.ground_albedo = read_vec3(reader);
        reader.read(sun.angular_radius_degrees);
        sun.colour = read_vec3(reader);
        reader.read(sun.intensity);

        environment.fog.colour = read_vec3(reader);
        reader.read(environment.fog.extinction);
        reader.read(environment.fog.inscattering);
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
                         .version = instanced_models_section_version,
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
            SectionCodec{.type = scene_section::environment,
                         .version = environment_section_version,
                         .oldest_readable = 1,
                         .write = write_environment,
                         .read = read_environment},
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

    // ---- value checks --------------------------------------------------------------------------------------------
    //
    // Index checks keep instantiate from reading out of bounds; these keep NaN, infinities and negative sizes out of
    // transforms, physics and GPU buffers, where they poison culling, shadow and physics state for the whole scene.

    [[nodiscard]] auto finite(float value) noexcept -> bool { return std::isfinite(value); }

    template<glm::length_t N>
    [[nodiscard]] auto finite(glm::vec<N, float> const &value) noexcept -> bool {
        for (glm::length_t i = 0; i < N; ++i) {
            if (!std::isfinite(value[i])) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] auto finite(glm::mat4 const &value) noexcept -> bool {
        return finite(value[0]) && finite(value[1]) && finite(value[2]) && finite(value[3]);
    }

    // finite and >= 0
    [[nodiscard]] auto non_negative(float value) noexcept -> bool { return std::isfinite(value) && value >= 0.0F; }

    // Why `environment` can't be rendered or saved, if anything.
    [[nodiscard]] auto environment_problem(SceneEnvironment const &environment) -> std::optional<std::string_view> {
        auto const &sun = environment.sun;
        auto const &fog = environment.fog;

        if (!finite(environment.ambient_intensity) || !finite(environment.rotation_degrees) ||
            !finite(environment.exposure_ev) || !finite(environment.diffuse_intensity) ||
            !finite(environment.specular_intensity) || !finite(environment.specular_occlusion) ||
            !finite(environment.sky_intensity) || !finite(sun.azimuth_degrees) || !finite(sun.elevation_degrees) ||
            !finite(sun.turbidity) || !finite(sun.ground_albedo) || !finite(sun.angular_radius_degrees) ||
            !finite(sun.colour) || !finite(sun.intensity) || !finite(fog.colour) || !finite(fog.extinction) ||
            !finite(fog.inscattering)) {
            return "non-finite environment value";
        }

        if (sun.turbidity < 2.0F || sun.turbidity > 10.0F) {
            return "environment turbidity out of range";
        }

        if (glm::any(glm::lessThan(sun.ground_albedo, glm::vec3{0.0F})) ||
            glm::any(glm::greaterThan(sun.ground_albedo, glm::vec3{1.0F}))) {
            return "environment ground albedo out of range";
        }

        if (sun.elevation_degrees < -90.0F || sun.elevation_degrees > 90.0F) {
            return "environment sun elevation out of range";
        }

        if (sun.angular_radius_degrees <= 0.0F || sun.angular_radius_degrees > 10.0F) {
            return "environment sun radius out of range";
        }

        if (environment.exposure_ev < -20.0F || environment.exposure_ev > 20.0F) {
            return "environment exposure out of range";
        }

        if (environment.ambient_intensity < 0.0F || environment.diffuse_intensity < 0.0F ||
            environment.specular_intensity < 0.0F || environment.specular_occlusion < 0.0F ||
            environment.sky_intensity < 0.0F || sun.intensity < 0.0F || sun.colour.x < 0.0F || sun.colour.y < 0.0F ||
            sun.colour.z < 0.0F || fog.extinction < 0.0F || fog.inscattering < 0.0F || fog.colour.x < 0.0F ||
            fog.colour.y < 0.0F || fog.colour.z < 0.0F) {
            return "negative environment intensity";
        }

        if (environment.hdr_cube_size < 256 || environment.hdr_cube_size > 1024 ||
            !std::has_single_bit(environment.hdr_cube_size)) {
            return "environment cube size must be 256, 512 or 1024";
        }

        if (environment.source == EnvironmentSource::hdr_image && environment.hdr_source.empty()) {
            return "environment image source has no path";
        }

        return std::nullopt;
    }


    [[nodiscard]] auto valid_transform(Components::Transform const &transform) noexcept -> bool {
        auto const &q = transform.rotation;
        auto const length_squared = (q.w * q.w) + (q.x * q.x) + (q.y * q.y) + (q.z * q.z);

        // A zero quaternion has no direction to normalise to; it turns every child matrix into NaN.
        return finite(transform.position) && finite(transform.scale) && std::isfinite(length_squared) &&
               length_squared > 1e-12F;
    }

    // Each entity may hold at most one of any component; instantiate emplaces them, and EnTT treats a second
    // emplace as a bug (assert in debug, undefined behaviour in release).
    template<typename Components>
    [[nodiscard]] auto entities_unique(SceneDescription const &scene, Components const &components) -> bool {
        std::vector<bool> seen(scene.entities.size(), false);

        for (auto const &component: components) {
            if (seen[component.entity]) {
                return false;
            }

            seen[component.entity] = true;
        }

        return true;
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

    // Everything from here on indexes by entity, which is now known to be in range.
    if (!entities_unique(scene, scene.model_components) || !entities_unique(scene, scene.material_overrides) ||
        !entities_unique(scene, scene.instanced_models) || !entities_unique(scene, scene.point_lights) ||
        !entities_unique(scene, scene.spot_lights) || !entities_unique(scene, scene.rigid_bodies) ||
        !entities_unique(scene, scene.scripts) || !entities_unique(scene, scene.lifetimes)) {
        return fail("entity has the same component more than once");
    }

    if (!finite(scene.physics_settings.gravity) || !finite(scene.physics_settings.ground_y)) {
        return fail("non-finite physics settings");
    }

    if (auto problem = environment_problem(scene.environment); problem.has_value()) {
        return fail(*problem);
    }

    for (auto const &entity: scene.entities) {
        if (entity.transform.has_value() && !valid_transform(*entity.transform)) {
            return fail("entity transform is not finite or has a zero rotation");
        }
    }

    for (auto const &material: scene.materials) {
        if (!finite(material.base_colour_factor) || !finite(material.emissive_factor) ||
            !finite(material.emissive_strength) || !finite(material.metallic_factor) ||
            !finite(material.roughness_factor) || !finite(material.normal_scale) ||
            !finite(material.occlusion_strength) || !finite(material.alpha_cutoff) || !finite(material.wind_strength)) {
            return fail("material has a non-finite factor");
        }

        // The renderer buckets shadow batches by cascade 0..shadow_cascade_count-1; anything else matches no bucket.
        if (material.max_shadow_cascade >= shadow_cascade_count &&
            material.max_shadow_cascade != GpuMaterial::no_shadow_cascade) {
            return fail("material shadow cascade out of range");
        }
    }

    for (auto const &component: scene.instanced_models) {
        if (!std::ranges::all_of(component.transforms, [](glm::mat4 const &matrix) { return finite(matrix); })) {
            return fail("instanced model has a non-finite transform");
        }
    }

    for (auto const &component: scene.point_lights) {
        auto const &light = component.light;

        if (!finite(light.colour) || !non_negative(light.intensity) || !non_negative(light.range)) {
            return fail("point light has a non-finite or negative value");
        }
    }

    for (auto const &component: scene.spot_lights) {
        auto const &light = component.light;

        if (!finite(light.colour) || !non_negative(light.intensity) || !non_negative(light.range) ||
            !non_negative(light.inner_cone_degrees) || !non_negative(light.outer_cone_degrees)) {
            return fail("spot light has a non-finite or negative value");
        }
    }

    for (auto const &component: scene.rigid_bodies) {
        auto const &body = component.body;

        auto const boxes_valid = body.compound_boxes == nullptr ||
                                 std::ranges::all_of(*body.compound_boxes, [](Components::CompoundBoxChild const &box) {
                                     return finite(box.local_centre) && finite(box.half_extents) &&
                                            glm::all(glm::greaterThanEqual(box.half_extents, glm::vec3{0.0F}));
                                 });

        if (!finite(body.velocity) || !finite(body.half_extents) ||
            !glm::all(glm::greaterThanEqual(body.half_extents, glm::vec3{0.0F})) ||
            !non_negative(body.capsule_radius) || !non_negative(body.capsule_height) || !finite(body.restitution) ||
            !non_negative(body.mass) || !boxes_valid) {
            return fail("rigid body has a non-finite or negative value");
        }
    }

    for (auto const &component: scene.lifetimes) {
        if (!finite(component.remaining_seconds)) {
            return fail("lifetime is not finite");
        }
    }

    return {};
}
