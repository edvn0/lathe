#include "serialisation/cooked_model.hxx"

#include <algorithm>
#include <limits>
#include <format>
#include <source_location>
#include <utility>

#include <meshoptimizer.h>

#include "serialisation/byte_stream.hxx"

// Layouts baked into MODL v1. If one of these fires, bump cooked_model_version (see cooked_model.hxx).
static_assert(sizeof(CompressedModelVertex) == 20, "CompressedModelVertex changed: bump cooked_model_version");
static_assert(sizeof(GpuMeshlet) == 48, "GpuMeshlet changed: bump cooked_model_version");
static_assert(meshlet_max_vertices == 64 && meshlet_max_triangles == 124,
              "meshlet limits changed: bump cooked_model_version");
static_assert(lod_count <= 8, "the LOD mask is 8 bits");

namespace {

    constexpr auto no_index = std::numeric_limits<std::uint32_t>::max();

    enum class IndexEncoding : std::uint8_t {
        raw = 0,
        meshopt = 1,
    };

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

    auto write_image_index(ByteWriter &writer, std::optional<std::size_t> index) -> void {
        writer.write(index.has_value() ? static_cast<std::uint32_t>(*index) : no_index);
    }

    [[nodiscard]] auto read_image_index(ByteReader &reader, std::size_t image_count) -> std::optional<std::size_t> {
        auto const index = reader.read<std::uint32_t>();

        if (index == no_index) {
            return std::nullopt;
        }

        if (index >= image_count) {
            reader.fail();
            return std::nullopt;
        }

        return index;
    }

    [[nodiscard]]
    auto primitive_bounds(ModelCpuPrimitive const &primitive) -> std::pair<glm::vec3, glm::vec3> {
        if (primitive.bounds.has_value()) {
            return *primitive.bounds;
        }

        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const &vertex: primitive.vertices) {
            bounds_min = glm::min(bounds_min, vertex.position);
            bounds_max = glm::max(bounds_max, vertex.position);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    auto accumulate_bounds(ModelCpuData const &cpu_data, std::uint32_t node_index, glm::mat4 const &parent,
                           glm::vec3 &bounds_min, glm::vec3 &bounds_max, std::uint32_t depth) -> void {
        // A cyclic node graph would recurse forever; glTF forbids cycles but files aren't always valid.
        if (node_index >= cpu_data.nodes.size() || depth > 256) {
            return;
        }

        auto const &node = cpu_data.nodes[node_index];
        auto const local_to_model = parent * node.local_transform;

        if (node.mesh_index < cpu_data.meshes.size()) {
            for (auto const &primitive: cpu_data.meshes[node.mesh_index].primitives) {
                auto const accumulate = [&](glm::vec3 const &local) {
                    auto const position = glm::vec3{local_to_model * glm::vec4{local, 1.0F}};
                    bounds_min = glm::min(bounds_min, position);
                    bounds_max = glm::max(bounds_max, position);
                };

                if (!primitive.vertices.empty()) {
                    for (auto const &vertex: primitive.vertices) {
                        accumulate(vertex.position);
                    }
                    continue;
                }

                auto const [local_min, local_max] = primitive_bounds(primitive);

                for (std::uint32_t corner = 0; corner < 8; ++corner) {
                    accumulate(glm::vec3{(corner & 1U) != 0 ? local_max.x : local_min.x,
                                         (corner & 2U) != 0 ? local_max.y : local_min.y,
                                         (corner & 4U) != 0 ? local_max.z : local_min.z});
                }
            }
        }

        for (auto const child: node.children) {
            accumulate_bounds(cpu_data, child, local_to_model, bounds_min, bounds_max, depth + 1);
        }
    }

    [[nodiscard]] auto model_bounds(ModelCpuData const &cpu_data) -> std::pair<glm::vec3, glm::vec3> {
        if (cpu_data.bounds.has_value()) {
            return *cpu_data.bounds;
        }

        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const root: cpu_data.scene_roots) {
            accumulate_bounds(cpu_data, root, glm::mat4{1.0F}, bounds_min, bounds_max, 0);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    auto write_indices(ByteWriter &writer, std::vector<std::uint32_t> const &indices, std::size_t vertex_count)
            -> void {
        writer.write(static_cast<std::uint32_t>(indices.size()));

        if (!indices.empty() && indices.size() % 3 == 0) {
            std::vector<std::byte> encoded(meshopt_encodeIndexBufferBound(indices.size(), vertex_count));
            auto const size = meshopt_encodeIndexBuffer(reinterpret_cast<unsigned char *>(encoded.data()),
                                                        encoded.size(), indices.data(), indices.size());

            if (size != 0) {
                writer.write(std::to_underlying(IndexEncoding::meshopt));
                writer.write(static_cast<std::uint32_t>(size));
                writer.write_span(std::span<std::byte const>{encoded}.first(size));
                return;
            }
        }

        writer.write(std::to_underlying(IndexEncoding::raw));
        writer.write(static_cast<std::uint32_t>(indices.size() * sizeof(std::uint32_t)));
        writer.write_span(std::span<std::uint32_t const>{indices});
    }

    // Failure latches in `reader`.
    auto read_indices(ByteReader &reader, std::vector<std::uint32_t> &indices) -> bool {
        auto const count = reader.read<std::uint32_t>();
        auto const encoding = reader.read<std::uint8_t>();
        auto const size = reader.read<std::uint32_t>();
        auto const bytes = reader.read_bytes(size);

        if (reader.failed() || count > std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t)) {
            reader.fail();
            return false;
        }

        indices.resize(count);

        switch (static_cast<IndexEncoding>(encoding)) {
            case IndexEncoding::raw:
                if (size != count * sizeof(std::uint32_t)) {
                    reader.fail();
                    return false;
                }

                if (count != 0) {
                    std::memcpy(indices.data(), bytes.data(), size);
                }
                return true;

            case IndexEncoding::meshopt:
                if (count % 3 != 0 ||
                    meshopt_decodeIndexBuffer(indices.data(), count, sizeof(std::uint32_t),
                                              reinterpret_cast<unsigned char const *>(bytes.data()), bytes.size()) != 0) {
                    reader.fail();
                    return false;
                }
                return true;
        }

        reader.fail();
        return false;
    }

    auto write_meshlets(ByteWriter &writer, MeshletBuild const &build) -> void {
        writer.write_array(build.meshlets);
        writer.write_array(build.topology.data);
    }

    auto read_meshlets(ByteReader &reader, MeshletBuild &build) -> bool {
        reader.read_array(build.meshlets);
        reader.read_array(build.topology.data);

        build.topology.meshlets.clear();
        build.topology.meshlets.reserve(build.meshlets.size());

        auto const data_size = build.topology.data.size();

        for (auto const &meshlet: build.meshlets) {
            // Vertex indices then packed triangles, both inside `data`.
            if (static_cast<std::uint64_t>(meshlet.vertex_offset) + meshlet.vertex_count > data_size ||
                static_cast<std::uint64_t>(meshlet.triangle_offset) + meshlet.triangle_count > data_size) {
                reader.fail();
                return false;
            }

            build.topology.meshlets.push_back(MeshletTopology::Range{
                    .vertex_offset = meshlet.vertex_offset,
                    .triangle_offset = meshlet.triangle_offset,
                    .vertex_count = meshlet.vertex_count,
                    .triangle_count = meshlet.triangle_count,
            });
        }

        return reader.ok();
    }

} // namespace

auto encode_cooked_model(ModelCpuData const &cpu_data, std::span<CookedImageRef const> images,
                         std::span<DefaultSampler const> material_samplers)
        -> std::expected<std::vector<std::byte>, LbfError> {
    ZoneScopedNC("encode_cooked_model", tracy::Color::Goldenrod);

    if (images.size() != cpu_data.image_sources.size() || material_samplers.size() != cpu_data.materials.size()) {
        return std::unexpected(make_error(LbfErrorType::cook_failed, "image/sampler tables don't match the model"));
    }

    ByteWriter writer;

    auto const [bounds_min, bounds_max] = model_bounds(cpu_data);
    write_vec3(writer, bounds_min);
    write_vec3(writer, bounds_max);

    writer.write(static_cast<std::uint32_t>(images.size()));

    for (auto const &image: images) {
        writer.write(image.texture.value);
        writer.write(std::to_underlying(image.slot));
        writer.write_string(image.debug_name);
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.materials.size()));

    for (std::size_t index = 0; index < cpu_data.materials.size(); ++index) {
        auto const &material = cpu_data.materials[index];

        write_vec4(writer, material.base_colour_factor);
        write_vec3(writer, material.emissive_factor);
        writer.write(material.emissive_strength);
        writer.write(material.metallic_factor);
        writer.write(material.roughness_factor);
        writer.write(material.alpha_cutoff);
        writer.write(material.normal_scale);
        writer.write(material.occlusion_strength);
        writer.write(std::to_underlying(material.alpha_mode));
        writer.write(std::to_underlying(material_samplers[index]));

        write_image_index(writer, material.base_colour_image);
        write_image_index(writer, material.metallic_roughness_image);
        write_image_index(writer, material.normal_image);
        write_image_index(writer, material.occlusion_image);
        write_image_index(writer, material.emissive_image);
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.nodes.size()));

    for (auto const &node: cpu_data.nodes) {
        writer.write_span(std::span<float const>{&node.local_transform[0][0], 16});
        writer.write(node.mesh_index);
        writer.write_array(node.children);
    }

    writer.write_array(cpu_data.scene_roots);

    writer.write(static_cast<std::uint32_t>(cpu_data.lights.size()));

    for (auto const &light: cpu_data.lights) {
        writer.write(std::to_underlying(light.type));
        write_vec3(writer, light.position);
        write_vec3(writer, light.direction);
        write_vec3(writer, light.colour);
        writer.write(light.intensity);
        writer.write(light.range);
        writer.write(light.inner_cone_degrees);
        writer.write(light.outer_cone_degrees);
    }

    writer.write(static_cast<std::uint32_t>(cpu_data.meshes.size()));

    for (auto const &mesh: cpu_data.meshes) {
        writer.write(static_cast<std::uint32_t>(mesh.primitives.size()));

        for (auto const &primitive: mesh.primitives) {
            if (primitive.compressed_vertices.empty() || !primitive.meshlets[0].has_value()) {
                return std::unexpected(make_error(LbfErrorType::cook_failed, "primitive was not finalized"));
            }

            writer.write(primitive.material_index.value_or(no_index));

            auto const [primitive_min, primitive_max] = primitive_bounds(primitive);
            write_vec3(writer, primitive_min);
            write_vec3(writer, primitive_max);

            auto const &vertices = primitive.compressed_vertices;
            std::vector<std::byte> encoded(meshopt_encodeVertexBufferBound(vertices.size(), sizeof(CompressedModelVertex)));
            auto const encoded_size =
                    meshopt_encodeVertexBuffer(reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
                                               vertices.data(), vertices.size(), sizeof(CompressedModelVertex));

            if (encoded_size == 0) {
                return std::unexpected(make_error(LbfErrorType::cook_failed, "meshopt_encodeVertexBuffer failed"));
            }

            writer.write(static_cast<std::uint32_t>(vertices.size()));
            writer.write(static_cast<std::uint32_t>(encoded_size));
            writer.write_span(std::span<std::byte const>{encoded}.first(encoded_size));

            // Bit L: level L has its own index buffer and meshlets; otherwise it aliases level L-1.
            std::uint8_t level_mask = 1;

            for (std::uint32_t level = 1; level < lod_count; ++level) {
                if (primitive.reduced_indices[level - 1].has_value()) {
                    if (!primitive.meshlets[level].has_value()) {
                        return std::unexpected(make_error(LbfErrorType::cook_failed, "LOD without meshlets"));
                    }

                    level_mask = static_cast<std::uint8_t>(level_mask | (1U << level));
                }
            }

            writer.write(level_mask);

            for (std::uint32_t level = 0; level < lod_count; ++level) {
                if ((level_mask & (1U << level)) == 0) {
                    continue;
                }

                auto const &indices = level == 0 ? primitive.indices : *primitive.reduced_indices[level - 1];
                write_indices(writer, indices, vertices.size());
                write_meshlets(writer, *primitive.meshlets[level]);
            }
        }
    }

    return writer.take();
}

auto decode_cooked_model(std::span<std::byte const> payload, std::uint16_t version)
        -> std::expected<CookedModel, LbfError> {
    ZoneScopedNC("decode_cooked_model", tracy::Color::Goldenrod);

    if (version < cooked_model_oldest_readable_version || version > cooked_model_version) {
        return std::unexpected(make_error(LbfErrorType::unsupported_version,
                                          std::format("MODL v{}, this build reads v{}-v{}", version,
                                                      cooked_model_oldest_readable_version, cooked_model_version)));
    }

    ByteReader reader{payload};
    CookedModel cooked;
    auto &cpu_data = cooked.cpu_data;

    auto const bounds_min = read_vec3(reader);
    auto const bounds_max = read_vec3(reader);
    cpu_data.bounds = std::pair{bounds_min, bounds_max};

    // Every record below is at least a few bytes, so a count can't exceed the bytes left.
    auto const bounded_count = [&] {
        auto const count = reader.read<std::uint32_t>();

        if (count > reader.remaining()) {
            reader.fail();
            return std::uint32_t{0};
        }

        return count;
    };

    auto const image_count = bounded_count();
    cooked.images.resize(image_count);
    cpu_data.image_sources.resize(image_count);

    for (std::size_t index = 0; index < image_count; ++index) {
        auto &image = cooked.images[index];
        image.texture.value = reader.read<std::uint64_t>();

        auto const slot = reader.read<std::uint8_t>();

        if (slot > std::to_underlying(ModelTextureSlot::emissive)) {
            reader.fail();
        }

        image.slot = static_cast<ModelTextureSlot>(slot);
        reader.read_string(image.debug_name);

        cpu_data.image_sources[index].slot = image.slot;
        cpu_data.image_sources[index].debug_name = image.debug_name;
    }

    auto const material_count = bounded_count();
    cpu_data.materials.resize(material_count);
    cooked.material_samplers.resize(material_count);

    for (std::size_t index = 0; index < material_count; ++index) {
        auto &material = cpu_data.materials[index];

        material.base_colour_factor = read_vec4(reader);
        material.emissive_factor = read_vec3(reader);
        reader.read(material.emissive_strength);
        reader.read(material.metallic_factor);
        reader.read(material.roughness_factor);
        reader.read(material.alpha_cutoff);
        reader.read(material.normal_scale);
        reader.read(material.occlusion_strength);

        auto const alpha_mode = reader.read<std::uint32_t>();
        auto const sampler = reader.read<std::uint8_t>();

        if (alpha_mode > std::to_underlying(AlphaMode::blend) ||
            sampler > std::to_underlying(DefaultSampler::nearest_clamp)) {
            reader.fail();
        }

        material.alpha_mode = static_cast<AlphaMode>(alpha_mode);
        cooked.material_samplers[index] = static_cast<DefaultSampler>(sampler);

        material.base_colour_image = read_image_index(reader, image_count);
        material.metallic_roughness_image = read_image_index(reader, image_count);
        material.normal_image = read_image_index(reader, image_count);
        material.occlusion_image = read_image_index(reader, image_count);
        material.emissive_image = read_image_index(reader, image_count);
    }

    auto const node_count = bounded_count();
    cpu_data.nodes.resize(node_count);

    for (auto &node: cpu_data.nodes) {
        reader.read_span(std::span<float>{&node.local_transform[0][0], 16});
        reader.read(node.mesh_index);
        reader.read_array(node.children);
    }

    reader.read_array(cpu_data.scene_roots);

    auto const light_count = bounded_count();
    cpu_data.lights.resize(light_count);

    for (auto &light: cpu_data.lights) {
        auto const type = reader.read<std::uint8_t>();

        if (type > std::to_underlying(ModelLightType::spot)) {
            reader.fail();
        }

        light.type = static_cast<ModelLightType>(type);
        light.position = read_vec3(reader);
        light.direction = read_vec3(reader);
        light.colour = read_vec3(reader);
        reader.read(light.intensity);
        reader.read(light.range);
        reader.read(light.inner_cone_degrees);
        reader.read(light.outer_cone_degrees);
    }

    auto const mesh_count = bounded_count();
    cpu_data.meshes.resize(mesh_count);

    for (auto &mesh: cpu_data.meshes) {
        mesh.primitives.resize(bounded_count());

        for (auto &primitive: mesh.primitives) {
            if (reader.failed()) {
                break;
            }

            auto const material_index = reader.read<std::uint32_t>();

            if (material_index != no_index) {
                if (material_index >= material_count) {
                    reader.fail();
                }
                primitive.material_index = material_index;
            }

            auto const primitive_min = read_vec3(reader);
            auto const primitive_max = read_vec3(reader);
            primitive.bounds = std::pair{primitive_min, primitive_max};

            auto const vertex_count = reader.read<std::uint32_t>();
            auto const encoded_size = reader.read<std::uint32_t>();
            auto const encoded = reader.read_bytes(encoded_size);

            if (reader.failed() || vertex_count == 0) {
                reader.fail();
                break;
            }

            primitive.compressed_vertices.resize(vertex_count);

            if (meshopt_decodeVertexBuffer(primitive.compressed_vertices.data(), vertex_count,
                                           sizeof(CompressedModelVertex),
                                           reinterpret_cast<unsigned char const *>(encoded.data()), encoded.size()) != 0) {
                reader.fail();
                break;
            }

            auto const level_mask = reader.read<std::uint8_t>();

            if ((level_mask & 1U) == 0) {
                reader.fail();
                break;
            }

            for (std::uint32_t level = 0; level < lod_count; ++level) {
                if ((level_mask & (1U << level)) == 0) {
                    continue;
                }

                auto &indices = level == 0 ? primitive.indices : primitive.reduced_indices[level - 1].emplace();
                read_indices(reader, indices);

                for (auto const index: indices) {
                    if (index >= vertex_count) {
                        reader.fail();
                        break;
                    }
                }

                read_meshlets(reader, primitive.meshlets[level].emplace());
            }
        }
    }

    if (reader.failed()) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, "MODL chunk is truncated or inconsistent"));
    }

    auto const valid_node = [&](std::uint32_t index) { return index < node_count; };

    for (auto const &node: cpu_data.nodes) {
        if ((node.mesh_index != no_index && node.mesh_index >= mesh_count) ||
            !std::ranges::all_of(node.children, valid_node)) {
            return std::unexpected(make_error(LbfErrorType::malformed_payload, "node references a missing mesh or node"));
        }
    }

    if (!std::ranges::all_of(cpu_data.scene_roots, valid_node)) {
        return std::unexpected(make_error(LbfErrorType::malformed_payload, "scene root references a missing node"));
    }

    return cooked;
}
