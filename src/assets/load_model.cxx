#include "assets/load_model.hxx"

#include <bit>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <future>
#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <vector>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <limits>
#include <optional>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assets/material_storage.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "gpu/image.hxx"
#include "gpu/image_storage.hxx"
#include "gpu/sampler_storage.hxx"

namespace {

    auto accumulate_node_lights(fastgltf::Asset const &asset, ModelCpuData const &cpu_data, std::uint32_t node_index,
                                glm::mat4 const &parent_transform, std::vector<ModelCpuLight> &out_lights) -> void {
        auto const &gltf_node = asset.nodes[node_index];
        auto const local_to_model = parent_transform * cpu_data.nodes[node_index].local_transform;

        debug("accumulate_node_lights: visiting node {} ('{}'), lightIndex={}", node_index,
              gltf_node.name.empty() ? "<unnamed>" : std::string{gltf_node.name},
              gltf_node.lightIndex.has_value() ? static_cast<int>(*gltf_node.lightIndex) : -1);


        if (gltf_node.lightIndex.has_value()) {
            auto const &gltf_light = asset.lights[*gltf_node.lightIndex];

            if (gltf_light.type != fastgltf::LightType::Directional) {
                ModelCpuLight light{};

                light.type =
                        gltf_light.type == fastgltf::LightType::Spot ? ModelLightType::spot : ModelLightType::point;

                light.position = glm::vec3{local_to_model[3]};
                light.direction = glm::normalize(glm::mat3{local_to_model} * glm::vec3{0.0F, 0.0F, -1.0F});

                light.colour = glm::vec3{gltf_light.color[0], gltf_light.color[1], gltf_light.color[2]};
                light.intensity = gltf_light.intensity;
                light.range = gltf_light.range.value_or(light.range);

                if (gltf_light.type == fastgltf::LightType::Spot) {
                    light.inner_cone_degrees = glm::degrees(gltf_light.innerConeAngle.value_or(0.0F));
                    light.outer_cone_degrees =
                            glm::degrees(gltf_light.outerConeAngle.value_or(glm::quarter_pi<float>()));
                }

                out_lights.push_back(light);
            }
        }

        for (auto const child: cpu_data.nodes[node_index].children) {
            accumulate_node_lights(asset, cpu_data, child, local_to_model, out_lights);
        }
    }

    auto accumulate_node_bounds(ModelCpuData const &cpu_data, std::uint32_t node_index,
                                glm::mat4 const &parent_transform, glm::vec3 &bounds_min, glm::vec3 &bounds_max)
            -> void {
        if (node_index >= cpu_data.nodes.size()) {
            return;
        }

        auto const &node = cpu_data.nodes[node_index];
        auto const local_to_model = parent_transform * node.local_transform;

        constexpr auto invalid_mesh = std::numeric_limits<std::uint32_t>::max();

        if (node.mesh_index != invalid_mesh && node.mesh_index < cpu_data.meshes.size()) {
            for (auto const &primitive: cpu_data.meshes[node.mesh_index].primitives) {
                for (auto const &vertex: primitive.vertices) {
                    auto const model_space_position = glm::vec3{local_to_model * glm::vec4{vertex.position, 1.0F}};

                    bounds_min = glm::min(bounds_min, model_space_position);
                    bounds_max = glm::max(bounds_max, model_space_position);
                }
            }
        }

        for (auto const child: node.children) {
            accumulate_node_bounds(cpu_data, child, local_to_model, bounds_min, bounds_max);
        }
    }

    auto compute_primitive_bounds(std::span<ModelVertex const> vertices) -> std::pair<glm::vec3, glm::vec3> {
        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const &vertex: vertices) {
            bounds_min = glm::min(bounds_min, vertex.position);
            bounds_max = glm::max(bounds_max, vertex.position);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    auto compute_model_bounds(ModelCpuData const &cpu_data) -> std::pair<glm::vec3, glm::vec3> {
        auto bounds_min = glm::vec3{std::numeric_limits<float>::max()};
        auto bounds_max = glm::vec3{std::numeric_limits<float>::lowest()};

        for (auto const root: cpu_data.scene_roots) {
            accumulate_node_bounds(cpu_data, root, glm::mat4{1.0F}, bounds_min, bounds_max);
        }

        if (bounds_min.x > bounds_max.x) {
            return {glm::vec3{-0.5F}, glm::vec3{0.5F}};
        }

        return {bounds_min, bounds_max};
    }

    auto to_glm(fastgltf::math::fmat4x4 const &matrix) noexcept -> glm::mat4 { return glm::make_mat4(matrix.data()); }

    auto find_attribute(fastgltf::Primitive const &primitive, std::string_view name) -> std::optional<std::size_t> {
        auto const iterator = primitive.findAttribute(name);

        if (iterator == primitive.attributes.end()) {
            return std::nullopt;
        }

        return iterator->accessorIndex;
    }

    template<typename T>
    auto read_accessor(fastgltf::Asset const &asset, std::size_t accessor_index) -> std::vector<T> {
        auto const &accessor = asset.accessors[accessor_index];
        std::vector<T> values(accessor.count);
        fastgltf::copyFromAccessor<T>(asset, accessor, values.data());
        return values;
    }

    auto load_indices(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive, std::size_t vertex_count)
            -> std::expected<std::vector<std::uint32_t>, ModelLoadError> {
        if (!primitive.indicesAccessor.has_value()) {
            std::vector<std::uint32_t> indices(vertex_count);

            for (std::uint32_t index = 0; index < static_cast<std::uint32_t>(vertex_count); ++index) {
                indices[index] = index;
            }

            return indices;
        }

        auto const accessor_index = *primitive.indicesAccessor;

        if (accessor_index >= asset.accessors.size()) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::invalid_accessor,
            });
        }

        auto const &accessor = asset.accessors[accessor_index];

        std::vector<std::uint32_t> indices(accessor.count);
        fastgltf::copyFromAccessor<std::uint32_t>(asset, accessor, indices.data());

        return indices;
    }

    struct MikktspaceUserData {
        std::vector<ModelVertex> *vertices = nullptr;
    };

    auto mikktspace_vertex_index(int face, int vert) noexcept -> std::size_t {
        return static_cast<std::size_t>(face) * 3 + static_cast<std::size_t>(vert);
    }

    auto mikktspace_get_num_faces(SMikkTSpaceContext const *context) -> int {
        auto const *user_data = static_cast<MikktspaceUserData const *>(context->m_pUserData);

        return static_cast<int>(user_data->vertices->size() / 3);
    }

    auto mikktspace_get_num_vertices_of_face(SMikkTSpaceContext const *, int) -> int { return 3; }

    auto mikktspace_get_position(SMikkTSpaceContext const *context, float out[3], int face, int vert) -> void {
        auto const *user_data = static_cast<MikktspaceUserData const *>(context->m_pUserData);
        auto const &position = (*user_data->vertices)[mikktspace_vertex_index(face, vert)].position;

        out[0] = position.x;
        out[1] = position.y;
        out[2] = position.z;
    }

    auto mikktspace_get_normal(SMikkTSpaceContext const *context, float out[3], int face, int vert) -> void {
        auto const *user_data = static_cast<MikktspaceUserData const *>(context->m_pUserData);
        auto const &normal = (*user_data->vertices)[mikktspace_vertex_index(face, vert)].normal;

        out[0] = normal.x;
        out[1] = normal.y;
        out[2] = normal.z;
    }

    auto mikktspace_get_tex_coord(SMikkTSpaceContext const *context, float out[2], int face, int vert) -> void {
        auto const *user_data = static_cast<MikktspaceUserData const *>(context->m_pUserData);
        auto const &texcoord = (*user_data->vertices)[mikktspace_vertex_index(face, vert)].texcoord;

        out[0] = texcoord.x;
        out[1] = texcoord.y;
    }

    auto mikktspace_set_tspace_basic(SMikkTSpaceContext const *context, float const tangent[3], float sign, int face,
                                     int vert) -> void {
        auto *user_data = static_cast<MikktspaceUserData *>(context->m_pUserData);

        (*user_data->vertices)[mikktspace_vertex_index(face, vert)].tangent =
                glm::vec4{tangent[0], tangent[1], tangent[2], sign};
    }

} // namespace

auto generate_tangents(std::vector<ModelVertex> &vertices, std::vector<std::uint32_t> &indices)
        -> std::expected<void, ModelLoadError> {
    std::vector<ModelVertex> expanded(indices.size());

    for (std::size_t index = 0; index < indices.size(); ++index) {
        expanded[index] = vertices[indices[index]];
    }

    MikktspaceUserData user_data{.vertices = &expanded};

    SMikkTSpaceInterface interface{};
    interface.m_getNumFaces = mikktspace_get_num_faces;
    interface.m_getNumVerticesOfFace = mikktspace_get_num_vertices_of_face;
    interface.m_getPosition = mikktspace_get_position;
    interface.m_getNormal = mikktspace_get_normal;
    interface.m_getTexCoord = mikktspace_get_tex_coord;
    interface.m_setTSpaceBasic = mikktspace_set_tspace_basic;
    interface.m_setTSpace = nullptr;

    SMikkTSpaceContext context{};
    context.m_pInterface = &interface;
    context.m_pUserData = &user_data;

    if (genTangSpaceDefault(&context) == 0) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::tangent_generation_failed,
        });
    }

    std::vector<unsigned int> remap(expanded.size());

    auto const unique_vertex_count = meshopt_generateVertexRemap(remap.data(), nullptr, expanded.size(),
                                                                 expanded.data(), expanded.size(), sizeof(ModelVertex));

    std::vector<ModelVertex> welded_vertices(unique_vertex_count);
    meshopt_remapVertexBuffer(welded_vertices.data(), expanded.data(), expanded.size(), sizeof(ModelVertex),
                              remap.data());

    std::vector<std::uint32_t> welded_indices(expanded.size());
    meshopt_remapIndexBuffer(welded_indices.data(), nullptr, expanded.size(), remap.data());

    meshopt_optimizeVertexCache(welded_indices.data(), welded_indices.data(), welded_indices.size(),
                                unique_vertex_count);

    meshopt_optimizeOverdraw(welded_indices.data(), welded_indices.data(), welded_indices.size(),
                             &welded_vertices[0].position.x, unique_vertex_count, sizeof(ModelVertex), 1.05f);

    std::vector<ModelVertex> fetch_optimized_vertices(unique_vertex_count);
    auto const fetch_remap_count =
            meshopt_optimizeVertexFetch(fetch_optimized_vertices.data(), welded_indices.data(), welded_indices.size(),
                                        welded_vertices.data(), unique_vertex_count, sizeof(ModelVertex));
    fetch_optimized_vertices.resize(fetch_remap_count);

    vertices = std::move(fetch_optimized_vertices);
    indices = std::move(welded_indices);

    return {};
}

auto generate_mesh_lods(std::vector<ModelVertex> const &vertices, std::vector<std::uint32_t> const &indices)
        -> std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> {
    std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> reduced{};

    for (std::size_t level = 0; level < lod_simplification_ratios.size(); ++level) {
        auto target_index_count =
                static_cast<std::size_t>(static_cast<float>(indices.size()) * lod_simplification_ratios[level]);
        target_index_count -= target_index_count % 3;
        if (target_index_count == 0 || target_index_count >= indices.size()) {
            continue;
        }

        std::vector<std::uint32_t> simplified(indices.size());
        auto const result_count = meshopt_simplify(simplified.data(), indices.data(), indices.size(),
                                                   &vertices[0].position.x, vertices.size(), sizeof(ModelVertex),
                                                   target_index_count, 1e-2F, meshopt_SimplifyLockBorder, nullptr);
        if (result_count == 0 || result_count >= indices.size()) {
            continue;
        }

        simplified.resize(result_count);
        meshopt_optimizeVertexCache(simplified.data(), simplified.data(), simplified.size(), vertices.size());
        reduced[level] = std::move(simplified);
    }

    return reduced;
}

namespace {

    auto extract_primitive_cpu(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive,
                               ModelLoadProfile *profile) -> std::expected<ModelCpuPrimitive, ModelLoadError> {
        ScopedProfileSample extract_sample{profile != nullptr ? &profile->primitive_extract_ns : nullptr};

        if (primitive.type != fastgltf::PrimitiveType::Triangles) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::unsupported_primitive,
            });
        }

        auto const position_accessor = find_attribute(primitive, "POSITION");

        if (!position_accessor.has_value()) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::missing_position,
            });
        }

        auto const positions = read_accessor<glm::vec3>(asset, *position_accessor);

        std::vector<ModelVertex> vertices(positions.size());

        for (std::size_t index = 0; index < positions.size(); ++index) {
            vertices[index].position = positions[index];

            vertices[index].normal = glm::vec3{0.0F, 1.0F, 0.0F};
            vertices[index].tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F};
            vertices[index].texcoord = glm::vec2{0.0F};
        }

        if (auto const normal_accessor = find_attribute(primitive, "NORMAL")) {
            auto const normals = read_accessor<glm::vec3>(asset, *normal_accessor);

            if (normals.size() != vertices.size()) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::invalid_accessor,
                });
            }

            for (std::size_t index = 0; index < vertices.size(); ++index) {
                vertices[index].normal = normals[index];
            }
        }

        bool has_tangents = false;

        if (auto const tangent_accessor = find_attribute(primitive, "TANGENT")) {
            auto const tangents = read_accessor<glm::vec4>(asset, *tangent_accessor);

            if (tangents.size() != vertices.size()) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::invalid_accessor,
                });
            }

            for (std::size_t index = 0; index < vertices.size(); ++index) {
                vertices[index].tangent = tangents[index];
            }

            has_tangents = true;
        }

        if (auto const texcoord_accessor = find_attribute(primitive, "TEXCOORD_0")) {
            auto const texcoords = read_accessor<glm::vec2>(asset, *texcoord_accessor);

            if (texcoords.size() != vertices.size()) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::invalid_accessor,
                });
            }

            for (std::size_t index = 0; index < vertices.size(); ++index) {
                vertices[index].texcoord = texcoords[index];
            }
        }

        auto indices_result = load_indices(asset, primitive, vertices.size());

        if (!indices_result) {
            return std::unexpected(indices_result.error());
        }

        auto indices = std::move(*indices_result);

        return ModelCpuPrimitive{
                .vertices = std::move(vertices),
                .indices = std::move(indices),
                .material_index = primitive.materialIndex.has_value()
                                          ? std::optional(static_cast<std::uint32_t>(*primitive.materialIndex))
                                          : std::nullopt,
                .has_tangents = has_tangents,
        };
    }

    auto finalize_primitive_cpu(ModelCpuPrimitive raw, ModelLoadProfile *profile)
            -> std::expected<ModelCpuPrimitive, ModelLoadError> {
        auto vertices = std::move(raw.vertices);
        auto indices = std::move(raw.indices);

        if (!raw.has_tangents) {
            ScopedProfileSample const tangent_sample{profile != nullptr ? &profile->tangent_generation_ns : nullptr};

            auto tangent_result = generate_tangents(vertices, indices);

            if (!tangent_result) {
                return std::unexpected(tangent_result.error());
            }
        }

        std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> reduced_indices{};

        {
            ScopedProfileSample const lod_sample{profile != nullptr ? &profile->lod_generation_ns : nullptr};

            reduced_indices = generate_mesh_lods(vertices, indices);
        }

        return ModelCpuPrimitive{
                .vertices = std::move(vertices),
                .indices = std::move(indices),
                .reduced_indices = std::move(reduced_indices),
                .material_index = raw.material_index,
        };
    }

    struct ImageSource {
        std::filesystem::path path; // external file; empty if embedded
        std::vector<std::byte> encoded; // embedded bytes; empty if `path` is set
    };

    auto resolve_image_source(fastgltf::Asset const &asset, fastgltf::Image const &image,
                              std::filesystem::path const &base_directory)
            -> std::expected<ImageSource, ModelLoadError> {
        if (auto const *uri_source = std::get_if<fastgltf::sources::URI>(&image.data)) {
            if (uri_source->uri.isLocalPath() && uri_source->fileByteOffset == 0) {
                return ImageSource{.path = base_directory / uri_source->uri.fspath()};
            }

            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::unsupported_image_source,
            });
        }

        if (auto const *buffer_view_source = std::get_if<fastgltf::sources::BufferView>(&image.data)) {
            auto const &buffer_view = asset.bufferViews[buffer_view_source->bufferViewIndex];
            auto const &buffer = asset.buffers[buffer_view.bufferIndex];

            auto const *array_source = std::get_if<fastgltf::sources::Array>(&buffer.data);

            if (array_source == nullptr) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::unsupported_image_source,
                });
            }

            auto const *bytes = reinterpret_cast<std::byte const *>(array_source->bytes.data());

            return ImageSource{
                    .encoded = std::vector<std::byte>{bytes + buffer_view.byteOffset,
                                                      bytes + buffer_view.byteOffset + buffer_view.byteLength},
            };
        }

        if (auto const *array_source = std::get_if<fastgltf::sources::Array>(&image.data)) {
            auto const *bytes = reinterpret_cast<std::byte const *>(array_source->bytes.data());

            return ImageSource{.encoded = std::vector<std::byte>{bytes, bytes + array_source->bytes.size()}};
        }

        if (auto const *vector_source = std::get_if<fastgltf::sources::Vector>(&image.data)) {
            auto const *bytes = reinterpret_cast<std::byte const *>(vector_source->bytes.data());

            return ImageSource{.encoded = std::vector<std::byte>{bytes, bytes + vector_source->bytes.size()}};
        }

        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::unsupported_image_source,
        });
    }

    auto is_nearest_filter(fastgltf::Filter filter) -> bool {
        return filter == fastgltf::Filter::Nearest || filter == fastgltf::Filter::NearestMipMapNearest ||
               filter == fastgltf::Filter::NearestMipMapLinear;
    }

    auto select_sampler(const SamplerStorage &sampler_storage, fastgltf::Sampler const *gltf_sampler) -> SamplerHandle {
        bool const nearest =
                gltf_sampler != nullptr &&
                (gltf_sampler->minFilter.has_value() ? is_nearest_filter(*gltf_sampler->minFilter)
                                                     : (gltf_sampler->magFilter.has_value() &&
                                                        *gltf_sampler->magFilter == fastgltf::Filter::Nearest));

        bool const clamp = gltf_sampler != nullptr && (gltf_sampler->wrapS == fastgltf::Wrap::ClampToEdge ||
                                                       gltf_sampler->wrapT == fastgltf::Wrap::ClampToEdge);

        if (nearest) {
            return clamp ? sampler_storage.nearest_clamp() : sampler_storage.nearest_repeat();
        }

        return clamp ? sampler_storage.linear_clamp() : sampler_storage.linear_repeat();
    }

    auto select_material_sampler(fastgltf::Asset const &asset, fastgltf::Material const &gltf_material,
                                 SamplerStorage &sampler_storage) -> SamplerHandle {
        std::optional<std::size_t> texture_index;

        if (gltf_material.pbrData.baseColorTexture.has_value()) {
            texture_index = gltf_material.pbrData.baseColorTexture->textureIndex;
        } else if (gltf_material.pbrData.metallicRoughnessTexture.has_value()) {
            texture_index = gltf_material.pbrData.metallicRoughnessTexture->textureIndex;
        } else if (gltf_material.normalTexture.has_value()) {
            texture_index = gltf_material.normalTexture->textureIndex;
        }

        if (!texture_index.has_value()) {
            return select_sampler(sampler_storage, nullptr);
        }

        auto const &gltf_texture = asset.textures[*texture_index];

        if (!gltf_texture.samplerIndex.has_value()) {
            return select_sampler(sampler_storage, nullptr);
        }

        return select_sampler(sampler_storage, &asset.samplers[*gltf_texture.samplerIndex]);
    }

    using ImageCache = std::unordered_map<std::size_t, std::size_t>;
    using ImageSources = std::vector<ModelCpuImageSource>;

    template<typename T>
    concept IndexableTexture = requires(T &t) {
        { t.textureIndex } -> std::convertible_to<std::size_t>;
    };

    template<IndexableTexture TextureInfoT>
    auto resolve_texture_cpu(fastgltf::Asset const &asset, std::optional<TextureInfoT> const &info,
                             ModelTextureSlot slot, std::string_view slot_name, std::string_view material_name,
                             std::filesystem::path const &gltf_path, std::filesystem::path const &base_directory,
                             ImageCache &image_cache, ImageSources &image_sources)
            -> std::expected<std::optional<std::size_t>, ModelLoadError> {
        if (!info.has_value()) {
            return std::nullopt;
        }

        auto const &gltf_texture = asset.textures[info->textureIndex];

        if (!gltf_texture.imageIndex.has_value()) {
            return std::nullopt;
        }

        auto const image_index = *gltf_texture.imageIndex;

        if (auto const cached = image_cache.find(image_index); cached != image_cache.end()) {
            return cached->second;
        }

        auto source = resolve_image_source(asset, asset.images[image_index], base_directory);

        if (!source) {
            return std::unexpected(source.error());
        }

        auto const &gltf_image = asset.images[image_index];

        auto image_name = !gltf_image.name.empty()
                                  ? std::string{gltf_image.name}
                                  : (material_name.empty() ? std::format("image_{}", image_index)
                                                           : std::format("{}_{}", material_name, slot_name));

        auto const cpu_index = image_sources.size();

        auto const embedded_size = source->encoded.size();

        image_sources.push_back(ModelCpuImageSource{
                .path = std::move(source->path),
                .encoded = std::move(source->encoded),
                .cache_key = source->path.empty()
                                     ? std::format("{}#{}#{}", gltf_path.string(), image_index, embedded_size)
                                     : std::string{},
                .slot = slot,
                .debug_name = std::move(image_name),
        });

        image_cache.emplace(image_index, cpu_index);

        return cpu_index;
    }

    auto load_material_cpu(fastgltf::Asset const &asset, fastgltf::Material const &gltf_material,
                           SamplerStorage &sampler_storage, std::filesystem::path const &gltf_path,
                           std::filesystem::path const &base_directory,
                           std::unordered_map<std::size_t, std::size_t> &image_cache,
                           std::vector<ModelCpuImageSource> &image_sources)
            -> std::expected<ModelCpuMaterial, ModelLoadError> {
        ModelCpuMaterial material{};

        auto const &base_colour = gltf_material.pbrData.baseColorFactor;
        material.base_colour_factor = glm::vec4{base_colour[0], base_colour[1], base_colour[2], base_colour[3]};

        auto const &emissive = gltf_material.emissiveFactor;
        material.emissive_factor = glm::vec3{emissive[0], emissive[1], emissive[2]};

        material.emissive_strength = gltf_material.emissiveStrength;

        material.metallic_factor = gltf_material.pbrData.metallicFactor;
        material.roughness_factor = gltf_material.pbrData.roughnessFactor;
        material.alpha_cutoff = gltf_material.alphaCutoff;

        switch (gltf_material.alphaMode) {
            case fastgltf::AlphaMode::Opaque:
                material.alpha_mode = AlphaMode::opaque;
                break;
            case fastgltf::AlphaMode::Mask:
                material.alpha_mode = AlphaMode::mask;
                break;
            case fastgltf::AlphaMode::Blend:
                material.alpha_mode = AlphaMode::blend;
                break;
        }

        material.sampler = select_material_sampler(asset, gltf_material, sampler_storage);

        auto const material_name = std::string{gltf_material.name};

        auto base_colour_image =
                resolve_texture_cpu(asset, gltf_material.pbrData.baseColorTexture, ModelTextureSlot::base_colour,
                                    "basecolor", material_name, gltf_path, base_directory, image_cache, image_sources);

        if (!base_colour_image) {
            return std::unexpected(base_colour_image.error());
        }

        material.base_colour_image = *base_colour_image;

        auto metallic_roughness_image = resolve_texture_cpu(
                asset, gltf_material.pbrData.metallicRoughnessTexture, ModelTextureSlot::metallic_roughness,
                "metallic_roughness", material_name, gltf_path, base_directory, image_cache, image_sources);

        if (!metallic_roughness_image) {
            return std::unexpected(metallic_roughness_image.error());
        }

        material.metallic_roughness_image = *metallic_roughness_image;

        if (gltf_material.normalTexture.has_value()) {
            material.normal_scale = gltf_material.normalTexture->scale;
        }

        auto normal_image = resolve_texture_cpu(asset, gltf_material.normalTexture, ModelTextureSlot::normal, "normal",
                                                material_name, gltf_path, base_directory, image_cache, image_sources);

        if (!normal_image) {
            return std::unexpected(normal_image.error());
        }

        material.normal_image = *normal_image;

        if (gltf_material.occlusionTexture.has_value()) {
            material.occlusion_strength = gltf_material.occlusionTexture->strength;
        }

        auto occlusion_image =
                resolve_texture_cpu(asset, gltf_material.occlusionTexture, ModelTextureSlot::occlusion, "occlusion",
                                    material_name, gltf_path, base_directory, image_cache, image_sources);

        if (!occlusion_image) {
            return std::unexpected(occlusion_image.error());
        }

        material.occlusion_image = *occlusion_image;

        auto emissive_image =
                resolve_texture_cpu(asset, gltf_material.emissiveTexture, ModelTextureSlot::emissive, "emissive",
                                    material_name, gltf_path, base_directory, image_cache, image_sources);

        if (!emissive_image) {
            return std::unexpected(emissive_image.error());
        }

        material.emissive_image = *emissive_image;

        return material;
    }

} // namespace

auto load_model_cpu_unfinalized(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                                std::shared_ptr<ModelLoadProfile> profile)
        -> std::expected<ModelCpuData, ModelLoadError> {
    ZoneScopedNC("LoadModelCpuUnfinalized", tracy::Color::Goldenrod);

    auto *const profile_ptr = profile.get();

    ScopedProfileSample gltf_sample{profile_ptr != nullptr ? &profile_ptr->gltf_parse_ns : nullptr};

    auto file_data = fastgltf::GltfDataBuffer::FromPath(path);
    if (!file_data) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::file_not_found,
        });
    }

    static thread_local fastgltf::Parser parser{
            fastgltf::Extensions::KHR_materials_emissive_strength | fastgltf::Extensions::KHR_lights_punctual,
    };
    constexpr auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices |
                             fastgltf::Options::LoadExternalImages;
    auto const base_directory = path.parent_path();
    auto asset_result = parser.loadGltf(file_data.get(), base_directory, options);
    gltf_sample.stop();

    if (asset_result.error() != fastgltf::Error::None) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::parse_error,
        });
    }

    auto asset = std::move(asset_result.get());

    debug("[load_model_cpu]: parsed {} nodes, {} lights, {} scenes", asset.nodes.size(), asset.lights.size(),
          asset.scenes.size());

    ModelCpuData cpu_data;
    cpu_data.meshes.reserve(asset.meshes.size());
    cpu_data.nodes.reserve(asset.nodes.size());
    cpu_data.materials.reserve(asset.materials.size());
    cpu_data.profile = std::move(profile);

    std::unordered_map<std::size_t, std::size_t> image_cache;

    for (auto const &gltf_material: asset.materials) {
        ScopedProfileSample const material_sample{profile_ptr != nullptr ? &profile_ptr->material_resolve_ns : nullptr};

        auto material = load_material_cpu(asset, gltf_material, sampler_storage, path, base_directory, image_cache,
                                          cpu_data.image_sources);

        if (!material) {
            return std::unexpected(material.error());
        }

        cpu_data.materials.push_back(std::move(*material));
    }

    for (auto const &gltf_mesh: asset.meshes) {
        ModelCpuMesh mesh;
        mesh.primitives.reserve(gltf_mesh.primitives.size());

        for (auto const &gltf_primitive: gltf_mesh.primitives) {
            auto primitive = extract_primitive_cpu(asset, gltf_primitive, profile_ptr);

            if (!primitive) {
                return std::unexpected(primitive.error());
            }

            if (primitive->material_index.has_value() && *primitive->material_index >= cpu_data.materials.size()) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::invalid_material_index,
                });
            }

            mesh.primitives.push_back(std::move(*primitive));
        }

        cpu_data.meshes.push_back(std::move(mesh));
    }

    for (auto const &gltf_node: asset.nodes) {
        ModelNode node{
                .local_transform = to_glm(fastgltf::getTransformMatrix(gltf_node)),
        };

        if (gltf_node.meshIndex.has_value()) {
            node.mesh_index = static_cast<std::uint32_t>(*gltf_node.meshIndex);
        }

        node.children.reserve(gltf_node.children.size());

        for (auto const child: gltf_node.children) {
            node.children.push_back(static_cast<std::uint32_t>(child));
        }

        cpu_data.nodes.push_back(std::move(node));
    }


    if (auto const scene_index = asset.defaultScene.value_or(0); scene_index < asset.scenes.size()) {
        auto const &scene = asset.scenes[scene_index];

        cpu_data.scene_roots.reserve(scene.nodeIndices.size());

        for (auto const node_index: scene.nodeIndices) {
            cpu_data.scene_roots.push_back(static_cast<std::uint32_t>(node_index));
        }
    }

    cpu_data.lights.reserve(asset.lights.size());

    for (auto const root: cpu_data.scene_roots) {
        accumulate_node_lights(asset, cpu_data, root, glm::mat4{1.0F}, cpu_data.lights);
    }
    debug("[load_model_cpu]: extracted {} lights from {} scene roots", cpu_data.lights.size(),
          cpu_data.scene_roots.size());

    return cpu_data;
}

auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                    std::shared_ptr<ModelLoadProfile> profile) -> std::expected<ModelCpuData, ModelLoadError> {
    ZoneScopedNC("LoadModelCpu", tracy::Color::Goldenrod);

    auto cpu_data = load_model_cpu_unfinalized(path, sampler_storage, std::move(profile));

    if (!cpu_data) {
        return cpu_data;
    }

    auto *const profile_ptr = cpu_data->profile.get();

    for (auto &mesh: cpu_data->meshes) {
        for (auto &primitive: mesh.primitives) {
            auto finalized = finalize_primitive_cpu(std::move(primitive), profile_ptr);

            if (!finalized) {
                return std::unexpected(finalized.error());
            }

            primitive = std::move(*finalized);
        }
    }

    return cpu_data;
}

auto load_model_cpu_async(std::filesystem::path path, SamplerStorage &sampler_storage,
                          std::shared_ptr<ModelLoadProfile> profile)
        -> std::future<std::expected<ModelCpuData, ModelLoadError>> {
    return thread_pool().submit_task([path = std::move(path), &sampler_storage, profile = std::move(profile)] {
        return load_model_cpu_unfinalized(path, sampler_storage, profile);
    });
}

auto start_primitive_finalization(ModelCpuData cpu_data) -> ModelPrimitiveFinalization {
    ZoneScopedNC("StartPrimitiveFinalization", tracy::Color::Goldenrod);

    ModelPrimitiveFinalization finalization{.cpu_data = std::move(cpu_data)};

    auto *const profile = finalization.cpu_data.profile.get();
    auto &pool = thread_pool();

    for (std::size_t mesh_index = 0; mesh_index < finalization.cpu_data.meshes.size(); ++mesh_index) {
        auto &mesh = finalization.cpu_data.meshes[mesh_index];

        for (std::size_t primitive_index = 0; primitive_index < mesh.primitives.size(); ++primitive_index) {
            finalization.targets.emplace_back(mesh_index, primitive_index);

            finalization.tasks.push_back(
                    pool.submit_task([raw = std::move(mesh.primitives[primitive_index]), profile]() mutable {
                        return finalize_primitive_cpu(std::move(raw), profile);
                    }));
        }
    }

    return finalization;
}

auto step_primitive_finalization(ModelPrimitiveFinalization &finalization)
        -> std::expected<std::optional<ModelCpuData>, ModelLoadError> {
    ZoneScopedNC("StepPrimitiveFinalization", tracy::Color::Goldenrod);

    using namespace std::chrono_literals;

    std::size_t write = 0;

    for (std::size_t read = 0; read < finalization.tasks.size(); ++read) {
        if (finalization.tasks[read].wait_for(0s) != std::future_status::ready) {
            if (write != read) {
                finalization.tasks[write] = std::move(finalization.tasks[read]);
                finalization.targets[write] = finalization.targets[read];
            }

            ++write;
            continue;
        }

        auto result = finalization.tasks[read].get();

        if (!result) {
            return std::unexpected(result.error());
        }

        auto const [mesh_index, primitive_index] = finalization.targets[read];
        finalization.cpu_data.meshes[mesh_index].primitives[primitive_index] = std::move(*result);
    }

    finalization.tasks.resize(write);
    finalization.targets.resize(write);

    if (!finalization.tasks.empty()) {
        return std::optional<ModelCpuData>{std::nullopt};
    }

    return std::optional<ModelCpuData>{std::move(finalization.cpu_data)};
}


namespace {

    [[nodiscard]]
    auto texture_role_for_slot(ModelTextureSlot slot) noexcept -> TextureRole {
        switch (slot) {
            using enum ModelTextureSlot;
            case base_colour:
            case emissive:
                return TextureRole::colour;

            case normal:
                return TextureRole::normal_map;

            case metallic_roughness:
            case occlusion:
                return TextureRole::generic;
        }

        return TextureRole::generic;
    }

    [[nodiscard]]
    auto default_fallback_for_slot(ImageStorage const &image_storage, ModelTextureSlot slot) noexcept -> ImageHandle {
        switch (slot) {
            case ModelTextureSlot::base_colour:
                return image_storage.white();
            case ModelTextureSlot::normal:
                return image_storage.flat_normal();
            case ModelTextureSlot::metallic_roughness:
                return image_storage.metallic_roughness();
            case ModelTextureSlot::occlusion:
                return image_storage.occlusion();
            case ModelTextureSlot::emissive:
                return image_storage.emissive();
        }

        return image_storage.white();
    }

} // namespace

auto start_model_gpu_upload(ModelCpuData cpu_data, ImageStorage &image_storage, TextureStreamer &texture_streamer)
        -> ModelGpuUpload {
    ZoneScopedNC("StartModelGpuUpload", tracy::Color::Goldenrod);

    ModelGpuUpload upload{.cpu_data = std::move(cpu_data)};

    upload.image_handles.reserve(upload.cpu_data.image_sources.size());

    for (auto const &source: upload.cpu_data.image_sources) {
        auto const role = texture_role_for_slot(source.slot);
        auto const fallback = default_fallback_for_slot(image_storage, source.slot);

        auto const handle =
                source.path.empty()
                        ? texture_streamer.request_from_memory(image_storage, source.encoded, role, source.cache_key,
                                                               fallback, source.debug_name, upload.cpu_data.profile)
                        : texture_streamer.request(image_storage, source.path, role, fallback, source.debug_name,
                                                   upload.cpu_data.profile);

        upload.image_handles.push_back(handle);
    }

    upload.materials.reserve(upload.cpu_data.materials.size());
    upload.meshes.reserve(upload.cpu_data.meshes.size());

    return upload;
}

auto step_model_gpu_upload(ModelGpuUpload &upload, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                           ImageStorage &image_storage, MaterialStorage &material_storage, std::uint32_t item_budget)
        -> std::expected<std::optional<Model>, ModelLoadError> {
    ZoneScopedNC("StepModelGpuUpload", tracy::Color::Goldenrod);

    if (command_buffer == VK_NULL_HANDLE) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::invalid_argument,
        });
    }

    auto const &cpu_data = upload.cpu_data;
    auto *const profile = cpu_data.profile.get();

    if (profile != nullptr) {
        profile->gpu_upload_frames.fetch_add(1, std::memory_order_seq_cst);
    }

    auto const resolve_image = [&](std::optional<std::size_t> cpu_index, ImageHandle fallback) {
        return cpu_index.has_value() ? upload.image_handles[*cpu_index] : fallback;
    };

    for (std::uint32_t processed = 0; processed < item_budget;) {
        if (upload.material_cursor < cpu_data.materials.size()) {
            auto const &cpu_material = cpu_data.materials[upload.material_cursor];

            MaterialCreateInfo const info{
                    .base_colour_factor = cpu_material.base_colour_factor,
                    .emissive_factor = cpu_material.emissive_factor,
                    .emissive_strength = cpu_material.emissive_strength,
                    .metallic_factor = cpu_material.metallic_factor,
                    .roughness_factor = cpu_material.roughness_factor,
                    .normal_scale = cpu_material.normal_scale,
                    .occlusion_strength = cpu_material.occlusion_strength,
                    .alpha_cutoff = cpu_material.alpha_cutoff,
                    .base_colour_texture = resolve_image(cpu_material.base_colour_image, image_storage.white()),
                    .normal_texture = resolve_image(cpu_material.normal_image, image_storage.flat_normal()),
                    .metallic_roughness_texture =
                            resolve_image(cpu_material.metallic_roughness_image, image_storage.metallic_roughness()),
                    .occlusion_texture = resolve_image(cpu_material.occlusion_image, image_storage.occlusion()),
                    .emissive_texture = resolve_image(cpu_material.emissive_image, image_storage.emissive()),
                    .sampler = cpu_material.sampler,
                    .alpha_mode = cpu_material.alpha_mode,
            };

            ScopedProfileSample material_sample{profile != nullptr ? &profile->material_creation_ns : nullptr};

            auto gpu_material = to_gpu_material(info);
            auto material_handle = material_storage.create_material(gpu_material);

            material_sample.stop();

            if (!material_handle) {
                return std::unexpected(ModelLoadError{
                        .type = ModelLoadErrorType::material_creation_failed,
                        .cause = ErrorCause{Boxed<MaterialStorageError>{material_handle.error()}},
                });
            }

            upload.materials.push_back(*material_handle);
            ++upload.material_cursor;
            ++processed;
            continue;
        }

        if (upload.mesh_cursor < cpu_data.meshes.size()) {
            auto const &cpu_mesh = cpu_data.meshes[upload.mesh_cursor];

            if (upload.meshes.size() == upload.mesh_cursor) {
                auto &mesh = upload.meshes.emplace_back();
                mesh.primitives.reserve(cpu_mesh.primitives.size());
            }

            if (upload.primitive_cursor < cpu_mesh.primitives.size()) {
                auto const &cpu_primitive = cpu_mesh.primitives[upload.primitive_cursor];

                ScopedProfileSample vertex_compress_sample{profile != nullptr ? &profile->vertex_compression_ns
                                                                              : nullptr};

                auto const compressed_vertices = compress_vertices(cpu_primitive.vertices);

                vertex_compress_sample.stop();

                ScopedProfileSample geometry_upload_sample{profile != nullptr ? &profile->geometry_upload_ns : nullptr};

                auto vertex_slice = geometry_arena.allocate_vertices(
                        command_buffer, std::span<CompressedModelVertex const>{compressed_vertices});

                if (!vertex_slice) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::geometry_upload_failed,
                            .cause = ErrorCause{Boxed<GeometryArenaError>{vertex_slice.error()}},
                    });
                }

                std::array<MeshGeometry, lod_count> lods{};

                for (std::uint32_t level = 0; level < lod_count; ++level) {
                    lods[level].vertices = *vertex_slice;

                    auto const *source_indices = [&]() {
                        if (level == 0) {
                            return &cpu_primitive.indices;
                        }

                        auto const &reduced = cpu_primitive.reduced_indices[level - 1];
                        return reduced.has_value() ? &*reduced : nullptr;
                    }();

                    if (source_indices == nullptr) {
                        lods[level].indices = lods[level - 1].indices;
                        continue;
                    }

                    auto index_slice = geometry_arena.allocate_indices(command_buffer,
                                                                       std::span<std::uint32_t const>{*source_indices});

                    if (!index_slice) {
                        return std::unexpected(ModelLoadError{
                                .type = ModelLoadErrorType::geometry_upload_failed,
                                .cause = ErrorCause{Boxed<GeometryArenaError>{index_slice.error()}},
                        });
                    }

                    lods[level].indices = *index_slice;
                }

                geometry_upload_sample.stop();

                if (cpu_primitive.material_index.has_value() &&
                    *cpu_primitive.material_index >= upload.materials.size()) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::invalid_material_index,
                    });
                }

                auto const [primitive_bounds_min, primitive_bounds_max] =
                        compute_primitive_bounds(std::span<ModelVertex const>{cpu_primitive.vertices});

                upload.meshes[upload.mesh_cursor].primitives.push_back(ModelPrimitive{
                        .lods = lods,
                        .material_index = cpu_primitive.material_index,
                        .bounds_min = primitive_bounds_min,
                        .bounds_max = primitive_bounds_max,
                });

                ++upload.primitive_cursor;
                ++processed;

                if (upload.primitive_cursor >= cpu_mesh.primitives.size()) {
                    ++upload.mesh_cursor;
                    upload.primitive_cursor = 0;
                }

                continue;
            }

            // Empty mesh -- nothing to process, advance without spending budget.
            ++upload.mesh_cursor;
            upload.primitive_cursor = 0;
            continue;
        }

        Model model;
        model.meshes = std::move(upload.meshes);
        model.materials = std::move(upload.materials);
        model.nodes = cpu_data.nodes;
        model.scene_roots = cpu_data.scene_roots;
        std::tie(model.bounds_min, model.bounds_max) = compute_model_bounds(cpu_data);
        model.lights = cpu_data.lights;

        return std::optional<Model>{std::move(model)};
    }

    return std::optional<Model>{std::nullopt};
}

auto record_model_gpu_upload(ModelCpuData const &cpu_data, VkCommandBuffer command_buffer,
                             GeometryArena &geometry_arena, ImageStorage &image_storage,
                             TextureStreamer &texture_streamer, MaterialStorage &material_storage)
        -> std::expected<Model, ModelLoadError> {
    ZoneScopedNC("RecordModelGpuUpload", tracy::Color::Goldenrod);

    auto upload = start_model_gpu_upload(cpu_data, image_storage, texture_streamer);

    while (true) {
        auto stepped = step_model_gpu_upload(upload, command_buffer, geometry_arena, image_storage, material_storage,
                                             std::numeric_limits<std::uint32_t>::max());

        if (!stepped) {
            return std::unexpected(stepped.error());
        }

        if (stepped->has_value()) {
            return std::move(**stepped);
        }
    }
}

auto load_model(std::filesystem::path const &path, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                ImageStorage &image_storage, TextureStreamer &texture_streamer, SamplerStorage &sampler_storage,
                MaterialStorage &material_storage) -> std::expected<Model, ModelLoadError> {
    ZoneScopedNC("LoadModel", tracy::Color::Goldenrod);

    auto cpu_data = load_model_cpu(path, sampler_storage);

    if (!cpu_data) {
        return std::unexpected(cpu_data.error());
    }

    return record_model_gpu_upload(*cpu_data, command_buffer, geometry_arena, image_storage, texture_streamer,
                                   material_storage);
}
