#include "load_model.hxx"

#include <bit>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <future>
#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <meshoptimizer.h>
#include <mikktspace.h>
#include <mutex>
#include <vector>

#include <stb_image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "image.hxx"
#include "image_storage.hxx"
#include "logger.hxx"
#include "renderer.hxx"
#include "sampler_storage.hxx"

namespace {

    // Walks the node hierarchy from scene_roots, accumulating each node's
    // local_transform, and folds every mesh's vertex positions (transformed
    // into model space) into a running min/max.
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

    // ------------------------------------------------------------------------
    // Shared helpers — pure CPU, unchanged from before.
    // ------------------------------------------------------------------------

    auto to_glm(fastgltf::math::fmat4x4 const &matrix) noexcept -> glm::mat4 {
        static_assert(matrix.size() == 16, "fastgltf changed its mat4 impl...?");
        return glm::make_mat4(matrix.data());
    }

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

    // ------------------------------------------------------------------------
    // MikkTSpace tangent generation (unchanged — pure CPU already).
    // ------------------------------------------------------------------------

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

    auto const unique_vertex_count = meshopt_generateVertexRemap(
            remap.data(), nullptr, expanded.size(), expanded.data(), expanded.size(), sizeof(ModelVertex));

    std::vector<ModelVertex> welded_vertices(unique_vertex_count);
    meshopt_remapVertexBuffer(welded_vertices.data(), expanded.data(), expanded.size(), sizeof(ModelVertex),
                              remap.data());

    std::vector<std::uint32_t> welded_indices(expanded.size());
    meshopt_remapIndexBuffer(welded_indices.data(), nullptr, expanded.size(), remap.data());

    vertices = std::move(welded_vertices);
    indices = std::move(welded_indices);

    return {};
}

namespace {

    // ------------------------------------------------------------------------
    // CPU pass — geometry. No GeometryArena involved anymore: this only
    // builds the vertex/index arrays and hands them back as plain data.
    // ------------------------------------------------------------------------

    auto load_primitive_cpu(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive)
            -> std::expected<ModelCpuPrimitive, ModelLoadError> {
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

        if (!has_tangents) {
            auto tangent_result = generate_tangents(vertices, indices);

            if (!tangent_result) {
                return std::unexpected(tangent_result.error());
            }
        }

        return ModelCpuPrimitive{
                .vertices = std::move(vertices),
                .indices = std::move(indices),
                .material_index = primitive.materialIndex.has_value()
                                          ? std::optional(static_cast<std::uint32_t>(*primitive.materialIndex))
                                          : std::nullopt,
        };
    }

    // ------------------------------------------------------------------------
    // CPU pass — material / texture import. Everything here decodes bytes
    // into memory; nothing touches ImageStorage or MaterialStorage.
    // ------------------------------------------------------------------------

    // Returns owned bytes rather than a span: the URI branch below has to
    // read a file off disk, so every branch needs to hand back memory it
    // actually owns rather than a view into someone else's buffer.
    auto image_bytes(fastgltf::Asset const &asset, fastgltf::Image const &image,
                     std::filesystem::path const &base_directory)
            -> std::expected<std::vector<std::byte>, ModelLoadError> {
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

            return std::vector<std::byte>{bytes + buffer_view.byteOffset,
                                          bytes + buffer_view.byteOffset + buffer_view.byteLength};
        }

        if (auto const *array_source = std::get_if<fastgltf::sources::Array>(&image.data)) {
            auto const *bytes = reinterpret_cast<std::byte const *>(array_source->bytes.data());

            return std::vector<std::byte>{bytes, bytes + array_source->bytes.size()};
        }

        if (auto const *vector_source = std::get_if<fastgltf::sources::Vector>(&image.data)) {
            auto const *bytes = reinterpret_cast<std::byte const *>(vector_source->bytes.data());

            return std::vector<std::byte>{bytes, bytes + vector_source->bytes.size()};
        }

        // External image referenced by a relative/absolute file path in a
        // .gltf (JSON) document — e.g. "textures/albedo.png". Data URIs are
        // already decoded into sources::Array/Vector by fastgltf itself, so
        // by the time we get here a URI source is always a real file on
        // disk that we resolve relative to the glTF's own directory.
        if (auto const *uri_source = std::get_if<fastgltf::sources::URI>(&image.data)) {
            if (uri_source->uri.isLocalPath() && uri_source->fileByteOffset == 0) {
                auto const image_path = base_directory / uri_source->uri.fspath();

                std::ifstream file_stream{image_path, std::ios::binary | std::ios::ate};

                if (!file_stream.is_open()) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::unsupported_image_source,
                    });
                }

                auto const file_size = static_cast<std::size_t>(file_stream.tellg());
                file_stream.seekg(0);

                std::vector<std::byte> bytes(file_size);
                file_stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(file_size));

                if (!file_stream) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::unsupported_image_source,
                    });
                }

                return bytes;
            }
        }

        // A non-local URI (e.g. http/https), a byte-offset URI, or an
        // unsupported source (basisu/dds/webp extension images) lands here.
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::unsupported_image_source,
        });
    }

    struct DecodedImage {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::byte> pixels; // RGBA8
    };

    auto decode_image(std::span<std::byte const> encoded) -> std::expected<DecodedImage, ModelLoadError> {
        int width = 0;
        int height = 0;
        int source_channels = 0;

        auto *pixels = stbi_load_from_memory(reinterpret_cast<stbi_uc const *>(encoded.data()),
                                             static_cast<int>(encoded.size()), &width, &height, &source_channels, 4);

        if (pixels == nullptr) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::image_decode_failed,
            });
        }

        DecodedImage result{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
                .pixels = {},
        };

        auto const byte_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;

        result.pixels.assign(reinterpret_cast<std::byte const *>(pixels),
                             reinterpret_cast<std::byte const *>(pixels) + byte_count);

        stbi_image_free(pixels);

        return result;
    }

    // Picks one of the renderer's small fixed set of samplers to best match
    // a glTF sampler's filter/wrap settings. Just reads pre-created handles
    // out of SamplerStorage, so this is safe to call during the CPU pass.
    auto select_sampler(SamplerStorage &sampler_storage, fastgltf::Sampler const *gltf_sampler) -> SamplerHandle {
        bool const nearest = gltf_sampler != nullptr && gltf_sampler->magFilter.has_value() &&
                             *gltf_sampler->magFilter == fastgltf::Filter::Nearest;

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

    // Decodes the image behind a glTF texture reference into `cpu_images`
    // and returns its index there, or nullopt if `info` is empty (caller
    // substitutes the appropriate ImageStorage default at record time).
    // Cache is keyed by glTF image index only, so — same as before the
    // split — if one raw image is reused across an srgb and a non-srgb
    // slot, whichever texture type hits it first wins the format; that
    // quirk is preserved rather than fixed here.
    template<typename TextureInfoT>
    auto resolve_texture_cpu(fastgltf::Asset const &asset, std::optional<TextureInfoT> const &info, bool is_srgb,
                             std::filesystem::path const &base_directory,
                             std::unordered_map<std::size_t, std::size_t> &image_cache,
                             std::vector<ModelCpuImage> &cpu_images)
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

        auto encoded = image_bytes(asset, asset.images[image_index], base_directory);

        if (!encoded) {
            return std::unexpected(encoded.error());
        }

        auto decoded = decode_image(std::span<std::byte const>{*encoded});

        if (!decoded) {
            return std::unexpected(decoded.error());
        }

        auto const cpu_index = cpu_images.size();

        cpu_images.push_back(ModelCpuImage{
                .width = decoded->width,
                .height = decoded->height,
                .is_srgb = is_srgb,
                .pixels = std::move(decoded->pixels),
        });

        image_cache.emplace(image_index, cpu_index);

        return cpu_index;
    }

    auto load_material_cpu(fastgltf::Asset const &asset, fastgltf::Material const &gltf_material,
                           SamplerStorage &sampler_storage, std::filesystem::path const &base_directory,
                           std::unordered_map<std::size_t, std::size_t> &image_cache,
                           std::vector<ModelCpuImage> &cpu_images) -> std::expected<ModelCpuMaterial, ModelLoadError> {
        ModelCpuMaterial material{};

        auto const &base_colour = gltf_material.pbrData.baseColorFactor;
        material.base_colour_factor = glm::vec4{base_colour[0], base_colour[1], base_colour[2], base_colour[3]};

        auto const &emissive = gltf_material.emissiveFactor;
        material.emissive_factor = glm::vec3{emissive[0], emissive[1], emissive[2]};

        // ASSUMPTION: emissiveStrength requires KHR_materials_emissive_strength
        // to have been parsed by fastgltf; drop this line if it fails to compile
        // against your fastgltf version.
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

        auto base_colour_image = resolve_texture_cpu(asset, gltf_material.pbrData.baseColorTexture, true,
                                                     base_directory, image_cache, cpu_images);

        if (!base_colour_image) {
            return std::unexpected(base_colour_image.error());
        }

        material.base_colour_image = *base_colour_image;

        auto metallic_roughness_image = resolve_texture_cpu(asset, gltf_material.pbrData.metallicRoughnessTexture,
                                                            false, base_directory, image_cache, cpu_images);

        if (!metallic_roughness_image) {
            return std::unexpected(metallic_roughness_image.error());
        }

        material.metallic_roughness_image = *metallic_roughness_image;

        if (gltf_material.normalTexture.has_value()) {
            material.normal_scale = gltf_material.normalTexture->scale;
        }

        auto normal_image =
                resolve_texture_cpu(asset, gltf_material.normalTexture, false, base_directory, image_cache, cpu_images);

        if (!normal_image) {
            return std::unexpected(normal_image.error());
        }

        material.normal_image = *normal_image;

        if (gltf_material.occlusionTexture.has_value()) {
            material.occlusion_strength = gltf_material.occlusionTexture->strength;
        }

        auto occlusion_image = resolve_texture_cpu(asset, gltf_material.occlusionTexture, false, base_directory,
                                                   image_cache, cpu_images);

        if (!occlusion_image) {
            return std::unexpected(occlusion_image.error());
        }

        material.occlusion_image = *occlusion_image;

        auto emissive_image = resolve_texture_cpu(asset, gltf_material.emissiveTexture, true, base_directory,
                                                  image_cache, cpu_images);

        if (!emissive_image) {
            return std::unexpected(emissive_image.error());
        }

        material.emissive_image = *emissive_image;

        return material;
    }

} // namespace

// ------------------------------------------------------------------------
// Phase 1 — CPU pass. Safe to run on a background thread: touches the
// filesystem, stb_image, meshoptimizer, and MikkTSpace only.
// ------------------------------------------------------------------------

auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage)
        -> std::expected<ModelCpuData, ModelLoadError> {
    fastgltf::GltfFileStream file_stream{path};

    if (!file_stream.isOpen()) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::file_not_found,
        });
    }

    static thread_local fastgltf::Parser parser{fastgltf::Extensions::KHR_materials_emissive_strength};

    constexpr auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices |
                             fastgltf::Options::LoadExternalImages;

    auto const base_directory = path.parent_path();

    // loadGltf (rather than loadGltfBinary) sniffs the header and accepts
    // both .glb (binary) and .gltf (JSON, with external .bin/.png/.jpg
    // siblings) — same call handles both container types.
    auto asset_result = parser.loadGltf(file_stream, base_directory, options);

    if (asset_result.error() != fastgltf::Error::None) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::parse_error,
        });
    }

    auto asset = std::move(asset_result.get());

    ModelCpuData cpu_data;
    cpu_data.meshes.reserve(asset.meshes.size());
    cpu_data.nodes.reserve(asset.nodes.size());
    cpu_data.materials.reserve(asset.materials.size());

    // Dedup textures within this asset: several materials commonly share
    // one atlas/texture, and we only want to decode it once. Keyed by
    // glTF image index -> index into cpu_data.images.
    std::unordered_map<std::size_t, std::size_t> image_cache;

    for (auto const &gltf_material: asset.materials) {
        auto material =
                load_material_cpu(asset, gltf_material, sampler_storage, base_directory, image_cache, cpu_data.images);

        if (!material) {
            return std::unexpected(material.error());
        }

        cpu_data.materials.push_back(std::move(*material));
    }

    for (auto const &gltf_mesh: asset.meshes) {
        ModelCpuMesh mesh;
        mesh.primitives.reserve(gltf_mesh.primitives.size());

        for (auto const &gltf_primitive: gltf_mesh.primitives) {
            auto primitive = load_primitive_cpu(asset, gltf_primitive);

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

    auto const scene_index = asset.defaultScene.value_or(0);

    if (scene_index < asset.scenes.size()) {
        auto const &scene = asset.scenes[scene_index];

        cpu_data.scene_roots.reserve(scene.nodeIndices.size());

        for (auto const node_index: scene.nodeIndices) {
            cpu_data.scene_roots.push_back(static_cast<std::uint32_t>(node_index));
        }
    }

    return cpu_data;
}


auto record_model_gpu_upload(ModelCpuData const &cpu_data, VkCommandBuffer command_buffer,
                             GeometryArena &geometry_arena, ImageStorage &image_storage,
                             MaterialStorage &material_storage) -> std::expected<Model, ModelLoadError> {
    if (command_buffer == VK_NULL_HANDLE) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::invalid_argument,
        });
    }

    std::vector<ImageHandle> image_handles;
    image_handles.reserve(cpu_data.images.size());

    for (auto const &cpu_image: cpu_data.images) {
        auto uploaded = image_storage.create_image(
                ImageCreateInfo{
                        .extent =
                                VkExtent3D{
                                        .width = cpu_image.width,
                                        .height = cpu_image.height,
                                        .depth = 1,
                                },
                        .format = cpu_image.is_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
                        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .image_type = VK_IMAGE_TYPE_2D,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .flags = 0,
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = VK_IMAGE_TILING_OPTIMAL,
                        .mip_levels =
                                static_cast<std::uint32_t>(std::bit_width(std::max(cpu_image.width, cpu_image.height))),
                        .array_layers = 1,
                        .debug_name = "model_texture",
                },
                std::span<std::byte const>{cpu_image.pixels}, command_buffer);

        if (!uploaded || !uploaded->valid()) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::texture_upload_failed,
            });
        }

        image_handles.push_back(*uploaded);
    }

    auto const resolve_image = [&](std::optional<std::size_t> cpu_index, ImageHandle fallback) {
        return cpu_index.has_value() ? image_handles[*cpu_index] : fallback;
    };

    Model model;
    model.meshes.reserve(cpu_data.meshes.size());
    model.materials.reserve(cpu_data.materials.size());

    std::mutex sync_mutex;

    std::vector<std::future<std::expected<MaterialHandle, ModelLoadError>>> material_futures;
    material_futures.reserve(cpu_data.materials.size());

    for (auto const &cpu_material: cpu_data.materials) {
        material_futures.push_back(
                std::async(std::launch::async, [&]() -> std::expected<MaterialHandle, ModelLoadError> {
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
                            .metallic_roughness_texture = resolve_image(cpu_material.metallic_roughness_image,
                                                                        image_storage.metallic_roughness()),
                            .occlusion_texture = resolve_image(cpu_material.occlusion_image, image_storage.occlusion()),
                            .emissive_texture = resolve_image(cpu_material.emissive_image, image_storage.emissive()),
                            .sampler = cpu_material.sampler,
                            .alpha_mode = cpu_material.alpha_mode,
                    };

                    auto gpu_material = to_gpu_material(info);

                    // Synchronize the external insertion
                    std::lock_guard<std::mutex> lock(sync_mutex);
                    auto material_handle = material_storage.create_material(gpu_material);

                    if (!material_handle) {
                        return std::unexpected(ModelLoadError{
                                .type = ModelLoadErrorType::material_creation_failed,
                                .cause = ErrorCause{Boxed<MaterialStorageError>{material_handle.error()}},
                        });
                    }

                    return *material_handle;
                }));
    }

    for (auto &fut: material_futures) {
        auto result = fut.get();
        if (!result)
            return std::unexpected(result.error());
        model.materials.push_back(*result);
    }

    std::vector<std::future<std::expected<ModelMesh, ModelLoadError>>> mesh_futures;
    mesh_futures.reserve(cpu_data.meshes.size());

    for (auto const &cpu_mesh: cpu_data.meshes) {
        mesh_futures.push_back(std::async(std::launch::async, [&]() -> std::expected<ModelMesh, ModelLoadError> {
            ModelMesh mesh;
            mesh.primitives.reserve(cpu_mesh.primitives.size());

            for (auto const &cpu_primitive: cpu_mesh.primitives) {
                auto geometry = [&]() {
                    std::lock_guard<std::mutex> lock(sync_mutex);
                    return geometry_arena.allocate_mesh(std::span<const ModelVertex>{cpu_primitive.vertices},
                                                        std::span<const std::uint32_t>{cpu_primitive.indices});
                }();

                if (!geometry) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::geometry_upload_failed,
                            .cause = ErrorCause{Boxed<GeometryArenaError>{geometry.error()}},
                    });
                }

                if (cpu_primitive.material_index.has_value() &&
                    *cpu_primitive.material_index >= model.materials.size()) {
                    return std::unexpected(ModelLoadError{
                            .type = ModelLoadErrorType::invalid_material_index,
                    });
                }

                mesh.primitives.push_back(ModelPrimitive{
                        .geometry = *geometry,
                        .material_index = cpu_primitive.material_index,
                });
            }

            return mesh;
        }));
    }

    for (auto &fut: mesh_futures) {
        auto result = fut.get();
        if (!result)
            return std::unexpected(result.error());
        model.meshes.push_back(std::move(*result));
    }

    model.nodes = cpu_data.nodes;
    model.scene_roots = cpu_data.scene_roots;
    std::tie(model.bounds_min, model.bounds_max) = compute_model_bounds(cpu_data);

    return model;
}

auto load_model(std::filesystem::path const &path, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                ImageStorage &image_storage, SamplerStorage &sampler_storage, MaterialStorage &material_storage)
        -> std::expected<Model, ModelLoadError> {
    auto cpu_data = load_model_cpu(path, sampler_storage);

    if (!cpu_data) {
        return std::unexpected(cpu_data.error());
    }

    return record_model_gpu_upload(*cpu_data, command_buffer, geometry_arena, image_storage, material_storage);
}
