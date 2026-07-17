#include "load_model.hxx"

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <meshoptimizer.h>
#include <mikktspace.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

    auto to_glm(fastgltf::math::fmat4x4 const &matrix) noexcept -> glm::mat4 {
        glm::mat4 result{1.0F};

        for (std::size_t column = 0; column < std::size_t{4}; ++column) {
            for (std::size_t row = 0; row < std::size_t{4}; ++row) {
                auto c = static_cast<glm::length_t>(column);
                auto r = static_cast<glm::length_t>(row);
                result[c][r] = matrix[column][row];
            }
        }

        return result;
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
    // MikkTSpace tangent generation
    //
    // MikkTSpace evaluates tangents per triangle corner (face/vert), not per
    // unique vertex. A vertex shared by two triangles on opposite sides of a
    // UV seam legitimately wants two different tangents. Feeding it our
    // indexed buffer directly would let the second triangle's callback
    // silently overwrite the first triangle's result at that shared index.
    //
    // So: expand to an unindexed, per-face-vertex buffer, run MikkTSpace
    // over that, then re-weld with meshoptimizer's vertex remap. Seam
    // vertices end up split into distinct vertices automatically, same as
    // if the seam had been authored that way from the start.
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

    auto load_primitive(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive,
                        GeometryArena &geometry_arena) -> std::expected<ModelPrimitive, ModelLoadError> {
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

            // Safe defaults for optional glTF attributes.
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

        // Only synthesize tangents when the source asset didn't ship its
        // own — an authored TANGENT attribute is always preferable to a
        // generated one.
        if (!has_tangents) {
            auto tangent_result = generate_tangents(vertices, indices);

            if (!tangent_result) {
                return std::unexpected(tangent_result.error());
            }
        }

        auto geometry = geometry_arena.allocate_mesh(std::span<const ModelVertex>{vertices},
                                                     std::span<const std::uint32_t>{indices});

        if (!geometry) {
            return std::unexpected(ModelLoadError{
                    .type = ModelLoadErrorType::geometry_upload_failed,
                    .geometry_error = geometry.error(),
            });
        }

        return ModelPrimitive{
                .geometry = *geometry,
                .material_index =
                        primitive.materialIndex.has_value() ? static_cast<std::uint32_t>(*primitive.materialIndex) : 0,
        };
    }

} // namespace

auto load_model(std::filesystem::path const &path, GeometryArena &geometry_arena)
        -> std::expected<Model, ModelLoadError> {
    fastgltf::GltfFileStream file_stream{path};

    if (!file_stream.isOpen()) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::file_not_found,
        });
    }

    static fastgltf::Parser parser;

    constexpr auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices;

    auto asset_result = parser.loadGltfBinary(file_stream, path.parent_path(), options);

    if (asset_result.error() != fastgltf::Error::None) {
        return std::unexpected(ModelLoadError{
                .type = ModelLoadErrorType::parse_error,
        });
    }

    auto asset = std::move(asset_result.get());

    Model model;
    model.meshes.reserve(asset.meshes.size());
    model.nodes.reserve(asset.nodes.size());

    for (auto const &gltf_mesh: asset.meshes) {
        ModelMesh mesh;
        mesh.primitives.reserve(gltf_mesh.primitives.size());

        for (auto const &gltf_primitive: gltf_mesh.primitives) {
            auto primitive = load_primitive(asset, gltf_primitive, geometry_arena);

            if (!primitive) {
                return std::unexpected(primitive.error());
            }

            mesh.primitives.push_back(std::move(*primitive));
        }

        model.meshes.push_back(std::move(mesh));
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

        model.nodes.push_back(std::move(node));
    }

    auto const scene_index = asset.defaultScene.value_or(0);

    if (scene_index < asset.scenes.size()) {
        auto const &scene = asset.scenes[scene_index];

        model.scene_roots.reserve(scene.nodeIndices.size());

        for (auto const node_index: scene.nodeIndices) {
            model.scene_roots.push_back(static_cast<std::uint32_t>(node_index));
        }
    }

    return model;
}
