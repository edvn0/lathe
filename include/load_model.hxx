#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include "config.hxx"
#include "error_context.hxx"
#include "forward.hxx"
#include "geometry.hxx"
#include "geometry_arena.hxx"
#include "material.hxx"
#include "sampler.hxx"
#include "texture_streamer.hxx"

struct ModelVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec4 tangent{};
    glm::vec2 texcoord{};
};
constexpr auto default_vertex_description() {
    std::array<VkVertexInputAttributeDescription, 4> attributes{};
    attributes[0] = {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(ModelVertex, position),
    };
    attributes[1] = {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(ModelVertex, normal),
    };
    attributes[2] = {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(ModelVertex, tangent),
    };
    attributes[3] = {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(ModelVertex, texcoord),
    };
    std::array<VkVertexInputBindingDescription, 1> bindings{};
    bindings[0] = {
            .binding = 0,
            .stride = sizeof(ModelVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    return std::pair{attributes, bindings};
}


auto encode_octahedral(glm::vec3 direction) -> glm::vec2;
auto decode_octahedral(glm::vec2 encoded) -> glm::vec3;


#pragma pack(push, 1)
struct CompressedModelVertex {
    glm::uint32 normal_oct{}; // packSnorm2x16(encode_octahedral(normal))
    glm::uint32 tangent_oct{}; // packSnorm2x16(encode_octahedral(tangent.xyz));
                               // LSB of the packed value doubles as the
                               // handedness sign (tangent.w < 0 ? 1 : 0)
    glm::uint16 position_x{}, position_y{}, position_z{}; // half floats
    glm::uint16 texcoord_u{}, texcoord_v{}; // half floats
    glm::uint16 pad_{};
};
#pragma pack(pop)
static_assert(sizeof(CompressedModelVertex) == 20);

auto compress_vertex(ModelVertex const &vertex) -> CompressedModelVertex;
auto compress_vertices(std::span<ModelVertex const> vertices) -> std::vector<CompressedModelVertex>;

struct ModelPrimitive {
    std::array<MeshGeometry, lod_count> lods{};
    std::optional<std::uint32_t> material_index{std::nullopt};
    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

struct ModelMesh {
    std::vector<ModelPrimitive> primitives;
};

struct ModelNode {
    glm::mat4 local_transform{1.0F};
    std::uint32_t mesh_index = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> children{};
};

enum class ModelLightType : std::uint8_t {
    point,
    spot,
};

struct ModelCpuLight {
    ModelLightType type = ModelLightType::point;

    glm::vec3 position{0.0F};
    glm::vec3 direction{0.0F, -1.0F, 0.0F}; // world-space, only meaningful for spot

    glm::vec3 colour{1.0F};
    float intensity = 1.0F;
    float range = 10.0F;

    float inner_cone_degrees = 20.0F;
    float outer_cone_degrees = 30.0F;
};

struct Model {
    std::vector<ModelMesh> meshes;
    std::vector<ModelNode> nodes;
    std::vector<std::uint32_t> scene_roots;
    std::vector<MaterialHandle> materials;

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};

    std::vector<ModelCpuLight> lights{};
};

enum class ModelLoadErrorType : std::uint8_t {
    file_not_found,
    parse_error,
    unsupported_primitive,
    missing_position,
    invalid_accessor,
    geometry_upload_failed,
    texture_upload_failed,
    tangent_generation_failed,
    unsupported_image_source,
    material_creation_failed,
    invalid_material_index,
    invalid_argument,
    image_decode_failed
};

struct ModelLoadError {
    ModelLoadErrorType type = ModelLoadErrorType::parse_error;
    std::optional<ErrorCause> cause{std::nullopt};
};

// Which material slot a glTF texture was first resolved into. Determines
// both the TextureRole it streams in as (colour/generic -> BC7, normal ->
// BC5) and which ImageStorage default it renders as while pending -- same
// "whichever slot hits an image index first wins" dedup quirk load_material_cpu
// already had for is_srgb, just extended to cover this too.
enum class ModelTextureSlot : std::uint8_t {
    base_colour,
    normal,
    metallic_roughness,
    occlusion,
    emissive,
};

// A texture an image_sources entry hasn't been decoded, compressed, or
// uploaded yet -- only its source is known. Exactly one of `path`/`encoded`
// is populated: `path` for an external file (streamed straight off disk by
// TextureStreamer, no bytes read on the CPU pass), `encoded` for an image
// embedded in the glTF itself (bufferView/data URI, no file to stream from,
// so its still-compressed bytes are captured here instead).
struct ModelCpuImageSource {
    std::filesystem::path path;
    std::vector<std::byte> encoded;
    std::string cache_key; // only meaningful when `encoded` is populated
    ModelTextureSlot slot = ModelTextureSlot::base_colour;
    std::string debug_name;
};

struct ModelCpuMaterial {
    glm::vec4 base_colour_factor{1.0F};
    glm::vec3 emissive_factor{0.0F};
    float emissive_strength = 1.0F;
    float metallic_factor = 1.0F;
    float roughness_factor = 1.0F;
    float alpha_cutoff = 0.5F;
    float normal_scale = 1.0F;
    float occlusion_strength = 1.0F;
    AlphaMode alpha_mode = AlphaMode::opaque;
    SamplerHandle sampler;

    std::optional<std::size_t> base_colour_image;
    std::optional<std::size_t> metallic_roughness_image;
    std::optional<std::size_t> normal_image;
    std::optional<std::size_t> occlusion_image;
    std::optional<std::size_t> emissive_image;
};

struct ModelCpuPrimitive {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices; // LOD0, full detail

    // Simplified index buffers for LOD1..LOD(lod_count-1), sharing
    // `vertices` above. nullopt means "no distinct simplification
    // available for this level" (procedural meshes never populate this,
    // and meshopt_simplify may fail to reduce a level at all) -- the
    // upload step falls back to reusing the previous LOD's index buffer.
    std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> reduced_indices{};

    std::optional<std::uint32_t> material_index;
};

struct ModelCpuMesh {
    std::vector<ModelCpuPrimitive> primitives;
};


struct ModelCpuData {
    std::vector<ModelCpuMesh> meshes;
    std::vector<ModelCpuMaterial> materials;
    std::vector<ModelCpuImageSource> image_sources;
    std::vector<ModelNode> nodes;
    std::vector<std::uint32_t> scene_roots;
    std::vector<ModelCpuLight> lights;
};

auto generate_tangents(std::vector<ModelVertex> &vertices, std::vector<std::uint32_t> &indices)
        -> std::expected<void, ModelLoadError>;

// Generates simplified index-buffer variants for LOD1..LOD(lod_count-1)
// from `vertices`/`indices` (expected to already be the final, welded and
// optimized LOD0 buffers). Entries stay nullopt when meshopt_simplify
// can't reduce that level's index count at all.
auto generate_mesh_lods(std::vector<ModelVertex> const &vertices, std::vector<std::uint32_t> const &indices)
        -> std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1>;

auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage)
        -> std::expected<ModelCpuData, ModelLoadError>;

// Creates geometry/materials synchronously (needs command_buffer for the
// former), but textures merely get requested from `texture_streamer` --
// every material comes back wearing its default textures immediately, with
// the real BC5/BC7 ones swapping in over the following frames as their
// background jobs complete, same as any other TextureStreamer consumer.
auto record_model_gpu_upload(ModelCpuData const &cpu_data, VkCommandBuffer command_buffer,
                             GeometryArena &geometry_arena, ImageStorage &image_storage,
                             TextureStreamer &texture_streamer, MaterialStorage &material_storage)
        -> std::expected<Model, ModelLoadError>;

auto load_model(std::filesystem::path const &path, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                ImageStorage &image_storage, TextureStreamer &texture_streamer, SamplerStorage &sampler_storage,
                MaterialStorage &material_storage) -> std::expected<Model, ModelLoadError>;

template<>
struct std::formatter<ModelLoadErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ModelLoadErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ModelLoadErrorType::file_not_found:
                    return "file_not_found";
                case ModelLoadErrorType::parse_error:
                    return "parse_error";
                case ModelLoadErrorType::unsupported_primitive:
                    return "unsupported_primitive";
                case ModelLoadErrorType::missing_position:
                    return "missing_position";
                case ModelLoadErrorType::invalid_accessor:
                    return "invalid_accessor";
                case ModelLoadErrorType::geometry_upload_failed:
                    return "geometry_upload_failed";
                case ModelLoadErrorType::texture_upload_failed:
                    return "texture_upload_failed";
                case ModelLoadErrorType::tangent_generation_failed:
                    return "tangent_generation_failed";
                case ModelLoadErrorType::unsupported_image_source:
                    return "unsupported_image_source";
                case ModelLoadErrorType::material_creation_failed:
                    return "material_creation_failed";
                case ModelLoadErrorType::invalid_material_index:
                    return "invalid_material_index";
                case ModelLoadErrorType::invalid_argument:
                    return "invalid_argument";
                case ModelLoadErrorType::image_decode_failed:
                    return "image_decode_failed";
            }

            return "unknown_model_load_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
