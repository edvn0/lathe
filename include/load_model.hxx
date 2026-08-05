#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <vector>

#include "error_context.hxx"
#include "forward.hxx"

#include "geometry_arena.hxx"
#include "material_storage.hxx"

struct ModelVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec4 tangent{};
    glm::vec2 texcoord{};
};

struct ModelPrimitive {
    MeshGeometry geometry{};
    std::optional<std::uint32_t> material_index{
            std::nullopt}; // If it has -> index into global array, else -> index 0 in global array.

    // Local-space (untransformed) AABB over this primitive's own vertices
    // only -- unlike Model::bounds_min/max, no node transform is baked in,
    // since that's applied per-instance at render time. Used as the culling
    // volume for GPU frustum culling.
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

struct Model {
    std::vector<ModelMesh> meshes;
    std::vector<ModelNode> nodes;
    std::vector<std::uint32_t> scene_roots;
    std::vector<MaterialHandle> materials;

    // Model-space AABB across every vertex of every mesh reachable from
    // scene_roots, with each node's local_transform applied. Lets callers
    // (e.g. physics) size collision volumes to the actual geometry instead
    // of guessing.
    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
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

    std::optional<ErrorCause> cause;
};

struct ModelCpuImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool is_srgb = false;
    std::vector<std::byte> pixels; // RGBA8, tightly packed
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
    std::vector<std::uint32_t> indices;
    std::optional<std::uint32_t> material_index;
};

struct ModelCpuMesh {
    std::vector<ModelCpuPrimitive> primitives;
};

struct ModelCpuData {
    std::vector<ModelCpuMesh> meshes;
    std::vector<ModelCpuMaterial> materials;
    std::vector<ModelCpuImage> images;
    std::vector<ModelNode> nodes;
    std::vector<std::uint32_t> scene_roots;
};

// Generates MikkTSpace tangents for a triangle list, then welds duplicate
// vertices via meshoptimizer. Exposed (not just an internal helper of the
// glTF importer) so procedurally-generated meshes (see primitive_meshes.hxx)
// can produce the same ModelVertex layout without re-deriving tangent math.
auto generate_tangents(std::vector<ModelVertex> &vertices, std::vector<std::uint32_t> &indices)
        -> std::expected<void, ModelLoadError>;

auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage)
        -> std::expected<ModelCpuData, ModelLoadError>;

auto record_model_gpu_upload(ModelCpuData const &cpu_data, VkCommandBuffer command_buffer,
                             GeometryArena &geometry_arena, ImageStorage &image_storage,
                             MaterialStorage &material_storage) -> std::expected<Model, ModelLoadError>;

auto load_model(std::filesystem::path const &path, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                ImageStorage &image_storage, SamplerStorage &sampler_storage, MaterialStorage &material_storage)
        -> std::expected<Model, ModelLoadError>;

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
