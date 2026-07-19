#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <vector>

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

    GeometryArenaError geometry_error{};
    MaterialStorageError material_storage_error{};
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

auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage)
        -> std::expected<ModelCpuData, ModelLoadError>;

auto record_model_gpu_upload(ModelCpuData const &cpu_data, VkCommandBuffer command_buffer,
                             GeometryArena &geometry_arena, ImageStorage &image_storage,
                             MaterialStorage &material_storage) -> std::expected<Model, ModelLoadError>;

auto load_model(std::filesystem::path const &path, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                ImageStorage &image_storage, SamplerStorage &sampler_storage, MaterialStorage &material_storage)
        -> std::expected<Model, ModelLoadError>;
