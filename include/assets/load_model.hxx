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
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "assets/geometry.hxx"
#include "assets/geometry_arena.hxx"
#include "assets/material.hxx"
#include "assets/meshlet.hxx"
#include "assets/model_load_profile.hxx"
#include "assets/texture_streamer.hxx"
#include "core/config.hxx"
#include "core/error_context.hxx"
#include "core/fly_string.hxx"
#include "core/forward.hxx"
#include "gpu/compressed_texture.hxx"
#include "gpu/model_vertex.hxx"
#include "gpu/sampler.hxx"

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

// The material slot a glTF texture was first resolved into. Decides its TextureRole (colour/generic -> BC7,
// normal -> BC5) and the default it renders as while pending. The first slot to reference an image wins.
enum class ModelTextureSlot : std::uint8_t {
    base_colour,
    normal,
    metallic_roughness,
    occlusion,
    emissive,
};

// Produces an already block-compressed texture, e.g. a TEXR chunk read from an asset pack. Runs on
// thread_pool(), so it must be thread-safe and must not block on the pool.
using CookedTextureLoader = std::function<std::expected<CompressedTexture, TexturePipelineError>()>;

// An image not yet decoded or uploaded. Exactly one of `path` (external file, streamed from disk), `encoded`
// (embedded in the glTF) and `cooked` (pre-compressed, skips the texture pipeline) is set.
struct ModelCpuImageSource {
    std::filesystem::path path;
    std::vector<std::byte> encoded;
    CookedTextureLoader cooked;
    // Identifies the image for de-duplication: required with `encoded` and `cooked`, ignored with `path`.
    std::string cache_key;
    ModelTextureSlot slot = ModelTextureSlot::base_colour;
    FlyString debug_name;
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
    bool double_sided = false;
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

    // Simplified index buffers for LOD1..LOD(lod_count-1), sharing `vertices`. nullopt means no distinct
    // simplification; the upload reuses the previous level's buffer.
    std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> reduced_indices{};

    std::optional<std::uint32_t> material_index;

    // GPU-ready data built off the render thread: packed vertices, and the meshlet split of each level's index
    // buffer (nullopt where the level aliases the previous one).
    std::vector<CompressedModelVertex> compressed_vertices;
    std::array<std::optional<MeshletBuild>, lod_count> meshlets{};

    // Local-space AABB. Set by producers that don't keep `vertices` (cooked assets only carry
    // compressed_vertices); otherwise computed from `vertices`.
    std::optional<std::pair<glm::vec3, glm::vec3>> bounds;

    // Whether the glTF primitive had a TANGENT accessor. Only used between extract_primitive_cpu() and
    // finalize_primitive_cpu().
    bool has_tangents = false;
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

    // Model-space AABB over every vertex, for producers without `vertices`; computed when unset.
    std::optional<std::pair<glm::vec3, glm::vec3>> bounds;

    // Shared by the CPU parse, GPU upload and texture jobs so the whole load's timing ends up in one place.
    std::shared_ptr<ModelLoadProfile> profile;
};

auto generate_tangents(std::vector<ModelVertex> &vertices, std::vector<std::uint32_t> &indices)
        -> std::expected<void, ModelLoadError>;

// Simplified index buffers for LOD1..LOD(lod_count-1) from the final LOD0 buffers. Levels meshopt_simplify
// can't reduce stay nullopt.
auto generate_mesh_lods(std::vector<ModelVertex> const &vertices, std::vector<std::uint32_t> const &indices)
        -> std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1>;

// Fills compressed_vertices and meshlets from the primitive's final geometry. `profile` gets
// vertex_compression_ns and meshlet_build_ns.
auto prepare_primitive_gpu_data(ModelCpuPrimitive &primitive, ModelLoadProfile *profile = nullptr) -> void;

// `profile` gets the CPU-parse timings and is carried into the returned ModelCpuData.
[[nodiscard]]
auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                    std::shared_ptr<ModelLoadProfile> profile = nullptr) -> std::expected<ModelCpuData, ModelLoadError>;

// load_model_cpu() on thread_pool(). Doesn't create a handle, since GPU uploads must happen on the render
// thread; ModelStreamer pairs this with create_pending_model()/finish_model_load(). Bypasses the path cache.
//
// `sampler_storage` must outlive the returned future.
[[nodiscard]]
auto load_model_cpu_async(std::filesystem::path path, SamplerStorage &sampler_storage,
                          std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::future<std::expected<ModelCpuData, ModelLoadError>>;

// load_model_cpu() without per-primitive finalization (tangents, LOD simplification), so the caller can
// finalize primitives in parallel (see ModelPrimitiveFinalization).
[[nodiscard]]
auto load_model_cpu_unfinalized(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                                std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::expected<ModelCpuData, ModelLoadError>;

// Parallel per-primitive finalization of a load_model_cpu_unfinalized() result, one thread_pool() task per
// primitive.
//
// Start and step it from outside thread_pool(): a worker blocking on tasks in its own pool can deadlock.
struct ModelPrimitiveFinalization {
    ModelCpuData cpu_data;

    // tasks[i] goes to cpu_data.meshes[targets[i].first].primitives[targets[i].second].
    std::vector<std::future<std::expected<ModelCpuPrimitive, ModelLoadError>>> tasks;
    std::vector<std::pair<std::size_t, std::size_t>> targets;
};

[[nodiscard]]
auto start_primitive_finalization(ModelCpuData cpu_data) -> ModelPrimitiveFinalization;

// Collects finished tasks. Returns the ModelCpuData once every primitive is done, otherwise nullopt.
[[nodiscard]]
auto step_primitive_finalization(ModelPrimitiveFinalization &finalization)
        -> std::expected<std::optional<ModelCpuData>, ModelLoadError>;

// Incremental GPU upload of one model, stepped once per frame so large models spread their cost.
struct ModelGpuUpload {
    ModelCpuData cpu_data;
    std::vector<ImageHandle> image_handles;

    std::vector<MaterialHandle> materials;
    std::vector<ModelMesh> meshes;

    std::size_t material_cursor = 0;
    std::size_t mesh_cursor = 0;
    std::size_t primitive_cursor = 0;
};

// Requests every texture `cpu_data` references from the streamer and returns the initial upload state. Render
// thread only.
[[nodiscard]]
auto start_model_gpu_upload(ModelCpuData cpu_data, ImageStorage &image_storage, TextureStreamer &texture_streamer)
        -> ModelGpuUpload;

// Processes up to `item_budget` materials/primitives (a primitive with all its LODs is one item), recording
// copies into `command_buffer`. Returns the Model when done, otherwise nullopt. Render thread only;
// `command_buffer` must be recording for this frame.
[[nodiscard]]
auto step_model_gpu_upload(ModelGpuUpload &upload, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                           ImageStorage &image_storage, MaterialStorage &material_storage, std::uint32_t item_budget)
        -> std::expected<std::optional<Model>, ModelLoadError>;

// Runs the whole GPU upload in one call. Textures are only requested, so materials start with default
// textures. Meant for small procedural models; streamed models go through ModelStreamer.
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
