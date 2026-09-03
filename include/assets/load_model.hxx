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
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hxx"
#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "assets/geometry.hxx"
#include "assets/geometry_arena.hxx"
#include "assets/material.hxx"
#include "assets/model_load_profile.hxx"
#include "gpu/model_vertex.hxx"
#include "gpu/sampler.hxx"
#include "assets/texture_streamer.hxx"

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

    // Transient: true if the glTF primitive already carried a TANGENT
    // accessor. Only meaningful between extract_primitive_cpu() and
    // finalize_primitive_cpu() -- see load_model_cpu_unfinalized() and
    // ModelPrimitiveFinalization -- unused (and left at its default) for a
    // primitive built any other way, since generate_tangents() has either
    // already run or was never needed by the time one exists elsewhere.
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

    // Null unless load_model_cpu() was given one -- rides along into
    // start_model_gpu_upload()/step_model_gpu_upload() and every texture job
    // this model's images kick off, so the whole load's timing breakdown
    // ends up in one place. See ModelStreamer::request().
    std::shared_ptr<ModelLoadProfile> profile;
};

auto generate_tangents(std::vector<ModelVertex> &vertices, std::vector<std::uint32_t> &indices)
        -> std::expected<void, ModelLoadError>;

// Generates simplified index-buffer variants for LOD1..LOD(lod_count-1)
// from `vertices`/`indices` (expected to already be the final, welded and
// optimized LOD0 buffers). Entries stay nullopt when meshopt_simplify
// can't reduce that level's index count at all.
auto generate_mesh_lods(std::vector<ModelVertex> const &vertices, std::vector<std::uint32_t> const &indices)
        -> std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1>;

// `profile`, when non-null, gets the CPU-parse section of its timing
// breakdown filled in (see ModelLoadProfile) and is copied into the
// returned ModelCpuData::profile so later phases (GPU upload, texture
// pipeline) keep writing into the same instance.
[[nodiscard]]
auto load_model_cpu(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                    std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::expected<ModelCpuData, ModelLoadError>;

// Runs load_model_cpu() on thread_pool() instead of blocking the calling
// thread. GPU uploads aren't safe to issue off the render thread, so this
// doesn't create a ModelHandle itself -- pair it with
// IModelSink::create_pending_model()/finish_model_load() (or just use
// ModelStreamer, which already does this) to get a handle that's usable
// immediately and upgrades in place once the background work and GPU
// upload both finish. Bypasses load_model_cpu()'s caller-side path cache --
// the caller is responsible for not requesting the same path twice
// concurrently if that matters for their use case.
//
// `sampler_storage` must outlive the returned future -- the caller is
// responsible for waiting on or discarding it before sampler_storage is
// destroyed.
[[nodiscard]]
auto load_model_cpu_async(std::filesystem::path path, SamplerStorage &sampler_storage,
                          std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::future<std::expected<ModelCpuData, ModelLoadError>>;

// Everything load_model_cpu() does except each primitive's
// finalize_primitive_cpu() step (tangent generation + LOD simplification --
// see ModelCpuPrimitive::has_tangents): primitives come back raw,
// vertices/indices only. load_model_cpu() itself calls this and then
// finalizes every primitive sequentially, in place, to keep its existing
// all-in-one contract; load_model_cpu_async() calls this instead and leaves
// finalizing to the caller (see ModelPrimitiveFinalization below), so it
// can run in parallel across thread_pool() rather than one primitive at a
// time on a single background thread.
[[nodiscard]]
auto load_model_cpu_unfinalized(std::filesystem::path const &path, SamplerStorage &sampler_storage,
                                std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::expected<ModelCpuData, ModelLoadError>;

// Parallel per-primitive finalization (tangent generation + LOD
// simplification) for a ModelCpuData produced by
// load_model_cpu_unfinalized(), produced by start_primitive_finalization()
// and advanced by repeated step_primitive_finalization() calls -- one task
// per primitive across the whole model, submitted to thread_pool() up
// front (unlike ModelGpuUpload, there's no per-frame budget here: this
// never touches the render thread or a command buffer, so thread_pool()
// itself is the only thing pacing it).
//
// Must be started (and stepped) from a thread that is NOT itself a
// thread_pool() worker -- these tasks run on that same pool, and a worker
// blocking on a future for another task submitted to its own pool can
// deadlock if every worker ends up in that same blocked state (see
// BS::thread_pool's own wait() documentation for the identical warning).
// ModelStreamer follows this by driving finalization only from
// process_ready() (the render thread), never from inside
// load_model_cpu_async()'s own task.
struct ModelPrimitiveFinalization {
    ModelCpuData cpu_data;

    // Parallel arrays: tasks[i] is the finalized primitive destined for
    // cpu_data.meshes[targets[i].first].primitives[targets[i].second].
    std::vector<std::future<std::expected<ModelCpuPrimitive, ModelLoadError>>> tasks;
    std::vector<std::pair<std::size_t, std::size_t>> targets;
};

[[nodiscard]]
auto start_primitive_finalization(ModelCpuData cpu_data) -> ModelPrimitiveFinalization;

// Moves every one of `finalization`'s now-ready tasks into its target slot.
// Returns the finished ModelCpuData once every primitive has been
// finalized, or nullopt if there's more outstanding for a later call.
[[nodiscard]]
auto step_primitive_finalization(ModelPrimitiveFinalization &finalization)
        -> std::expected<std::optional<ModelCpuData>, ModelLoadError>;

// Incremental GPU-upload state for one model, produced by
// start_model_gpu_upload() and advanced by repeated step_model_gpu_upload()
// calls -- one call per frame is the intended cadence (see ModelStreamer::
// process_ready()), so a model with many materials/primitives (e.g. Sponza's
// ~25 materials and dozens of mesh primitives) spreads its GPU upload cost
// across several frames instead of spiking a single frame's CPU/GPU work to
// the size of the whole model.
struct ModelGpuUpload {
    ModelCpuData cpu_data;
    std::vector<ImageHandle> image_handles;

    std::vector<MaterialHandle> materials;
    std::vector<ModelMesh> meshes;

    std::size_t material_cursor = 0;
    std::size_t mesh_cursor = 0;
    std::size_t primitive_cursor = 0;
};

// Reserves texture-streamer slots for every image `cpu_data` references --
// cheap, just queues background decode/encode jobs (see
// TextureStreamer::request) -- and returns the initial state for
// step_model_gpu_upload() to advance. Call once per model, on the render
// thread.
[[nodiscard]]
auto start_model_gpu_upload(ModelCpuData cpu_data, ImageStorage &image_storage, TextureStreamer &texture_streamer)
        -> ModelGpuUpload;

// Processes up to `item_budget` materials/primitives of `upload` (one
// material, or one mesh primitive including all its LOD levels, counts as
// one item), recording any GPU copies into `command_buffer`. Returns the
// finished Model once every material and primitive has been processed, or
// nullopt if there's more left for a later call. Must run on the render
// thread -- `command_buffer` must be a currently-recording command buffer
// this frame will submit and wait on through the normal frames-in-flight
// fence discipline (same requirement as ModelStreamer::process_ready()).
[[nodiscard]]
auto step_model_gpu_upload(ModelGpuUpload &upload, VkCommandBuffer command_buffer, GeometryArena &geometry_arena,
                           ImageStorage &image_storage, MaterialStorage &material_storage,
                           std::uint32_t item_budget) -> std::expected<std::optional<Model>, ModelLoadError>;

// Creates geometry/materials synchronously (needs command_buffer for the
// former), but textures merely get requested from `texture_streamer` --
// every material comes back wearing its default textures immediately, with
// the real BC5/BC7 ones swapping in over the following frames as their
// background jobs complete, same as any other TextureStreamer consumer.
// Drives start_model_gpu_upload()/step_model_gpu_upload() to completion in
// one call -- fine for the small procedurally-generated models this is
// still used for (see Renderer::create_model_from_cpu_data), but streamed
// models loaded from disk should go through ModelStreamer instead, which
// paces step_model_gpu_upload() one bounded slice per frame.
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
