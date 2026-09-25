#pragma once

#include <volk.h>

#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <type_traits>
#include <vector>

#include "assets/geometry.hxx"
#include "assets/geometry_arena.hxx"
#include "gpu/model_vertex.hxx"

// Meshlet limits shared with assets/shaders/scene_types.slang (MeshletLimits)
// -- the mesh shaders declare their output arrays with exactly these sizes,
// so the two sides must agree. 64/124 is meshoptimizer's recommended split
// for EXT_mesh_shader (124 keeps max_triangles a multiple of 4).
inline constexpr std::uint32_t meshlet_max_vertices = 64;
inline constexpr std::uint32_t meshlet_max_triangles = 124;

// How many meshlets one task-shader workgroup culls (one per lane) before
// DispatchMesh()ing the survivors. Mirrors meshlets_per_task in
// scene_types.slang.
inline constexpr std::uint32_t meshlets_per_task = 32;

// One meshlet as the task/mesh shaders read it. Mirrors Meshlet in
// assets/shaders/scene_types.slang. centre/radius is a local-space bounding
// sphere; cone_axis/cone_cutoff is meshoptimizer's normal cone (backface
// cluster culling) -- cone_cutoff >= 1 means "never cone-cull this one".
// vertex_offset/triangle_offset index into the meshlet's MeshletSlice::data
// array (uint32 units): vertex_count vertex indices, then triangle_count
// packed triangles (i0 | i1 << 8 | i2 << 16, meshlet-local vertex indices).
struct GpuMeshlet {
    glm::vec3 centre{0.0F};
    float radius = 0.0F;
    glm::vec3 cone_axis{0.0F, 0.0F, 1.0F};
    float cone_cutoff = 1.0F;
    std::uint32_t vertex_offset = 0;
    std::uint32_t triangle_offset = 0;
    std::uint32_t vertex_count = 0;
    std::uint32_t triangle_count = 0;
};

static_assert(sizeof(GpuMeshlet) == 48);
static_assert(std::is_trivially_copyable_v<GpuMeshlet>);

// One batch's vkCmdDrawMeshTasksIndirectEXT command. Vulkan only reads the
// three group counts; the rest rides along because draws are issued with
// stride sizeof(GpuTaskCommand) and the task shader reads its own entry
// (via SV_DrawIndex) to map a task group back to an (instance, meshlet
// chunk) pair. Mirrors TaskCommand in assets/shaders/scene_types.slang.
struct GpuTaskCommand {
    std::uint32_t group_count_x = 0;
    std::uint32_t group_count_y = 1;
    std::uint32_t group_count_z = 1;
    std::uint32_t first_instance = 0;
    std::uint32_t instance_count = 0;
    std::uint32_t meshlet_count = 0;
    std::uint32_t _pad0 = 0;
    std::uint32_t _pad1 = 0;
};

static_assert(sizeof(GpuTaskCommand) == 32);
static_assert(offsetof(GpuTaskCommand, instance_count) == 16);
static_assert(std::is_trivially_copyable_v<GpuTaskCommand>);

// Largest per-dimension task group count Vulkan guarantees
// (maxTaskWorkGroupCount's required minimum). Mirrors
// max_task_group_count_x in assets/shaders/frustum_cull.slang.
inline constexpr std::uint32_t max_task_group_count_x = 65535;

// Spreads instance_count * ceil(meshlet_count / meshlets_per_task) task
// groups over X and Y. Must match set_task_group_counts() in
// frustum_cull.slang -- the shadow pass draws CPU-built commands, the main
// view GPU-culled ones, and run_meshlet_task() decodes both the same way.
constexpr auto set_task_group_counts(GpuTaskCommand &command) noexcept -> void {
    auto const chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    auto const total = command.instance_count * chunk_count;

    command.group_count_x = total < max_task_group_count_x ? total : max_task_group_count_x;
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

// The index-buffer-only half of a meshlet build: which vertices and
// triangles each meshlet covers, independent of where those vertices are.
// Kept separate from the bounds (GpuMeshlet) so a topology can be shared by
// several vertex buffers -- TerrainSlotPool builds one topology for its
// canonical chunk index buffer and recomputes only the bounds each time a
// slot's vertices are rewritten.
struct MeshletTopology {
    struct Range {
        std::uint32_t vertex_offset = 0;
        std::uint32_t triangle_offset = 0;
        std::uint32_t vertex_count = 0;
        std::uint32_t triangle_count = 0;
    };

    std::vector<Range> meshlets;

    // Vertex indices (one uint32 each) followed by packed triangles (one
    // uint32 each) -- exactly what gets uploaded as MeshletSlice::data.
    std::vector<std::uint32_t> data;
};

// Splits `indices` into meshlets. With `vertices` non-empty the split is
// spatially aware (meshopt_buildMeshlets, weighted towards tight normal
// cones for better backface culling); with it empty (topology shared by
// vertex buffers not known yet) it falls back to meshopt_buildMeshletsScan,
// which just walks the -- already vertex-cache optimized -- index order.
[[nodiscard]]
auto build_meshlet_topology(std::span<std::uint32_t const> indices, std::size_t vertex_count,
                            std::span<CompressedModelVertex const> vertices = {}) -> MeshletTopology;

// Bounding sphere + normal cone of every meshlet in `topology`, computed
// from the exact (half-float decoded) positions the mesh shader will read.
[[nodiscard]]
auto compute_meshlet_bounds(MeshletTopology const &topology, std::span<CompressedModelVertex const> vertices)
        -> std::vector<GpuMeshlet>;

// Uploads `topology.data` into its own GeometryArena range. `meshlets` is
// left zero -- pair with upload_meshlet_descriptors().
[[nodiscard]]
auto upload_meshlet_data(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                         MeshletTopology const &topology) -> std::expected<GeometrySlice, GeometryArenaError>;

[[nodiscard]]
auto upload_meshlet_descriptors(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                                std::span<GpuMeshlet const> meshlets)
        -> std::expected<GeometrySlice, GeometryArenaError>;

// A finished CPU-side meshlet split of one index buffer that belongs to
// exactly one vertex buffer: topology plus bounds, ready to upload.
struct MeshletBuild {
    MeshletTopology topology;
    std::vector<GpuMeshlet> meshlets;
};

// build_meshlet_topology + compute_meshlet_bounds. Pure CPU and touches no
// shared state, so it is safe (and meant) to run on a loading thread --
// see prepare_primitive_gpu_data() in load_model.hxx. Returns an empty
// build for degenerate input (fewer than one triangle).
[[nodiscard]]
auto build_meshlets(std::span<std::uint32_t const> indices, std::span<CompressedModelVertex const> vertices)
        -> MeshletBuild;

// Render-thread half: uploads a finished build's data and descriptors.
// Fails with invalid_argument for an empty build.
[[nodiscard]]
auto upload_meshlets(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletBuild const &build)
        -> std::expected<MeshletSlice, GeometryArenaError>;
