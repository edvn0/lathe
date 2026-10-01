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

// Mirror MeshletLimits in scene_types.slang; the mesh shaders size their outputs from these. 64/124 is
// meshoptimizer's recommendation for EXT_mesh_shader.
inline constexpr std::uint32_t meshlet_max_vertices = 64;
inline constexpr std::uint32_t meshlet_max_triangles = 124;

// Meshlets culled per task workgroup, one per lane.
inline constexpr std::uint32_t meshlets_per_task = 32;

// Mirrors Meshlet in scene_types.slang. centre/radius is a local-space bounding sphere; cone_axis/cone_cutoff
// is the normal cone, with cone_cutoff >= 1 disabling cone culling. vertex_offset/triangle_offset index
// MeshletSlice::data in uints: vertex_count vertex indices, then triangle_count packed triangles
// (i0 | i1 << 8 | i2 << 16, meshlet-local).
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

// Meshes with fewer meshlets are drawn with plain instancing. For small meshes (a grass clump is ~4 meshlets,
// drawn ~20K times) most task lanes idle and the launch overhead outweighs the culling.
inline constexpr std::uint32_t min_meshlets_for_task_path = meshlets_per_task;

[[nodiscard]] constexpr auto uses_meshlet_path(std::uint32_t meshlet_count) noexcept -> bool {
    return meshlet_count >= min_meshlets_for_task_path;
}

// Mirrors DrawCommand in scene_types.slang. Each scene pass issues vkCmdDrawMeshTasksIndirectEXT at offset 0
// and vkCmdDrawIndexedIndirect at offset 12 over the same commands; each batch zeroes the half it doesn't use.
// instance_count/first_instance are shared, so main_cs culls both at once. The task shader reads meshlet_count
// via SV_DrawIndex.
struct GpuDrawCommand {
    // VkDrawMeshTasksIndirectCommandEXT
    std::uint32_t group_count_x = 0;
    std::uint32_t group_count_y = 1;
    std::uint32_t group_count_z = 1;

    // VkDrawIndexedIndirectCommand
    std::uint32_t index_count = 0;
    std::uint32_t instance_count = 0;
    std::uint32_t first_index = 0;
    std::int32_t vertex_offset = 0;
    std::uint32_t first_instance = 0;

    std::uint32_t meshlet_count = 0;
    std::uint32_t _pad0 = 0;
};

inline constexpr std::size_t indexed_command_offset = 12;

static_assert(sizeof(GpuDrawCommand) == 40);
static_assert(offsetof(GpuDrawCommand, index_count) == indexed_command_offset);
static_assert(std::is_trivially_copyable_v<GpuDrawCommand>);

// Guaranteed minimum of maxTaskWorkGroupCount. Mirrors frustum_cull.slang.
inline constexpr std::uint32_t max_task_group_count_x = 65535;

// Spreads instance_count * ceil(meshlet_count / meshlets_per_task) task groups over X and Y. Must match
// set_task_group_counts() in frustum_cull.slang.
constexpr auto set_task_group_counts(GpuDrawCommand &command) noexcept -> void {
    auto const chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    auto const total = command.instance_count * chunk_count;

    command.group_count_x = total < max_task_group_count_x ? total : max_task_group_count_x;
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

// Which vertices and triangles each meshlet covers, without bounds, so one topology can serve several vertex
// buffers (TerrainSlotPool shares one across its chunks).
struct MeshletTopology {
    struct Range {
        std::uint32_t vertex_offset = 0;
        std::uint32_t triangle_offset = 0;
        std::uint32_t vertex_count = 0;
        std::uint32_t triangle_count = 0;
    };

    std::vector<Range> meshlets;

    // Vertex indices then packed triangles, uploaded as MeshletSlice::data.
    std::vector<std::uint32_t> data;
};

// Splits `indices` into meshlets: spatially with `vertices` (favouring tight normal cones), otherwise by
// walking the index order.
[[nodiscard]]
auto build_meshlet_topology(std::span<std::uint32_t const> indices, std::size_t vertex_count,
                            std::span<CompressedModelVertex const> vertices = {}) -> MeshletTopology;

// Bounding sphere and normal cone per meshlet, from the decoded half-float positions the mesh shader reads.
[[nodiscard]]
auto compute_meshlet_bounds(MeshletTopology const &topology, std::span<CompressedModelVertex const> vertices)
        -> std::vector<GpuMeshlet>;

// Uploads `topology.data`. `meshlets` is left zero; pair with upload_meshlet_descriptors().
[[nodiscard]]
auto upload_meshlet_data(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletTopology const &topology)
        -> std::expected<GeometrySlice, GeometryArenaError>;

[[nodiscard]]
auto upload_meshlet_descriptors(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                                std::span<GpuMeshlet const> meshlets)
        -> std::expected<GeometrySlice, GeometryArenaError>;

// A CPU-side meshlet split for one vertex buffer, ready to upload.
struct MeshletBuild {
    MeshletTopology topology;
    std::vector<GpuMeshlet> meshlets;
};

// Pure CPU, meant for loading threads. Returns an empty build for fewer than one triangle.
[[nodiscard]]
auto build_meshlets(std::span<std::uint32_t const> indices, std::span<CompressedModelVertex const> vertices)
        -> MeshletBuild;

// Render thread. Fails with invalid_argument for an empty build.
[[nodiscard]]
auto upload_meshlets(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletBuild const &build)
        -> std::expected<MeshletSlice, GeometryArenaError>;
