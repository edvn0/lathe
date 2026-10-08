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

inline constexpr std::uint32_t meshlet_max_vertices = 64;
inline constexpr std::uint32_t meshlet_max_triangles = 124;

inline constexpr std::uint32_t meshlets_per_task = 32;

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

inline constexpr std::uint32_t min_meshlets_for_task_path = meshlets_per_task;

[[nodiscard]] constexpr auto uses_meshlet_path(std::uint32_t meshlet_count) noexcept -> bool {
    return meshlet_count >= min_meshlets_for_task_path;
}

struct GpuDrawCommand {
    std::uint32_t group_count_x = 0;
    std::uint32_t group_count_y = 1;
    std::uint32_t group_count_z = 1;

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

inline constexpr std::uint32_t max_task_group_count_x = 65535;

constexpr auto set_task_group_counts(GpuDrawCommand &command) noexcept -> void {
    auto const chunk_count = (command.meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
    auto const total = command.instance_count * chunk_count;

    command.group_count_x = total < max_task_group_count_x ? total : max_task_group_count_x;
    command.group_count_y = total == 0 ? 1 : (total + command.group_count_x - 1) / command.group_count_x;
    command.group_count_z = 1;
}

struct MeshletTopology {
    struct Range {
        std::uint32_t vertex_offset = 0;
        std::uint32_t triangle_offset = 0;
        std::uint32_t vertex_count = 0;
        std::uint32_t triangle_count = 0;
    };

    std::vector<Range> meshlets;

    std::vector<std::uint32_t> data;
};

[[nodiscard]]
auto build_meshlet_topology(std::span<std::uint32_t const> indices, std::size_t vertex_count,
                            std::span<CompressedModelVertex const> vertices = {}) -> MeshletTopology;

[[nodiscard]]
auto compute_meshlet_bounds(MeshletTopology const &topology, std::span<CompressedModelVertex const> vertices)
        -> std::vector<GpuMeshlet>;

[[nodiscard]]
auto upload_meshlet_data(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletTopology const &topology)
        -> std::expected<GeometrySlice, GeometryArenaError>;

[[nodiscard]]
auto upload_meshlet_descriptors(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                                std::span<GpuMeshlet const> meshlets)
        -> std::expected<GeometrySlice, GeometryArenaError>;

struct MeshletBuild {
    MeshletTopology topology;
    std::vector<GpuMeshlet> meshlets;
};

[[nodiscard]]
auto build_meshlets(std::span<std::uint32_t const> indices, std::span<CompressedModelVertex const> vertices)
        -> MeshletBuild;

[[nodiscard]]
auto upload_meshlets(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletBuild const &build)
        -> std::expected<MeshletSlice, GeometryArenaError>;
