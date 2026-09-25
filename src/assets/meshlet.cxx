#include "assets/meshlet.hxx"

#include <glm/gtc/packing.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <cstddef>

namespace {

    // Meshlet bounds have to be computed from the positions the mesh shader
    // actually reads, and those are half floats (CompressedModelVertex) --
    // bounds from the original full-precision ModelVertex could be off by
    // enough to cull a meshlet whose decoded triangles are still on screen.
    [[nodiscard]] auto decode_positions(std::span<CompressedModelVertex const> vertices) -> std::vector<glm::vec3> {
        std::vector<glm::vec3> positions;
        positions.reserve(vertices.size());

        for (auto const &vertex: vertices) {
            positions.emplace_back(glm::unpackHalf1x16(vertex.position_x), glm::unpackHalf1x16(vertex.position_y),
                                   glm::unpackHalf1x16(vertex.position_z));
        }

        return positions;
    }

    // Weight towards tight normal cones over pure spatial locality -- the
    // task shaders cone-cull backfacing meshlets in the opaque main view.
    constexpr float meshlet_cone_weight = 0.25F;

} // namespace

auto build_meshlet_topology(std::span<std::uint32_t const> indices, std::size_t vertex_count,
                            std::span<CompressedModelVertex const> vertices) -> MeshletTopology {
    MeshletTopology topology;

    if (indices.size() < 3 || vertex_count == 0) {
        return topology;
    }

    auto const max_meshlets = meshopt_buildMeshletsBound(indices.size(), meshlet_max_vertices, meshlet_max_triangles);

    std::vector<meshopt_Meshlet> meshlets(max_meshlets);
    std::vector<unsigned int> meshlet_vertices(max_meshlets * meshlet_max_vertices);
    std::vector<unsigned char> meshlet_triangles(max_meshlets * meshlet_max_triangles * 3);

    std::size_t meshlet_count = 0;

    if (!vertices.empty()) {
        auto const positions = decode_positions(vertices);

        meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
                                              indices.data(), indices.size(), &positions[0].x, positions.size(),
                                              sizeof(glm::vec3), meshlet_max_vertices, meshlet_max_triangles,
                                              meshlet_cone_weight);
    } else {
        meshlet_count = meshopt_buildMeshletsScan(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(),
                                                  indices.data(), indices.size(), vertex_count, meshlet_max_vertices,
                                                  meshlet_max_triangles);
    }

    meshlets.resize(meshlet_count);

    std::size_t total_vertices = 0;
    std::size_t total_triangles = 0;

    for (auto const &meshlet: meshlets) {
        total_vertices += meshlet.vertex_count;
        total_triangles += meshlet.triangle_count;
    }

    topology.meshlets.reserve(meshlet_count);
    topology.data.reserve(total_vertices + total_triangles);

    for (auto const &meshlet: meshlets) {
        auto const vertex_offset = static_cast<std::uint32_t>(topology.data.size());

        topology.data.insert(topology.data.end(), meshlet_vertices.begin() + meshlet.vertex_offset,
                             meshlet_vertices.begin() + meshlet.vertex_offset + meshlet.vertex_count);

        auto const triangle_offset = static_cast<std::uint32_t>(topology.data.size());

        for (std::uint32_t triangle = 0; triangle < meshlet.triangle_count; ++triangle) {
            auto const *corners = &meshlet_triangles[meshlet.triangle_offset + triangle * 3];

            topology.data.push_back(static_cast<std::uint32_t>(corners[0]) |
                                    (static_cast<std::uint32_t>(corners[1]) << 8U) |
                                    (static_cast<std::uint32_t>(corners[2]) << 16U));
        }

        topology.meshlets.push_back(MeshletTopology::Range{
                .vertex_offset = vertex_offset,
                .triangle_offset = triangle_offset,
                .vertex_count = meshlet.vertex_count,
                .triangle_count = meshlet.triangle_count,
        });
    }

    return topology;
}

auto compute_meshlet_bounds(MeshletTopology const &topology, std::span<CompressedModelVertex const> vertices)
        -> std::vector<GpuMeshlet> {
    auto const positions = decode_positions(vertices);

    std::vector<GpuMeshlet> result;
    result.reserve(topology.meshlets.size());

    std::vector<unsigned char> triangles;
    triangles.reserve(meshlet_max_triangles * 3);

    for (auto const &range: topology.meshlets) {
        triangles.clear();

        for (std::uint32_t triangle = 0; triangle < range.triangle_count; ++triangle) {
            auto const packed = topology.data[range.triangle_offset + triangle];
            triangles.push_back(static_cast<unsigned char>(packed & 0xFFU));
            triangles.push_back(static_cast<unsigned char>((packed >> 8U) & 0xFFU));
            triangles.push_back(static_cast<unsigned char>((packed >> 16U) & 0xFFU));
        }

        auto const bounds = meshopt_computeMeshletBounds(&topology.data[range.vertex_offset], triangles.data(),
                                                         range.triangle_count, &positions[0].x, positions.size(),
                                                         sizeof(glm::vec3));

        result.push_back(GpuMeshlet{
                .centre = {bounds.center[0], bounds.center[1], bounds.center[2]},
                .radius = bounds.radius,
                .cone_axis = {bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]},
                .cone_cutoff = bounds.cone_cutoff,
                .vertex_offset = range.vertex_offset,
                .triangle_offset = range.triangle_offset,
                .vertex_count = range.vertex_count,
                .triangle_count = range.triangle_count,
        });
    }

    return result;
}

auto upload_meshlet_data(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                         MeshletTopology const &topology) -> std::expected<GeometrySlice, GeometryArenaError> {
    auto slice = geometry_arena.allocate_vertices(command_buffer, std::span<std::uint32_t const>{topology.data});

    if (!slice) {
        return std::unexpected(slice.error());
    }

    return slice->bytes;
}

auto upload_meshlet_descriptors(GeometryArena &geometry_arena, VkCommandBuffer command_buffer,
                                std::span<GpuMeshlet const> meshlets)
        -> std::expected<GeometrySlice, GeometryArenaError> {
    // 16-byte aligned so the float3/float pairs in each GpuMeshlet line up
    // with how the shader side reads them.
    auto slice = geometry_arena.allocate_vertices(command_buffer, std::as_bytes(meshlets),
                                                  static_cast<std::uint32_t>(sizeof(GpuMeshlet)), 16);

    if (!slice) {
        return std::unexpected(slice.error());
    }

    return slice->bytes;
}

auto build_meshlets(std::span<std::uint32_t const> indices, std::span<CompressedModelVertex const> vertices)
        -> MeshletBuild {
    MeshletBuild build{.topology = build_meshlet_topology(indices, vertices.size(), vertices), .meshlets = {}};
    build.meshlets = compute_meshlet_bounds(build.topology, vertices);
    return build;
}

auto upload_meshlets(GeometryArena &geometry_arena, VkCommandBuffer command_buffer, MeshletBuild const &build)
        -> std::expected<MeshletSlice, GeometryArenaError> {
    if (build.meshlets.empty()) {
        return std::unexpected(GeometryArenaError{
                .type = GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        });
    }

    auto data = upload_meshlet_data(geometry_arena, command_buffer, build.topology);

    if (!data) {
        return std::unexpected(data.error());
    }

    auto descriptor_slice = upload_meshlet_descriptors(geometry_arena, command_buffer, build.meshlets);

    if (!descriptor_slice) {
        geometry_arena.retire(*data);
        return std::unexpected(descriptor_slice.error());
    }

    return MeshletSlice{
            .descriptors = *descriptor_slice,
            .data = *data,
            .meshlet_count = static_cast<std::uint32_t>(build.meshlets.size()),
    };
}
