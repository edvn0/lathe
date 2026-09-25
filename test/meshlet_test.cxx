#include <doctest/doctest.h>

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include "assets/load_model.hxx"
#include "assets/meshlet.hxx"
#include "assets/primitive_meshes.hxx"
#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_mesh.hxx"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

//
// build_meshlet_topology/compute_meshlet_bounds/set_task_group_counts are
// pure CPU functions, and every scene pass draws exclusively through the
// meshlets they produce -- a dropped or duplicated triangle here is a hole
// or z-fight on screen, and a too-small bounding sphere is a meshlet the
// task shader culls while it's still visible.
//

namespace {

    [[nodiscard]] auto terrain_vertices() -> std::vector<CompressedModelVertex> {
        TerrainParams params{};
        params.height_range_min = -2.0F;
        params.height_range_max = 2.0F;

        TerrainField const field{params};
        return make_terrain_chunk(field,
                                  TerrainChunkRequest{
                                          .world_origin_x = 128.0F,
                                          .world_origin_z = -256.0F,
                                          .cell_size = 2.0F,
                                  })
                .vertices;
    }

    [[nodiscard]] auto decode_position(CompressedModelVertex const &vertex) -> glm::vec3 {
        return glm::vec3{glm::unpackHalf1x16(vertex.position_x), glm::unpackHalf1x16(vertex.position_y),
                         glm::unpackHalf1x16(vertex.position_z)};
    }

    using Triangle = std::array<std::uint32_t, 3>;

    // Rotated so the smallest index comes first -- keeps winding, drops
    // which corner a triangle happens to start at.
    [[nodiscard]] auto canonical(Triangle triangle) -> Triangle {
        auto const first = std::ranges::min_element(triangle) - triangle.begin();
        std::ranges::rotate(triangle, triangle.begin() + first);
        return triangle;
    }

    [[nodiscard]] auto source_triangles(std::vector<std::uint32_t> const &indices) -> std::vector<Triangle> {
        std::vector<Triangle> triangles;

        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            triangles.push_back(canonical({indices[i], indices[i + 1], indices[i + 2]}));
        }

        std::ranges::sort(triangles);
        return triangles;
    }

    [[nodiscard]] auto meshlet_triangles(MeshletTopology const &topology) -> std::vector<Triangle> {
        std::vector<Triangle> triangles;

        for (auto const &range: topology.meshlets) {
            for (std::uint32_t triangle = 0; triangle < range.triangle_count; ++triangle) {
                auto const packed = topology.data[range.triangle_offset + triangle];
                auto const corner = [&](std::uint32_t shift) {
                    return topology.data[range.vertex_offset + ((packed >> shift) & 0xFFU)];
                };

                triangles.push_back(canonical({corner(0), corner(8), corner(16)}));
            }
        }

        std::ranges::sort(triangles);
        return triangles;
    }

} // namespace

TEST_CASE("meshlets cover every triangle exactly once, with winding preserved") {
    auto const &indices = terrain_chunk_indices();
    auto const vertices = terrain_vertices();

    SUBCASE("spatial split (with positions)") {
        auto const topology = build_meshlet_topology(indices, vertices.size(), vertices);
        CHECK(meshlet_triangles(topology) == source_triangles(indices));
    }

    SUBCASE("scan split (topology only, as TerrainSlotPool builds it)") {
        auto const topology = build_meshlet_topology(indices, vertices.size());
        CHECK(meshlet_triangles(topology) == source_triangles(indices));
    }
}

TEST_CASE("meshlets respect the mesh shader output limits") {
    auto const &indices = terrain_chunk_indices();
    auto const vertices = terrain_vertices();
    auto const topology = build_meshlet_topology(indices, vertices.size(), vertices);

    REQUIRE_FALSE(topology.meshlets.empty());

    for (auto const &range: topology.meshlets) {
        CHECK(range.vertex_count > 0);
        CHECK(range.vertex_count <= meshlet_max_vertices);
        CHECK(range.triangle_count > 0);
        CHECK(range.triangle_count <= meshlet_max_triangles);
        CHECK(range.vertex_offset + range.vertex_count <= topology.data.size());
        CHECK(range.triangle_offset + range.triangle_count <= topology.data.size());

        for (std::uint32_t triangle = 0; triangle < range.triangle_count; ++triangle) {
            auto const packed = topology.data[range.triangle_offset + triangle];
            CHECK((packed >> 24U) == 0U);
            CHECK((packed & 0xFFU) < range.vertex_count);
            CHECK(((packed >> 8U) & 0xFFU) < range.vertex_count);
            CHECK(((packed >> 16U) & 0xFFU) < range.vertex_count);
        }
    }
}

TEST_CASE("meshlet bounding spheres contain every decoded vertex they draw") {
    auto const &indices = terrain_chunk_indices();
    auto const vertices = terrain_vertices();
    auto const topology = build_meshlet_topology(indices, vertices.size(), vertices);
    auto const meshlets = compute_meshlet_bounds(topology, vertices);

    REQUIRE(meshlets.size() == topology.meshlets.size());

    for (std::size_t i = 0; i < meshlets.size(); ++i) {
        auto const &meshlet = meshlets[i];
        auto const &range = topology.meshlets[i];

        CHECK(meshlet.vertex_offset == range.vertex_offset);
        CHECK(meshlet.triangle_offset == range.triangle_offset);
        CHECK(meshlet.vertex_count == range.vertex_count);
        CHECK(meshlet.triangle_count == range.triangle_count);

        for (std::uint32_t v = 0; v < range.vertex_count; ++v) {
            auto const position = decode_position(vertices[topology.data[range.vertex_offset + v]]);
            CHECK(glm::distance(position, meshlet.centre) <= meshlet.radius * 1.0001F + 1e-5F);
        }
    }
}

TEST_CASE("task group counts cover every (instance, meshlet chunk) within per-dimension limits") {
    for (std::uint32_t const instance_count: {0U, 1U, 7U, 65'535U, 65'536U, 250'000U}) {
        for (std::uint32_t const meshlet_count: {1U, 31U, 32U, 33U, 200U}) {
            GpuTaskCommand command{.instance_count = instance_count, .meshlet_count = meshlet_count};
            set_task_group_counts(command);

            auto const chunks = (meshlet_count + meshlets_per_task - 1) / meshlets_per_task;
            auto const needed = static_cast<std::uint64_t>(instance_count) * chunks;
            auto const dispatched =
                    static_cast<std::uint64_t>(command.group_count_x) * command.group_count_y * command.group_count_z;

            CAPTURE(instance_count);
            CAPTURE(meshlet_count);
            CHECK(command.group_count_x <= max_task_group_count_x);
            CHECK(command.group_count_y <= max_task_group_count_x);
            CHECK(command.group_count_z == 1);
            CHECK(dispatched >= needed);

            // At most one partial row of no-op groups.
            CHECK(dispatched - needed < std::max<std::uint64_t>(command.group_count_x, 1));
        }
    }
}

TEST_CASE("prepare_primitive_gpu_data builds meshlets exactly for levels with their own index buffer") {
    auto capsule = make_capsule_mesh(16, 8);
    REQUIRE(capsule.has_value());

    SUBCASE("procedural meshes arrive prepared from to_model_cpu_data") {
        auto const cpu_data = to_model_cpu_data(*capsule);
        auto const &primitive = cpu_data.meshes.front().primitives.front();

        CHECK(primitive.compressed_vertices.size() == primitive.vertices.size());
        REQUIRE(primitive.meshlets[0].has_value());
        CHECK_FALSE(primitive.meshlets[0]->meshlets.empty());

        for (std::size_t level = 1; level < lod_count; ++level) {
            CHECK_FALSE(primitive.meshlets[level].has_value());
        }
    }

    SUBCASE("reduced LODs get their own build, missing ones alias") {
        ModelCpuPrimitive primitive{.vertices = capsule->vertices, .indices = capsule->indices};

        // Every other triangle -- any valid, distinct index buffer will do.
        std::vector<std::uint32_t> reduced;
        for (std::size_t i = 0; i + 2 < primitive.indices.size(); i += 6) {
            reduced.insert(reduced.end(), primitive.indices.begin() + static_cast<std::ptrdiff_t>(i),
                           primitive.indices.begin() + static_cast<std::ptrdiff_t>(i + 3));
        }
        primitive.reduced_indices[0] = reduced;

        prepare_primitive_gpu_data(primitive);

        REQUIRE(primitive.meshlets[0].has_value());
        REQUIRE(primitive.meshlets[1].has_value());
        CHECK(meshlet_triangles(primitive.meshlets[1]->topology) == source_triangles(reduced));

        for (std::size_t level = 2; level < lod_count; ++level) {
            CHECK_FALSE(primitive.meshlets[level].has_value());
        }
    }
}
