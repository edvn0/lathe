#include "rendering/ray_tracing_poc.hxx"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <span>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

namespace {
    constexpr auto subdivisions = 2;
    constexpr auto group_size = std::uint32_t{8};

    struct Mesh {
        std::vector<float> vertices;
        std::vector<std::uint32_t> indices;
    };

    // An icosahedron split `subdivisions` times, every vertex pushed out to the unit sphere.
    auto make_sphere() -> Mesh {
        constexpr auto t = 1.6180339887F;
        auto points = std::vector<glm::vec3>{
                {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
        };
        for (auto &point: points) {
            point = glm::normalize(point);
        }
        auto triangles = std::vector<std::array<std::uint32_t, 3>>{
                {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                {3, 8, 9},  {4, 9, 5},  {2, 4, 11},  {6, 2, 10}, {8, 6, 7},   {9, 8, 1},
        };

        for (auto round = 0; round < subdivisions; ++round) {
            auto midpoints = std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>{};
            auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
                auto const key = std::minmax(a, b);
                auto const [found, inserted] = midpoints.try_emplace(key, static_cast<std::uint32_t>(points.size()));
                if (inserted) {
                    points.push_back(glm::normalize(points[a] + points[b]));
                }
                return found->second;
            };
            auto next = std::vector<std::array<std::uint32_t, 3>>{};
            for (auto const &[a, b, c]: triangles) {
                auto const ab = midpoint(a, b);
                auto const bc = midpoint(b, c);
                auto const ca = midpoint(c, a);
                next.insert(next.end(), {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}});
            }
            triangles = std::move(next);
        }

        auto mesh = Mesh{};
        for (auto const &point: points) {
            mesh.vertices.insert(mesh.vertices.end(), {point.x, point.y, point.z});
        }
        for (auto const &triangle: triangles) {
            mesh.indices.insert(mesh.indices.end(), triangle.begin(), triangle.end());
        }
        return mesh;
    }

    auto device_failure(DeviceError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::device_error,
                .cause = ErrorCause{Boxed<DeviceError>{error}},
        };
    }

    auto vulkan_failure(char const *what, VkResult result) -> RendererError {
        error("{} failed: VkResult {}", what, static_cast<int>(result));
        return RendererError{.type = RendererErrorType::device_error};
    }

    // Host-visible buffer the acceleration structure build reads from.
    auto create_build_input(VulkanContext &context, std::span<std::byte const> bytes, char const *name)
            -> std::expected<Buffer, RendererError> {
        auto buffer = Buffer::create(context, BufferCreateInfo{
                                                      .size = bytes.size(),
                                                      .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                      .memory = BufferMemory::upload,
                                                      .debug_name = name,
                                              });
        if (!buffer) {
            return std::unexpected(device_failure(buffer.error()));
        }
        if (auto const written = buffer->write(0, bytes); !written) {
            return std::unexpected(device_failure(written.error()));
        }
        return std::move(*buffer);
    }

    // Row major 3x4: a uniform scale and a translation.
    auto transform_of(glm::vec3 translation, float scale) -> VkTransformMatrixKHR {
        return VkTransformMatrixKHR{.matrix = {{scale, 0.0F, 0.0F, translation.x},
                                               {0.0F, scale, 0.0F, translation.y},
                                               {0.0F, 0.0F, scale, translation.z}}};
    }

    // Mirrors PushConstants in rt_poc.slang.
    struct Params {
        glm::mat4 inverse_view_projection{1.0F};
        VkDeviceAddress vertices = 0;
        VkDeviceAddress indices = 0;
        ImageRead hdr;
        ImageRead depth;
        ImageWrite result;
        float sun_x = 0.0F;
        float sun_y = 0.0F;
        float sun_z = 0.0F;
        std::uint32_t mode = 0;

        using Layout = PushLayout<&Params::inverse_view_projection, &Params::vertices, &Params::indices,
                                  &Params::hdr, &Params::depth, &Params::result, &Params::sun_x, &Params::sun_y,
                                  &Params::sun_z, &Params::mode>;
    };

    static_assert(GameGraph::push_size<Params>(Params::Layout{}) == 108);
}

RayTracingPoc::~RayTracingPoc() { destroy(); }

auto RayTracingPoc::destroy() noexcept -> void {
    ready_ = false;
    if (device_ == VK_NULL_HANDLE) {
        return;
    }
    for (auto *structure: {&tlas_, &blas_}) {
        if (structure->handle != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(device_, structure->handle, nullptr);
            structure->handle = VK_NULL_HANDLE;
        }
        structure->storage.destroy();
    }
    instances_.destroy();
    indices_.destroy();
    vertices_.destroy();
    device_ = VK_NULL_HANDLE;
}

auto RayTracingPoc::create(Renderer &renderer) -> std::expected<void, RendererError> {
    auto &context = renderer.context();
    if (!context.ray_query_supported) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }
    device_ = context.device;

    auto const mesh = make_sphere();
    auto const triangle_count = static_cast<std::uint32_t>(mesh.indices.size() / 3);

    auto vertices = create_build_input(context, std::as_bytes(std::span{mesh.vertices}), "rt_poc_vertices");
    if (!vertices) {
        return std::unexpected(vertices.error());
    }
    vertices_ = std::move(*vertices);

    auto indices = create_build_input(context, std::as_bytes(std::span{mesh.indices}), "rt_poc_indices");
    if (!indices) {
        return std::unexpected(indices.error());
    }
    indices_ = std::move(*indices);

    // Bottom level: the triangles. Opaque, so traversal never calls back for an any-hit decision.
    auto const triangles = VkAccelerationStructureGeometryTrianglesDataKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = {.deviceAddress = vertices_.device_address},
            .vertexStride = 3 * sizeof(float),
            .maxVertex = static_cast<std::uint32_t>(mesh.vertices.size() / 3 - 1),
            .indexType = VK_INDEX_TYPE_UINT32,
            .indexData = {.deviceAddress = indices_.device_address},
    };
    auto const blas_geometry = VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry = {.triangles = triangles},
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    auto blas_build = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1,
            .pGeometries = &blas_geometry,
    };

    auto blas_sizes = VkAccelerationStructureBuildSizesInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &blas_build, &triangle_count, &blas_sizes);

    // Top level: one instance of the bottom level, translated to where the sphere sits.
    auto instance = VkAccelerationStructureInstanceKHR{};
    instance.transform = transform_of(sphere_centre, sphere_radius);
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    // Filled in below, once the bottom level exists and has an address.

    auto properties = VkPhysicalDeviceAccelerationStructurePropertiesKHR{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    auto properties2 = VkPhysicalDeviceProperties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                   .pNext = &properties};
    vkGetPhysicalDeviceProperties2(context.physical_device, &properties2);
    auto const scratch_alignment = std::max<VkDeviceSize>(properties.minAccelerationStructureScratchOffsetAlignment, 1);

    auto create_structure = [&](AccelerationStructure &structure, VkAccelerationStructureTypeKHR type,
                                VkDeviceSize size, char const *name) -> std::expected<void, RendererError> {
        auto storage = Buffer::create(context, BufferCreateInfo{
                                                       .size = size,
                                                       .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       .memory = BufferMemory::device,
                                                       .debug_name = name,
                                               });
        if (!storage) {
            return std::unexpected(device_failure(storage.error()));
        }
        structure.storage = std::move(*storage);
        auto const create_info = VkAccelerationStructureCreateInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                .buffer = structure.storage.buffer,
                .offset = 0,
                .size = size,
                .type = type,
        };
        if (auto const result = vkCreateAccelerationStructureKHR(context.device, &create_info, nullptr, &structure.handle);
            result != VK_SUCCESS) {
            return std::unexpected(vulkan_failure("vkCreateAccelerationStructureKHR", result));
        }
        auto const address_info = VkAccelerationStructureDeviceAddressInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                .accelerationStructure = structure.handle,
        };
        structure.address = vkGetAccelerationStructureDeviceAddressKHR(context.device, &address_info);
        return {};
    };

    if (auto const created = create_structure(blas_, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                              blas_sizes.accelerationStructureSize, "rt_poc_blas");
        !created) {
        return std::unexpected(created.error());
    }
    instance.accelerationStructureReference = blas_.address;

    auto instances = create_build_input(context, std::as_bytes(std::span{&instance, 1}), "rt_poc_instances");
    if (!instances) {
        return std::unexpected(instances.error());
    }
    instances_ = std::move(*instances);

    auto const tlas_geometry = VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                       .arrayOfPointers = VK_FALSE,
                                       .data = {.deviceAddress = instances_.device_address}}},
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    auto tlas_build = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1,
            .pGeometries = &tlas_geometry,
    };
    auto const instance_count = std::uint32_t{1};
    auto tlas_sizes = VkAccelerationStructureBuildSizesInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &tlas_build, &instance_count, &tlas_sizes);

    if (auto const created = create_structure(tlas_, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                              tlas_sizes.accelerationStructureSize, "rt_poc_tlas");
        !created) {
        return std::unexpected(created.error());
    }

    // One scratch buffer serves both builds; the barrier between them orders its reuse. It is over-allocated so its
    // address can be rounded up to the device's alignment.
    auto scratch = Buffer::create(
            context,
            BufferCreateInfo{
                    .size = std::max(blas_sizes.buildScratchSize, tlas_sizes.buildScratchSize) + scratch_alignment,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    .memory = BufferMemory::device,
                    .debug_name = "rt_poc_scratch",
            });
    if (!scratch) {
        return std::unexpected(device_failure(scratch.error()));
    }
    auto const scratch_address = (scratch->device_address + scratch_alignment - 1) / scratch_alignment * scratch_alignment;

    blas_build.dstAccelerationStructure = blas_.handle;
    blas_build.scratchData.deviceAddress = scratch_address;
    tlas_build.dstAccelerationStructure = tlas_.handle;
    tlas_build.scratchData.deviceAddress = scratch_address;

    auto const blas_range = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = triangle_count};
    auto const tlas_range = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = instance_count};
    auto const *blas_ranges = &blas_range;
    auto const *tlas_ranges = &tlas_range;

    context.one_time_submit([&](VkCommandBuffer command_buffer) {
        vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &blas_build, &blas_ranges);

        // The top level reads the bottom level, and reuses the scratch memory.
        auto const barrier = VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        };
        auto const dependency = VkDependencyInfo{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &tlas_build, &tlas_ranges);
    });
    // one_time_submit waits for the fence, so the scratch memory is free to go when it returns.
    scratch->destroy();

    auto shader = renderer.game_gpu().register_compute({.source = "assets/shaders/rt_poc.slang",
                                                        .entry_point = "main_cs",
                                                        .debug_name = "rt_poc"});
    if (!shader) {
        return std::unexpected(shader.error());
    }
    shader_ = *shader;

    // The shader reads the top level through the bindless set (binding 6), not through a push constant: a device
    // address converted to an acceleration structure in SPIR-V needs SPV_KHR_ray_tracing, which needs the pipeline.
    renderer.resource_table().set_acceleration_structure(tlas_.handle);

    ready_ = true;
    info("Ray tracing PoC ready: {} triangles", triangle_count);
    return {};
}

auto RayTracingPoc::declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection,
                            glm::vec3 sun_direction) -> void {
    if (!ready_ || graph.slot() != GameSlot::after_lighting) {
        return;
    }
    auto const hdr = graph.scene_hdr();
    auto const depth = graph.scene_depth();
    if (!hdr || !depth) {
        return;
    }
    ZoneScopedNC("Ray tracing PoC declare", tracy::Color::DeepPink);

    auto const sun = glm::normalize(sun_direction);
    auto params = Params{
            .inverse_view_projection = glm::inverse(projection * view),
            .vertices = vertices_.device_address,
            .indices = indices_.device_address,
            .hdr = ImageRead{*hdr},
            .depth = ImageRead{*depth},
            .result = ImageWrite{GameImageDesc{.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "rt_poc"}},
            .sun_x = sun.x,
            .sun_y = sun.y,
            .sun_z = sun.z,
            .mode = view_,
    };

    auto const extent = graph.scene_extent();
    graph.add_compute("rt_poc", params, shader_, Threads::of_extent(extent, group_size, group_size),
                      {.label = "Ray tracing PoC"});
    graph.replace_scene_colour(params.result.image);
}
