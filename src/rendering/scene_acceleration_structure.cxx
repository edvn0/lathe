#include "rendering/scene_acceleration_structure.hxx"

#include <algorithm>
#include <bit>
#include <cstring>
#include <ranges>

#include <tracy/Tracy.hpp>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "gpu/gpu_resource_table.hxx"

namespace {
    constexpr auto min_instance_capacity = std::uint32_t{1024};

    constexpr auto build_input_usage =
            VkBufferUsageFlags{VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR};

    // Row major 3x4 from a column major glm matrix.
    auto transform_of(glm::mat4 const &matrix) -> VkTransformMatrixKHR {
        auto result = VkTransformMatrixKHR{};
        for (auto row = 0; row < 3; ++row) {
            for (auto column = 0; column < 4; ++column) {
                result.matrix[row][column] = matrix[column][row];
            }
        }
        return result;
    }

    auto triangle_geometry(VkDeviceAddress vertices, VkDeviceAddress indices, std::uint32_t vertex_count)
            -> VkAccelerationStructureGeometryKHR {
        return VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry = {.triangles =
                                     {
                                             .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                                             .vertexFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
                                             .vertexData = {.deviceAddress = vertices},
                                             .vertexStride = SceneAccelerationStructure::vertex_stride,
                                             .maxVertex = vertex_count - 1,
                                             .indexType = VK_INDEX_TYPE_UINT32,
                                             .indexData = {.deviceAddress = indices},
                                     }},
                .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        };
    }

    auto memory_barrier(VkCommandBuffer command_buffer, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                        VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) -> void {
        auto const barrier = VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = src_stage,
                .srcAccessMask = src_access,
                .dstStageMask = dst_stage,
                .dstAccessMask = dst_access,
        };
        auto const dependency = VkDependencyInfo{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(command_buffer, &dependency);
    }
}

SceneAccelerationStructure::~SceneAccelerationStructure() { destroy(); }

auto SceneAccelerationStructure::create(VulkanContext &context, GpuResourceTable &resource_table) -> bool {
    if (!context.ray_query_supported) {
        return false;
    }
    context_ = &context;
    resource_table_ = &resource_table;

    auto properties = VkPhysicalDeviceAccelerationStructurePropertiesKHR{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    auto properties2 = VkPhysicalDeviceProperties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                   .pNext = &properties};
    vkGetPhysicalDeviceProperties2(context.physical_device, &properties2);
    scratch_alignment_ = std::max<VkDeviceSize>(properties.minAccelerationStructureScratchOffsetAlignment, 1);
    return true;
}

auto SceneAccelerationStructure::destroy_structure(AccelerationStructure &structure) noexcept -> void {
    if (structure.handle != VK_NULL_HANDLE && context_ != nullptr) {
        vkDestroyAccelerationStructureKHR(context_->device, structure.handle, nullptr);
    }
    structure.handle = VK_NULL_HANDLE;
    structure.storage.destroy();
    structure.address = 0;
    structure.size = 0;
}

// The caller has idled the device (teardown, device loss).
auto SceneAccelerationStructure::destroy() noexcept -> void {
    if (context_ == nullptr) {
        return;
    }
    for (auto &blas: blases_ | std::views::values) {
        destroy_structure(blas.structure);
    }
    blases_.clear();
    for (auto &entry: retired_) {
        destroy_structure(entry.structure);
    }
    retired_.clear();
    for (auto &frame: frames_) {
        for (auto const handle: frame.dynamic_handles) {
            vkDestroyAccelerationStructureKHR(context_->device, handle, nullptr);
        }
        frame.dynamic_storage.destroy();
        destroy_structure(frame.tlas);
        frame.instances.destroy();
        frame.table.destroy();
        frame.scratch.destroy();
        frame = FrameData{};
    }
    context_ = nullptr;
    resource_table_ = nullptr;
}

auto SceneAccelerationStructure::align_scratch(VkDeviceSize value) const noexcept -> VkDeviceSize {
    return (value + scratch_alignment_ - 1) / scratch_alignment_ * scratch_alignment_;
}

auto SceneAccelerationStructure::create_structure(AccelerationStructure &structure,
                                                  VkAccelerationStructureTypeKHR type, VkDeviceSize size,
                                                  char const *name) -> bool {
    auto storage = Buffer::create(*context_, BufferCreateInfo{
                                                     .size = size,
                                                     .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                     .memory = BufferMemory::device,
                                                     .debug_name = name,
                                             });
    if (!storage) {
        return false;
    }
    structure.storage = std::move(*storage);
    structure.size = size;
    auto const create_info = VkAccelerationStructureCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = structure.storage.buffer,
            .offset = 0,
            .size = size,
            .type = type,
    };
    if (vkCreateAccelerationStructureKHR(context_->device, &create_info, nullptr, &structure.handle) != VK_SUCCESS) {
        structure.storage.destroy();
        structure.handle = VK_NULL_HANDLE;
        return false;
    }
    auto const address_info = VkAccelerationStructureDeviceAddressInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = structure.handle,
    };
    structure.address = vkGetAccelerationStructureDeviceAddressKHR(context_->device, &address_info);
    return true;
}

auto SceneAccelerationStructure::retire(Blas &blas) -> void {
    retired_.push_back(Retired{.structure = std::move(blas.structure)});
    blas.structure = AccelerationStructure{};
}

auto SceneAccelerationStructure::begin_frame(std::uint32_t frame_index) -> void {
    if (!ready()) {
        return;
    }
    frame_ = frame_index % frames_in_flight;
    ++frame_counter_;
    staged_instances_.clear();
    staged_table_.clear();
    pending_.clear();
    pending_scratch_ = 0;
    waiting_ = 0;
    dynamic_.clear();
    dynamic_scratch_ = 0;
    dynamic_vertices_ = 0;
    dynamic_skipped_ = 0;
    building_ = true;
    // The skinned structures this slot made two frames ago are done with.
    for (auto const handle: frames_[frame_].dynamic_handles) {
        vkDestroyAccelerationStructureKHR(context_->device, handle, nullptr);
    }
    frames_[frame_].dynamic_handles.clear();

    // Structures a frame in flight may still traverse go only after those frames are done.
    for (auto &entry: retired_) {
        if (entry.frames_remaining > 0) {
            --entry.frames_remaining;
        }
    }
    std::erase_if(retired_, [this](Retired &entry) {
        if (entry.frames_remaining != 0) {
            return false;
        }
        destroy_structure(entry.structure);
        return true;
    });

    // Free what nothing instanced for a while (mesh unload, streaming).
    if (frame_counter_ % 64 == 0) {
        for (auto it = blases_.begin(); it != blases_.end();) {
            if (frame_counter_ - it->second.last_used > idle_frames) {
                retire(it->second);
                it = blases_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

auto SceneAccelerationStructure::make_blas(Geometry const &geometry) -> Blas * {
    // The builder reads positions at vertex_address + 8 with a stride of 20: both must be multiples of the 2 byte
    // component, and the indices 4 byte aligned.
    if (geometry.index_type != VK_INDEX_TYPE_UINT32 || geometry.index_count < 3 || geometry.vertex_count == 0 ||
        ((geometry.vertex_address + position_offset) & 1U) != 0 || (geometry.index_address & 3U) != 0) {
        if (!warned_misaligned_) {
            warned_misaligned_ = true;
            warn("SceneAccelerationStructure: a submesh has geometry the BLAS builder cannot read (index type {}, "
                 "vertex address {:#x}, index address {:#x}); it is left out of the ray tracing scene",
                 static_cast<int>(geometry.index_type), geometry.vertex_address, geometry.index_address);
        }
        return nullptr;
    }

    auto const triangle_count = geometry.index_count / 3;
    auto const shape = triangle_geometry(geometry.vertex_address + position_offset, geometry.index_address,
                                         geometry.vertex_count);
    auto build = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1,
            .pGeometries = &shape,
    };
    auto sizes = VkAccelerationStructureBuildSizesInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(context_->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                            &triangle_count, &sizes);

    auto const scratch = align_scratch(sizes.buildScratchSize);
    if (pending_.size() >= max_builds_per_frame || pending_scratch_ + scratch > max_scratch_per_frame) {
        return nullptr;
    }

    auto blas = Blas{
            .vertex_offset = geometry.vertex_offset,
            .index_offset = geometry.index_offset,
            .vertex_count = geometry.vertex_count,
            .index_count = geometry.index_count,
            .revision = geometry.revision,
            .scratch_size = scratch,
            .last_used = frame_counter_,
    };
    if (!create_structure(blas.structure, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                          sizes.accelerationStructureSize, "scene_blas")) {
        return nullptr;
    }

    auto [it, inserted] = blases_.insert_or_assign(geometry.key, std::move(blas));
    pending_.push_back(PendingBuild{
            .blas = &it->second,
            .vertex_address = geometry.vertex_address + position_offset,
            .index_address = geometry.index_address,
            .scratch_offset = pending_scratch_,
    });
    pending_scratch_ += scratch;
    return &it->second;
}

auto SceneAccelerationStructure::add(Geometry const &geometry, std::uint32_t material_index,
                                     std::span<glm::mat4 const *const> transforms, std::uint32_t flags) -> void {
    if (!ready() || transforms.empty()) {
        return;
    }

    auto *blas = static_cast<Blas *>(nullptr);
    if (auto const found = blases_.find(geometry.key); found != blases_.end()) {
        blas = &found->second;
        if (blas->vertex_offset != geometry.vertex_offset || blas->index_offset != geometry.index_offset ||
            blas->vertex_count != geometry.vertex_count || blas->index_count != geometry.index_count ||
            blas->revision != geometry.revision) {
            // The mesh was re-uploaded somewhere else: the old structure describes data that is gone.
            // A pending build of it this frame would reference a structure about to be retired, so drop it first.
            std::erase_if(pending_, [blas](PendingBuild const &pending) { return pending.blas == blas; });
            retire(*blas);
            blases_.erase(found);
            blas = nullptr;
        }
    }
    if (blas == nullptr) {
        blas = make_blas(geometry);
        if (blas == nullptr) {
            waiting_ += static_cast<std::uint32_t>(transforms.size());
            return;
        }
    }
    blas->last_used = frame_counter_;

    for (auto const *transform: transforms) {
        auto const index = static_cast<std::uint32_t>(staged_table_.size());
        if (index >= (1U << 24U)) {
            return;
        }
        auto instance = VkAccelerationStructureInstanceKHR{};
        instance.transform = transform_of(*transform);
        instance.instanceCustomIndex = index & 0xFFFFFFU;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas->structure.address;
        staged_instances_.push_back(instance);
        staged_table_.push_back(InstanceData{
                .vertex_address = geometry.vertex_address,
                .index_address = geometry.index_address,
                .material_index = material_index,
                .index_count = geometry.index_count,
                .flags = flags,
        });
    }
}

auto SceneAccelerationStructure::add_skinned(SkinnedGeometry const &geometry, std::uint32_t material_index,
                                             glm::mat4 const &transform, std::uint32_t flags) -> void {
    if (!ready() || !building_ || geometry.vertex_count == 0 || geometry.index_count < 3) {
        return;
    }
    auto const index = static_cast<std::uint32_t>(staged_table_.size());
    if (index >= (1U << 24U)) {
        return;
    }
    if (dynamic_vertices_ + geometry.vertex_count > max_skinned_vertices) {
        ++dynamic_skipped_;
        return;
    }
    dynamic_vertices_ += geometry.vertex_count;

    auto const key = (std::uint64_t{geometry.vertex_count} << 32U) | geometry.index_count;
    auto found = dynamic_sizes_.find(key);
    if (found == dynamic_sizes_.end()) {
        auto const triangle_count = geometry.index_count / 3;
        auto const shape = triangle_geometry(0, 0, geometry.vertex_count);
        auto build = VkAccelerationStructureBuildGeometryInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .geometryCount = 1,
                .pGeometries = &shape,
        };
        auto sizes = VkAccelerationStructureBuildSizesInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(context_->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &build, &triangle_count, &sizes);
        found = dynamic_sizes_
                        .emplace(key, std::array<VkDeviceSize, 2>{(sizes.accelerationStructureSize + 255) / 256 * 256,
                                                                  align_scratch(sizes.buildScratchSize)})
                        .first;
    }

    auto instance = VkAccelerationStructureInstanceKHR{};
    instance.transform = transform_of(transform);
    instance.instanceCustomIndex = index & 0xFFFFFFU;
    instance.mask = 0xFF;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    // The reference is filled in by end_frame, when the structure exists.
    dynamic_.push_back(DynamicBuild{.geometry = geometry,
                                    .size = found->second[0],
                                    .scratch = found->second[1],
                                    .instance_slot = static_cast<std::uint32_t>(staged_instances_.size())});
    staged_instances_.push_back(instance);
    staged_table_.push_back(InstanceData{
            .vertex_address = geometry.vertex_address,
            .index_address = geometry.index_address,
            .material_index = material_index,
            .index_count = geometry.index_count,
            .flags = flags,
    });
}

auto SceneAccelerationStructure::end_frame() -> void {
    building_ = false;
    if (!ready()) {
        return;
    }
    ZoneScopedNC("Scene acceleration structure upload", tracy::Color::DeepPink);

    auto &frame = frames_[frame_];

    // Bottom levels of the skinned instances: one storage buffer, a structure each.
    if (!dynamic_.empty()) {
        auto storage_total = VkDeviceSize{0};
        for (auto &build: dynamic_) {
            build.storage_offset = storage_total;
            build.scratch_offset = pending_scratch_ + dynamic_scratch_;
            storage_total += build.size;
            dynamic_scratch_ += build.scratch;
        }
        if (!frame.dynamic_storage.valid() || frame.dynamic_storage.size() < storage_total) {
            frame.dynamic_storage.destroy();
            auto storage = Buffer::create(
                    *context_, BufferCreateInfo{.size = std::bit_ceil(storage_total),
                                                .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                .memory = BufferMemory::device,
                                                .debug_name = "scene_skinned_blas"});
            if (storage) {
                frame.dynamic_storage = std::move(*storage);
            }
        }
        for (auto &build: dynamic_) {
            auto &instance = staged_instances_[build.instance_slot];
            if (!frame.dynamic_storage.valid()) {
                instance.accelerationStructureReference = 0;
                continue;
            }
            auto const create_info = VkAccelerationStructureCreateInfoKHR{
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                    .buffer = frame.dynamic_storage.buffer,
                    .offset = build.storage_offset,
                    .size = build.size,
                    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            };
            if (vkCreateAccelerationStructureKHR(context_->device, &create_info, nullptr, &build.handle) !=
                VK_SUCCESS) {
                build.handle = VK_NULL_HANDLE;
                instance.accelerationStructureReference = 0;
                continue;
            }
            frame.dynamic_handles.push_back(build.handle);
            auto const address_info = VkAccelerationStructureDeviceAddressInfoKHR{
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    .accelerationStructure = build.handle,
            };
            instance.accelerationStructureReference =
                    vkGetAccelerationStructureDeviceAddressKHR(context_->device, &address_info);
        }
    }

    auto const count = static_cast<std::uint32_t>(staged_instances_.size());
    frame.instance_count = count;

    stats_.instances = count;
    stats_.blas_total = static_cast<std::uint32_t>(blases_.size());
    stats_.blas_built_this_frame = static_cast<std::uint32_t>(pending_.size());
    stats_.instances_waiting = waiting_;
    stats_.blas_bytes = 0;
    for (auto const &blas: blases_ | std::views::values) {
        stats_.blas_bytes += blas.structure.size;
    }

    if (count > frame.capacity || frame.tlas.handle == VK_NULL_HANDLE) {
        auto const capacity = std::max(min_instance_capacity, std::bit_ceil(std::max(count, 1U)));

        // This slot's last frame has finished, so its old structure and the set that names it are free to change.
        destroy_structure(frame.tlas);
        frame.instances.destroy();
        frame.table.destroy();

        auto instances = Buffer::create(
                *context_, BufferCreateInfo{.size = VkDeviceSize{capacity} * sizeof(VkAccelerationStructureInstanceKHR),
                                            .usage = build_input_usage,
                                            .memory = BufferMemory::upload,
                                            .debug_name = "scene_tlas_instances"});
        auto table = Buffer::create(*context_,
                                    BufferCreateInfo{.size = VkDeviceSize{capacity} * sizeof(InstanceData),
                                                     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                     .memory = BufferMemory::upload,
                                                     .debug_name = "scene_rt_instance_table"});
        if (!instances || !table) {
            error("SceneAccelerationStructure: could not allocate the instance buffers for {} instances", capacity);
            return;
        }
        frame.instances = std::move(*instances);
        frame.table = std::move(*table);

        auto const shape = VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                           .arrayOfPointers = VK_FALSE,
                                           .data = {.deviceAddress = frame.instances.device_address}}},
                .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        };
        auto build = VkAccelerationStructureBuildGeometryInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .geometryCount = 1,
                .pGeometries = &shape,
        };
        auto sizes = VkAccelerationStructureBuildSizesInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(context_->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &build, &capacity, &sizes);
        if (!create_structure(frame.tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes.accelerationStructureSize,
                              "scene_tlas")) {
            error("SceneAccelerationStructure: could not create a TLAS for {} instances", capacity);
            return;
        }
        frame.capacity = capacity;
        frame.tlas_scratch = align_scratch(sizes.buildScratchSize);
        resource_table_->set_acceleration_structure(frame_, frame.tlas.handle);
        info("Ray tracing scene: TLAS for up to {} instances ({:.1f} MiB)", capacity,
             static_cast<double>(sizes.accelerationStructureSize) / (1024.0 * 1024.0));
    }
    stats_.tlas_bytes = frame.tlas.size;

    if (frame_counter_ % 240 == 1 || !pending_.empty()) {
        if (!pending_.empty() || waiting_ != 0 || frame_counter_ % 240 == 1) {
            info("Ray tracing scene: {} instances ({} waiting for a BLAS), {} BLASes (+{} this frame, {:.1f} MiB), "
                  "TLAS {:.1f} MiB",
                  count, waiting_, blases_.size(), pending_.size(),
                  static_cast<double>(stats_.blas_bytes) / (1024.0 * 1024.0),
                  static_cast<double>(stats_.tlas_bytes) / (1024.0 * 1024.0));
        }
    }

    // One scratch buffer serves the BLAS builds (a region each) and then the TLAS build.
    auto const scratch_needed = std::max(pending_scratch_ + dynamic_scratch_, frame.tlas_scratch) + scratch_alignment_;
    if (!frame.scratch.valid() || frame.scratch.size() < scratch_needed) {
        frame.scratch.destroy();
        auto scratch = Buffer::create(*context_, BufferCreateInfo{.size = std::bit_ceil(scratch_needed),
                                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                  .memory = BufferMemory::device,
                                                                  .debug_name = "scene_as_scratch"});
        if (!scratch) {
            error("SceneAccelerationStructure: could not allocate scratch memory");
            pending_.clear();
            return;
        }
        frame.scratch = std::move(*scratch);
    }

    if (count != 0) {
        std::memcpy(frame.instances.mapped_data(), staged_instances_.data(),
                    staged_instances_.size() * sizeof(VkAccelerationStructureInstanceKHR));
        std::memcpy(frame.table.mapped_data(), staged_table_.data(), staged_table_.size() * sizeof(InstanceData));
        [[maybe_unused]] auto const a = frame.instances.flush();
        [[maybe_unused]] auto const b = frame.table.flush();
    }
}

auto SceneAccelerationStructure::record_build(VkCommandBuffer command_buffer) -> void {
    auto &frame = frames_[frame_];
    if (!ready() || frame.tlas.handle == VK_NULL_HANDLE || !frame.scratch.valid()) {
        return;
    }
    ZoneScopedNC("Scene acceleration structure build", tracy::Color::DeepPink);

    auto const scratch_base = (frame.scratch.device_address + scratch_alignment_ - 1) / scratch_alignment_ *
                              scratch_alignment_;

    // Geometry uploaded earlier this frame (arena copies) and the host written instances, to the builders.
    memory_barrier(command_buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    if (!pending_.empty()) {
        auto builds = std::vector<VkAccelerationStructureBuildGeometryInfoKHR>{};
        auto shapes = std::vector<VkAccelerationStructureGeometryKHR>{};
        auto ranges = std::vector<VkAccelerationStructureBuildRangeInfoKHR>{};
        builds.reserve(pending_.size());
        shapes.reserve(pending_.size());
        ranges.reserve(pending_.size());

        for (auto const &pending: pending_) {
            shapes.push_back(triangle_geometry(pending.vertex_address, pending.index_address,
                                               pending.blas->vertex_count));
            ranges.push_back(VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = pending.blas->index_count / 3});
        }
        for (auto index = std::size_t{0}; index < pending_.size(); ++index) {
            builds.push_back(VkAccelerationStructureBuildGeometryInfoKHR{
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                    .dstAccelerationStructure = pending_[index].blas->structure.handle,
                    .geometryCount = 1,
                    .pGeometries = &shapes[index],
                    .scratchData = {.deviceAddress = scratch_base + pending_[index].scratch_offset},
            });
        }
        auto range_pointers = std::vector<VkAccelerationStructureBuildRangeInfoKHR const *>{};
        range_pointers.reserve(ranges.size());
        for (auto const &range: ranges) {
            range_pointers.push_back(&range);
        }
        vkCmdBuildAccelerationStructuresKHR(command_buffer, static_cast<std::uint32_t>(builds.size()), builds.data(),
                                            range_pointers.data());

    }

    auto dynamic_builds = std::vector<VkAccelerationStructureBuildGeometryInfoKHR>{};
    auto dynamic_shapes = std::vector<VkAccelerationStructureGeometryKHR>{};
    auto dynamic_ranges = std::vector<VkAccelerationStructureBuildRangeInfoKHR>{};
    dynamic_shapes.reserve(dynamic_.size());
    for (auto const &build: dynamic_) {
        if (build.handle == VK_NULL_HANDLE) {
            continue;
        }
        dynamic_shapes.push_back(triangle_geometry(build.geometry.vertex_address + position_offset,
                                                   build.geometry.index_address, build.geometry.vertex_count));
        dynamic_ranges.push_back(
                VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = build.geometry.index_count / 3});
        dynamic_builds.push_back(VkAccelerationStructureBuildGeometryInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .dstAccelerationStructure = build.handle,
                .geometryCount = 1,
                .pGeometries = &dynamic_shapes.back(),
                .scratchData = {.deviceAddress = scratch_base + build.scratch_offset},
        });
    }
    if (!dynamic_builds.empty()) {
        auto range_pointers = std::vector<VkAccelerationStructureBuildRangeInfoKHR const *>{};
        for (auto const &range: dynamic_ranges) {
            range_pointers.push_back(&range);
        }
        vkCmdBuildAccelerationStructuresKHR(command_buffer, static_cast<std::uint32_t>(dynamic_builds.size()),
                                            dynamic_builds.data(), range_pointers.data());
    }

    if (!pending_.empty() || !dynamic_builds.empty()) {
        // The top level reads the bottom levels and reuses the scratch memory.
        memory_barrier(command_buffer, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    }

    auto const shape = VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                       .arrayOfPointers = VK_FALSE,
                                       .data = {.deviceAddress = frame.instances.device_address}}},
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    auto const build = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .dstAccelerationStructure = frame.tlas.handle,
            .geometryCount = 1,
            .pGeometries = &shape,
            .scratchData = {.deviceAddress = scratch_base},
    };
    auto const range = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = frame.instance_count};
    auto const *range_pointer = &range;
    vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build, &range_pointer);

    // Whatever runs later on this queue (ray queries in compute or fragment shaders) sees the finished structure.
    memory_barrier(command_buffer, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}
