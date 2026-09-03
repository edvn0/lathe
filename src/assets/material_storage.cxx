#include "assets/material_storage.hxx"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

    auto make_error(MaterialStorageErrorType type) -> MaterialStorageError {
        return MaterialStorageError{
                .type = type,
        };
    }

    auto make_device_error(DeviceError error) -> MaterialStorageError {
        return MaterialStorageError{
                .type = MaterialStorageErrorType::device_error,
                .cause = ErrorCause{Boxed<DeviceError>{std::move(error)}},
        };
    }

    auto checked_buffer_size(std::uint32_t capacity) -> std::expected<VkDeviceSize, MaterialStorageError> {
        constexpr auto stride = static_cast<VkDeviceSize>(sizeof(GpuMaterial));

        constexpr auto maximum = std::numeric_limits<VkDeviceSize>::max();

        if (capacity == 0 || static_cast<VkDeviceSize>(capacity) > maximum / stride) {
            return std::unexpected(make_error(MaterialStorageErrorType::invalid_argument));
        }

        return static_cast<VkDeviceSize>(capacity) * stride;
    }

} // namespace

MaterialStorage::MaterialStorage(MaterialStorage &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), slots_(std::move(other.slots_)),
    gpu_buffer_(std::move(other.gpu_buffer_)), upload_buffer_(std::move(other.upload_buffer_)) {}

auto MaterialStorage::operator=(MaterialStorage &&other) noexcept -> MaterialStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);
    slots_ = std::move(other.slots_);
    gpu_buffer_ = std::move(other.gpu_buffer_);
    upload_buffer_ = std::move(other.upload_buffer_);

    return *this;
}

auto MaterialStorage::create(VulkanContext &context, MaterialStorageCreateInfo const &create_info)
        -> std::expected<MaterialStorage, MaterialStorageError> {
    // Slot zero is permanently reserved as the default material.
    if (create_info.capacity < 2) {
        return std::unexpected(make_error(MaterialStorageErrorType::invalid_argument));
    }

    auto size_result = checked_buffer_size(create_info.capacity);

    if (!size_result) {
        return std::unexpected(size_result.error());
    }

    auto const size = *size_result;

    std::string const gpu_name = std::string{create_info.debug_name} + ".gpu";

    auto gpu_buffer = Buffer::create(context, BufferCreateInfo{
                                                      .size = size,
                                                      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                      .memory = BufferMemory::device,
                                                      .debug_name = gpu_name,
                                              });

    if (!gpu_buffer) {
        return std::unexpected(make_device_error(gpu_buffer.error()));
    }

    std::string const upload_name = std::string{create_info.debug_name} + ".upload";

    auto upload_buffer = Buffer::create(context, BufferCreateInfo{
                                                         .size = size,
                                                         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                         .memory = BufferMemory::upload,
                                                         .debug_name = upload_name,
                                                 });

    if (!upload_buffer) {
        gpu_buffer->destroy();

        return std::unexpected(make_device_error(upload_buffer.error()));
    }

    MaterialStorage storage;

    storage.context_ = &context;
    storage.gpu_buffer_ = std::move(*gpu_buffer);
    storage.upload_buffer_ = std::move(*upload_buffer);
    storage.slots_ = ObjectPool<MaterialSlotData, 0>::create(create_info.capacity);

    // Slot zero is permanently reserved as the default material: it's the
    // very first allocation out of a freshly-created pool, so it always
    // lands on index 0 with generation 1, and this is the only allocate()
    // call MaterialStorage ever makes without exposing a way to release it
    // (see the explicit handle.index == 0 guard in destroy_material below).
    auto &default_slot = storage.slots_.allocate()->second;

    default_slot.material = GpuMaterial{};
    default_slot.dirty = true;

    return storage;
}

auto MaterialStorage::create_material(GpuMaterial const &material)
        -> std::expected<MaterialHandle, MaterialStorageError> {
    auto allocation = slots_.allocate();

    if (!allocation) {
        return std::unexpected(make_error(MaterialStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot.material = material;
    slot.dirty = true;

    return handle;
}

auto MaterialStorage::update_material(MaterialHandle handle, GpuMaterial const &material)
        -> std::expected<void, MaterialStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(MaterialStorageErrorType::invalid_handle));
    }

    slot->material = material;
    slot->dirty = true;

    return {};
}

auto MaterialStorage::destroy_material(MaterialHandle handle) -> std::expected<void, MaterialStorageError> {
    // Slot zero is the permanent default material.
    if (handle.index == 0) {
        return std::unexpected(make_error(MaterialStorageErrorType::invalid_handle));
    }

    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(MaterialStorageErrorType::invalid_handle));
    }

    // ObjectPool::release() deliberately leaves the payload as-is (see its
    // doc comment), so zero the published GPU data ourselves before freeing
    // the slot, matching the pre-refactor behavior of never leaving a freed
    // slot's last material contents sitting in the GPU buffer.
    slot->material = GpuMaterial{};
    slot->dirty = true;

    static_cast<void>(slots_.release(handle));

    return {};
}

auto MaterialStorage::get(MaterialHandle handle) const noexcept -> GpuMaterial const * {
    auto const *slot = slots_.get(handle);

    return slot != nullptr ? &slot->material : nullptr;
}

auto MaterialStorage::gpu_index(MaterialHandle handle) const noexcept -> std::uint32_t {
    return slots_.contains(handle) ? handle.index : 0;
}

auto MaterialStorage::prepare_frame(VkCommandBuffer command_buffer, std::uint32_t frame_index)
        -> std::expected<void, MaterialStorageError> {
    static_cast<void>(frame_index);

    if (command_buffer == VK_NULL_HANDLE || upload_buffer_.allocation_info.pMappedData == nullptr) {
        return std::unexpected(make_error(MaterialStorageErrorType::invalid_argument));
    }

    struct DirtyRange {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    std::vector<DirtyRange> dirty_ranges;
    dirty_ranges.reserve(slots_.size());

    auto *mapped = static_cast<std::byte *>(upload_buffer_.allocation_info.pMappedData);

    bool range_open = false;
    DirtyRange current_range{};

    for (std::uint32_t index = 0; index < slots_.capacity(); ++index) {
        auto &slot = *slots_.get_at(index);

        if (!slot.dirty) {
            if (range_open) {
                dirty_ranges.push_back(current_range);
                current_range = {};
                range_open = false;
            }

            continue;
        }

        auto const byte_offset = static_cast<std::size_t>(index) * sizeof(GpuMaterial);

        std::memcpy(mapped + byte_offset, &slot.material, sizeof(GpuMaterial));

        if (!range_open) {
            current_range = DirtyRange{
                    .first = index,
                    .count = 1,
            };

            range_open = true;
        } else {
            ++current_range.count;
        }
    }

    if (range_open) {
        dirty_ranges.push_back(current_range);
    }

    if (dirty_ranges.empty()) {
        return {};
    }

    std::vector<VkBufferCopy2> copy_regions;
    copy_regions.reserve(dirty_ranges.size());

    for (auto const &range: dirty_ranges) {
        auto const offset = static_cast<VkDeviceSize>(range.first) * sizeof(GpuMaterial);

        auto const size = static_cast<VkDeviceSize>(range.count) * sizeof(GpuMaterial);

        copy_regions.push_back(VkBufferCopy2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                .pNext = nullptr,
                .srcOffset = offset,
                .dstOffset = offset,
                .size = size,
        });
    }

    VkCopyBufferInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .pNext = nullptr,
            .srcBuffer = upload_buffer_.buffer,
            .dstBuffer = gpu_buffer_.buffer,
            .regionCount = static_cast<std::uint32_t>(copy_regions.size()),
            .pRegions = copy_regions.data(),
    };

    vkCmdCopyBuffer2(command_buffer, &copy_info);

    VkBufferMemoryBarrier2 const barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = gpu_buffer_.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
    };

    VkDependencyInfo const dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
            .imageMemoryBarrierCount = 0,
            .pImageMemoryBarriers = nullptr,
    };

    vkCmdPipelineBarrier2(command_buffer, &dependency_info);

    for (auto const &range: dirty_ranges) {
        auto const end = range.first + range.count;

        for (auto index = range.first; index < end; ++index) {
            slots_.get_at(index)->dirty = false;
        }
    }

    return {};
}

auto MaterialStorage::destroy() noexcept -> void {
    upload_buffer_.destroy();
    gpu_buffer_.destroy();

    slots_ = ObjectPool<MaterialSlotData, 0>{};

    context_ = nullptr;
}

auto to_gpu_material(MaterialCreateInfo const &create_info) noexcept -> GpuMaterial {
    return {
            .base_colour_factor = create_info.base_colour_factor,
            .emissive_factor = create_info.emissive_factor,
            .emissive_strength = create_info.emissive_strength,
            .metallic_factor = create_info.metallic_factor,
            .roughness_factor = create_info.roughness_factor,
            .normal_scale = create_info.normal_scale,
            .occlusion_strength = create_info.occlusion_strength,
            .base_colour_texture = create_info.base_colour_texture.index,
            .normal_texture = create_info.normal_texture.index,
            .metallic_roughness_texture = create_info.metallic_roughness_texture.index,
            .occlusion_texture = create_info.occlusion_texture.index,
            .emissive_texture = create_info.emissive_texture.index,
            .sampler_index = create_info.sampler.index,
            .alpha_mode = create_info.alpha_mode,
            .alpha_cutoff = create_info.alpha_cutoff,
            .wind_strength = create_info.wind_strength,
            .max_shadow_cascade = create_info.max_shadow_cascade,
    };
}
