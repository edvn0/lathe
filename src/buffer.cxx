#include "buffer.hxx"

#include "context.hxx"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

namespace {

    [[nodiscard]]
    auto make_allocation_create_info(BufferMemory memory) -> VmaAllocationCreateInfo {
        VmaAllocationCreateInfo create_info{
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                .preferredFlags = 0,
                .memoryTypeBits = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr,
                .priority = 0.0F,
                .minAlignment = 0,
        };

        switch (memory) {
            case BufferMemory::device:
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

                create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

                break;

            case BufferMemory::upload:
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

                create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

                break;

            case BufferMemory::readback:
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

                create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

                break;
        }

        return create_info;
    }

    [[nodiscard]]
    auto set_buffer_debug_name(VulkanContext const &ctx, VkBuffer buffer, char const *debug_name) -> VkResult {
        if (vkSetDebugUtilsObjectNameEXT == nullptr) {
            return VK_SUCCESS;
        }

        auto const name_info = VkDebugUtilsObjectNameInfoEXT{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = VK_OBJECT_TYPE_BUFFER,
                .objectHandle = reinterpret_cast<std::uint64_t>(buffer),
                .pObjectName = debug_name,
        };

        return vkSetDebugUtilsObjectNameEXT(ctx.device, &name_info);
    }

    [[nodiscard]]
    auto make_buffer_error(std::string_view message, VkResult result) -> DeviceError {
        return DeviceError::buffer_creation(message, result);
    }

} // namespace

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer &&other) noexcept :
    buffer(std::exchange(other.buffer, VK_NULL_HANDLE)), allocation(std::exchange(other.allocation, VK_NULL_HANDLE)),
    allocation_info(std::exchange(other.allocation_info, VmaAllocationInfo{})),
    device_address(std::exchange(other.device_address, 0)), allocator(std::exchange(other.allocator, VK_NULL_HANDLE)),
    buffer_size(std::exchange(other.buffer_size, 0)) {}

auto Buffer::operator=(Buffer &&other) noexcept -> Buffer & {
    if (this == &other) {
        return *this;
    }

    destroy();

    buffer = std::exchange(other.buffer, VK_NULL_HANDLE);

    allocation = std::exchange(other.allocation, VK_NULL_HANDLE);

    allocation_info = std::exchange(other.allocation_info, VmaAllocationInfo{});

    device_address = std::exchange(other.device_address, 0);

    allocator = std::exchange(other.allocator, VK_NULL_HANDLE);

    buffer_size = std::exchange(other.buffer_size, 0);

    return *this;
}

auto Buffer::validate_range(VkDeviceSize offset, VkDeviceSize range_size) const noexcept -> bool {
    if (!valid() || !mapped()) {
        return false;
    }

    if (offset > buffer_size) {
        return false;
    }

    if (range_size == VK_WHOLE_SIZE) {
        return true;
    }

    return range_size <= buffer_size - offset;
}

auto Buffer::write(VkDeviceSize offset, std::span<const std::byte> data) -> std::expected<void, DeviceError> {
    if (data.empty()) {
        return {};
    }

    auto const write_size = static_cast<VkDeviceSize>(data.size_bytes());

    if (!validate_range(offset, write_size)) {
        return std::unexpected{make_buffer_error("Buffer write range is invalid", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    std::memcpy(mapped_data() + offset, data.data(), data.size_bytes());

    return flush(offset, write_size);
}

auto Buffer::read(VkDeviceSize offset, std::span<std::byte> destination) -> std::expected<void, DeviceError> {
    if (destination.empty()) {
        return {};
    }

    auto const read_size = static_cast<VkDeviceSize>(destination.size_bytes());

    if (!validate_range(offset, read_size)) {
        return std::unexpected{make_buffer_error("Buffer read range is invalid", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    auto invalidated = invalidate(offset, read_size);

    if (!invalidated) {
        return std::unexpected(std::move(invalidated.error()));
    }

    std::memcpy(destination.data(), mapped_data() + offset, destination.size_bytes());

    return {};
}

auto Buffer::flush(VkDeviceSize offset, VkDeviceSize flush_size) -> std::expected<void, DeviceError> {
    if (!validate_range(offset, flush_size)) {
        return std::unexpected{make_buffer_error("Buffer flush range is invalid", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    auto const result = vmaFlushAllocation(allocator, allocation, offset, flush_size);

    if (result != VK_SUCCESS) {
        return std::unexpected{make_buffer_error("vmaFlushAllocation failed", result)};
    }

    return {};
}

auto Buffer::invalidate(VkDeviceSize offset, VkDeviceSize invalidate_size) -> std::expected<void, DeviceError> {
    if (!validate_range(offset, invalidate_size)) {
        return std::unexpected{make_buffer_error("Buffer invalidate range is invalid", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    auto const result = vmaInvalidateAllocation(allocator, allocation, offset, invalidate_size);

    if (result != VK_SUCCESS) {
        return std::unexpected{make_buffer_error("vmaInvalidateAllocation failed", result)};
    }

    return {};
}

auto Buffer::destroy() noexcept -> void {
    if (allocator != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }

    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    allocation_info = {};
    device_address = 0;
    allocator = VK_NULL_HANDLE;
    buffer_size = 0;
}

auto Buffer::create(VulkanContext &ctx, BufferCreateInfo const &create_info) -> std::expected<Buffer, DeviceError> {
    if (create_info.size == 0) {
        return std::unexpected{
                make_buffer_error("Buffer size must be greater than zero", VK_ERROR_INITIALIZATION_FAILED)};
    }

    if (create_info.usage == 0) {
        return std::unexpected{make_buffer_error("Buffer usage must not be zero", VK_ERROR_INITIALIZATION_FAILED)};
    }

    auto const vk_create_info = VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = create_info.flags,
            .size = create_info.size,
            .usage = create_info.usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
    };

    auto allocation_create_info = make_allocation_create_info(create_info.memory);

    Buffer result{};
    result.allocator = ctx.allocator;
    result.buffer_size = create_info.size;

    auto const vk_result = vmaCreateBuffer(ctx.allocator, &vk_create_info, &allocation_create_info, &result.buffer,
                                           &result.allocation, &result.allocation_info);

    if (vk_result != VK_SUCCESS) {
        result.allocator = VK_NULL_HANDLE;
        result.buffer_size = 0;

        return std::unexpected{make_buffer_error("vmaCreateBuffer failed", vk_result)};
    }

    if (result.allocation_info.pMappedData == nullptr) {
        result.destroy();

        return std::unexpected{
                make_buffer_error("VMA returned an unmapped buffer allocation", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    if (!create_info.debug_name.empty()) {
        auto const debug_name = std::string{create_info.debug_name};

        vmaSetAllocationName(ctx.allocator, result.allocation, debug_name.c_str());

        static_cast<void>(set_buffer_debug_name(ctx, result.buffer, debug_name.c_str()));
    }

    if (create_info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        auto const address_info = VkBufferDeviceAddressInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = result.buffer,
        };

        result.device_address = vkGetBufferDeviceAddress(ctx.device, &address_info);

        if (result.device_address == 0) {
            result.destroy();

            return std::unexpected{
                    make_buffer_error("vkGetBufferDeviceAddress returned zero", VK_ERROR_INITIALIZATION_FAILED)};
        }
    }

    return std::expected<Buffer, DeviceError>{std::in_place, std::move(result)};
}
