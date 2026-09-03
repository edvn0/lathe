#include "gpu/buffer.hxx"

#include "gpu/context.hxx"
#include "core/logger.hxx"
#include "gpu/vk_object_name.hxx"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace {

    [[nodiscard]]
    auto requires_mapping(BufferMemory memory) noexcept -> bool {

        switch (memory) {
            case BufferMemory::device:
                return false;

            case BufferMemory::upload:
            case BufferMemory::readback:
                return true;
        }

        return false;
    }

    [[nodiscard]]
    auto make_allocation_create_info(BufferMemory memory) -> VmaAllocationCreateInfo {

        VmaAllocationCreateInfo create_info{
                .flags = 0,
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = 0,
                .preferredFlags = 0,
                .memoryTypeBits = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr,
                .priority = 0.0F,
                .minAlignment = 0,
        };

        switch (memory) {
            case BufferMemory::device:
                //
                // Proper device-local memory.
                //
                // There is deliberately no HOST_VISIBLE requirement and no
                // persistent mapping request here.
                //
                // A buffer can still have a valid VkDeviceAddress while residing
                // entirely in GPU-local, non-host-visible memory.
                //
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

                break;

            case BufferMemory::upload:
                //
                // CPU -> GPU.
                //
                // Persistent mapping avoids map/unmap overhead for transient and
                // streaming uploads.
                //
                // SEQUENTIAL_WRITE tells VMA that CPU access primarily consists of
                // sequential writes, allowing it to pick an appropriate memory
                // type such as write-combined memory where applicable.
                //
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

                create_info.flags =
                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

                create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

                break;

            case BufferMemory::readback:
                //
                // GPU -> CPU.
                //
                // RANDOM is the appropriate VMA host-access mode for memory that
                // the CPU genuinely reads.
                //
                // HOST_CACHED is particularly useful for large readbacks such as
                // screenshots.
                //
                // HOST_COHERENT is not required. invalidate() handles
                // non-coherent allocations correctly.
                //
                create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

                create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

                create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

                create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

                break;
        }

        return create_info;
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

        return std::unexpected{
                make_buffer_error("Buffer write range is invalid or buffer is not mapped", VK_ERROR_MEMORY_MAP_FAILED)};
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

        return std::unexpected{
                make_buffer_error("Buffer read range is invalid or buffer is not mapped", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    if (auto invalidated = invalidate(offset, read_size); !invalidated) {

        return std::unexpected{std::move(invalidated.error())};
    }

    std::memcpy(destination.data(), mapped_data() + offset, destination.size_bytes());

    return {};
}

auto Buffer::flush(VkDeviceSize offset, VkDeviceSize flush_size) -> std::expected<void, DeviceError> {

    if (!validate_range(offset, flush_size)) {

        return std::unexpected{
                make_buffer_error("Buffer flush range is invalid or buffer is not mapped", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    auto const result = vmaFlushAllocation(allocator, allocation, offset, flush_size);

    if (result != VK_SUCCESS) {
        return std::unexpected{make_buffer_error("vmaFlushAllocation failed", result)};
    }

    return {};
}

auto Buffer::invalidate(VkDeviceSize offset, VkDeviceSize invalidate_size) -> std::expected<void, DeviceError> {

    if (!validate_range(offset, invalidate_size)) {

        return std::unexpected{make_buffer_error("Buffer invalidate range is invalid or buffer is not mapped",
                                                 VK_ERROR_MEMORY_MAP_FAILED)};
    }

    auto const result = vmaInvalidateAllocation(allocator, allocation, offset, invalidate_size);

    if (result != VK_SUCCESS) {
        return std::unexpected{make_buffer_error("vmaInvalidateAllocation failed", result)};
    }

    return {};
}

auto Buffer::zero() -> std::expected<void, DeviceError> {

    if (!valid() || !mapped()) {
        return std::unexpected{
                make_buffer_error("Cannot zero an invalid or unmapped buffer", VK_ERROR_MEMORY_MAP_FAILED)};
    }

    std::memset(mapped_data(), 0, static_cast<std::size_t>(buffer_size));

    return flush(0, buffer_size);
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

    VkBufferCreateInfo const vk_create_info{
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

    //
    // upload/readback explicitly requested persistent mappings.
    //
    // device buffers deliberately do not.
    //
    if (requires_mapping(create_info.memory) && result.allocation_info.pMappedData == nullptr) {
        result.destroy();
        return std::unexpected{make_buffer_error("VMA returned an unmapped host-visible buffer allocation",
                                                 VK_ERROR_MEMORY_MAP_FAILED)};
    }

    assert(!create_info.debug_name.empty() && "Creating a buffer requires a name");

    if (!create_info.debug_name.empty()) {
        auto const debug_name = std::string{create_info.debug_name};
        vmaSetAllocationName(ctx.allocator, result.allocation, debug_name.c_str());
        static_cast<void>(vk::set_object_name(ctx.device, VK_OBJECT_TYPE_BUFFER, vk::object_handle(result.buffer),
                                              debug_name.c_str()));
    }

    //
    // Buffer Device Address is independent of CPU visibility/mapping.
    //
    // A fully device-local allocation can and normally should still expose a
    // VkDeviceAddress when SHADER_DEVICE_ADDRESS usage was requested.
    //
    if ((create_info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {

        VkBufferDeviceAddressInfo const address_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .pNext = nullptr,
                .buffer = result.buffer,
        };

        result.device_address = vkGetBufferDeviceAddress(ctx.device, &address_info);

        if (result.device_address == 0) {
            error("We requested BDA for buffer '{}' but "
                  "vkGetBufferDeviceAddress returned zero",
                  create_info.debug_name);

            result.destroy();

            return std::unexpected{
                    make_buffer_error("vkGetBufferDeviceAddress returned zero", VK_ERROR_INITIALIZATION_FAILED)};
        }
    }

    return std::expected<Buffer, DeviceError>{std::in_place, std::move(result)};
}
