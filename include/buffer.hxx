#pragma once

#include <volk.h>

#include "allocator.hxx"
#include "device_error.hxx"
#include "forward.hxx"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

enum class BufferMemory : std::uint8_t {
    device,
    upload,
    readback,
};

struct BufferCreateInfo {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    BufferMemory memory = BufferMemory::device;
    VkBufferCreateFlags flags = 0;
    std::string_view debug_name{};
};

struct Buffer {
    Buffer() = default;
    ~Buffer();

    Buffer(Buffer const &) = delete;
    auto operator=(Buffer const &) -> Buffer & = delete;

    Buffer(Buffer &&other) noexcept;
    auto operator=(Buffer &&other) noexcept -> Buffer &;

    [[nodiscard]]
    auto mapped_data() noexcept -> std::byte * {
        return static_cast<std::byte *>(allocation_info.pMappedData);
    }

    [[nodiscard]]
    auto mapped_data() const noexcept -> std::byte const * {
        return static_cast<std::byte const *>(allocation_info.pMappedData);
    }

    template<typename T>
    [[nodiscard]]
    auto mapped_data_as() noexcept -> T * {
        return static_cast<T *>(allocation_info.pMappedData);
    }

    template<typename T>
    [[nodiscard]]
    auto mapped_data_as() const noexcept -> T const * {
        return static_cast<T const *>(allocation_info.pMappedData);
    }

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return buffer != VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto mapped() const noexcept -> bool {
        return allocation_info.pMappedData != nullptr;
    }

    [[nodiscard]]
    auto size() const noexcept -> VkDeviceSize {
        return buffer_size;
    }

    [[nodiscard]]
    auto write(VkDeviceSize offset, std::span<const std::byte> data) -> std::expected<void, DeviceError>;

    template<typename T>
    [[nodiscard]]
    auto write(VkDeviceSize offset, std::span<const T> data) -> std::expected<void, DeviceError> {
        return write(offset, std::as_bytes(data));
    }

    [[nodiscard]]
    auto read(VkDeviceSize offset, std::span<std::byte> destination) -> std::expected<void, DeviceError>;

    template<typename T>
    [[nodiscard]]
    auto read(VkDeviceSize offset, std::span<T> destination) -> std::expected<void, DeviceError> {
        return read(offset, std::as_writable_bytes(destination));
    }

    [[nodiscard]]
    auto flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) -> std::expected<void, DeviceError>;

    [[nodiscard]]
    auto invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) -> std::expected<void, DeviceError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    static auto create(VulkanContext &ctx, BufferCreateInfo const &create_info) -> std::expected<Buffer, DeviceError>;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocation_info{};
    VkDeviceAddress device_address = 0;

private:
    [[nodiscard]]
    auto validate_range(VkDeviceSize offset, VkDeviceSize size) const noexcept -> bool;

    VmaAllocator allocator = VK_NULL_HANDLE;
    VkDeviceSize buffer_size = 0;
};
