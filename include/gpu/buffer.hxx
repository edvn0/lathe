#pragma once

#include <volk.h>

#include "core/allocator.hxx"
#include "gpu/device_error.hxx"
#include "core/forward.hxx"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <type_traits>

enum class BufferMemory : std::uint8_t {
    // GPU-local memory.
    //
    // Not required to be HOST_VISIBLE and normally not CPU mapped.
    // Populate using transfers from an upload buffer.
    device,

    // CPU -> GPU memory.
    //
    // Persistently mapped and optimized for sequential CPU writes.
    upload,

    // GPU -> CPU memory.
    //
    // Persistently mapped and preferably HOST_CACHED for efficient CPU reads.
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
    auto valid() const noexcept -> bool {
        return buffer != VK_NULL_HANDLE;
    }

    [[nodiscard]]
    auto mapped() const noexcept -> bool {
        return allocation_info.pMappedData != nullptr;
    }

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
    auto size() const noexcept -> VkDeviceSize {
        return buffer_size;
    }

    [[nodiscard]]
    auto write(VkDeviceSize offset, std::span<const std::byte> data) -> std::expected<void, DeviceError>;

    template<typename T, std::size_t Extent = std::dynamic_extent>
        requires(!std::same_as<std::remove_cv_t<T>, std::byte>)
    [[nodiscard]]
    auto write(VkDeviceSize offset, std::span<const T, Extent> data) -> std::expected<void, DeviceError> {

        auto const bytes = std::as_bytes(data);

        return write(offset, std::span<const std::byte>{
                                     bytes.data(),
                                     bytes.size(),
                             });
    }

    template<typename T, std::size_t Extent = std::dynamic_extent>
        requires(!std::same_as<std::remove_cv_t<T>, std::byte>)
    [[nodiscard]]
    auto write(VkDeviceSize offset, std::span<T, Extent> data) -> std::expected<void, DeviceError> {

        return write(offset, std::span<const T, Extent>{
                                     data.data(),
                                     data.size(),
                             });
    }

    [[nodiscard]]
    auto read(VkDeviceSize offset, std::span<std::byte> destination) -> std::expected<void, DeviceError>;

    template<typename T, std::size_t Extent = std::dynamic_extent>
        requires(!std::same_as<std::remove_cv_t<T>, std::byte>)
    [[nodiscard]]
    auto read(VkDeviceSize offset, std::span<T, Extent> destination) -> std::expected<void, DeviceError> {

        auto const bytes = std::as_writable_bytes(destination);

        return read(offset, std::span<std::byte>{
                                    bytes.data(),
                                    bytes.size(),
                            });
    }

    [[nodiscard]]
    auto flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) -> std::expected<void, DeviceError>;

    [[nodiscard]]
    auto invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) -> std::expected<void, DeviceError>;

    [[nodiscard]]
    auto zero() -> std::expected<void, DeviceError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    static auto create(VulkanContext &ctx, BufferCreateInfo const &create_info) -> std::expected<Buffer, DeviceError>;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocation_info{};
    VkDeviceAddress device_address = 0;

private:
    [[nodiscard]]
    auto validate_range(VkDeviceSize offset, VkDeviceSize range_size) const noexcept -> bool;

    VmaAllocator allocator = VK_NULL_HANDLE;
    VkDeviceSize buffer_size = 0;
};
