#include "assets/geometry_arena.hxx"

#include "gpu/context.hxx"

#include <limits>
#include <string>
#include <utility>

namespace {

    [[nodiscard]]
    constexpr auto index_element_size(VkIndexType index_type) noexcept -> VkDeviceSize {
        switch (index_type) {
            case VK_INDEX_TYPE_UINT16:
                return sizeof(std::uint16_t);
            case VK_INDEX_TYPE_UINT32:
                return sizeof(std::uint32_t);
            default:
                return 0;
        }
    }

    [[nodiscard]]
    auto from_device_error(DeviceError error) -> GeometryArenaError {
        return GeometryArenaError{
                .type = GeometryArenaErrorType::device_error,
                .cause =
                        ErrorCause{
                                Boxed<DeviceError>{
                                        std::move(error),
                                },
                        },
        };
    }

} // namespace

template<GeometryAllocatorPolicy Allocator>
auto GeometryArenaT<Allocator>::destroy(VulkanContext &) -> void {
    upload_buffer.destroy();
    buffer.destroy();

    allocator_.reset(0);
}

template<GeometryAllocatorPolicy Allocator>
auto GeometryArenaT<Allocator>::create(VulkanContext &ctx, GeometryArenaCreateInfo const &create_info)
        -> std::expected<GeometryArenaT<Allocator>, GeometryArenaError> {

    if (create_info.capacity == 0) {
        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        }};
    }

    auto buffer = Buffer::create(
            ctx, BufferCreateInfo{
                         .size = create_info.capacity,
                         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         .memory = BufferMemory::device,
                         .debug_name = create_info.debug_name,
                 });

    if (!buffer) {
        return std::unexpected{from_device_error(std::move(buffer.error()))};
    }

    auto const upload_name = std::format("{}.upload", create_info.debug_name);
    auto upload_buffer = Buffer::create(ctx, BufferCreateInfo{
                                                     .size = create_info.capacity,
                                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     .memory = BufferMemory::upload,
                                                     .debug_name = upload_name,
                                             });

    if (!upload_buffer) {
        buffer->destroy();
        return std::unexpected{from_device_error(std::move(upload_buffer.error()))};
    }

    GeometryArenaT<Allocator> result{};
    result.buffer = std::move(*buffer);
    result.upload_buffer = std::move(*upload_buffer);
    result.allocator_.reset(create_info.capacity);
    return result;
}

template<GeometryAllocatorPolicy Allocator>
auto GeometryArenaT<Allocator>::write(VkCommandBuffer command_buffer, GeometrySlice const &slice,
                                      std::span<const std::byte> data) -> std::expected<void, GeometryArenaError> {

    if (command_buffer == VK_NULL_HANDLE || data.empty() || data.size_bytes() != slice.size ||
        slice.offset > allocator_.capacity() || slice.size > allocator_.capacity() - slice.offset) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        }};
    }

    if (auto written = upload_buffer.write(slice.offset, data); !written) {

        return std::unexpected{from_device_error(std::move(written.error()))};
    }

    VkBufferCopy2 const region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
            .pNext = nullptr,
            .srcOffset = slice.offset,
            .dstOffset = slice.offset,
            .size = slice.size,
    };

    VkCopyBufferInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
            .pNext = nullptr,
            .srcBuffer = upload_buffer.buffer,
            .dstBuffer = buffer.buffer,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyBuffer2(command_buffer, &copy_info);

    VkBufferMemoryBarrier2 const barrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer.buffer,
            .offset = slice.offset,
            .size = slice.size,
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

    return {};
}

template<GeometryAllocatorPolicy Allocator>
auto GeometryArenaT<Allocator>::allocate_vertices(VkCommandBuffer command_buffer, std::span<const std::byte> data,
                                                  std::uint32_t vertex_stride, VkDeviceSize alignment)
        -> std::expected<VertexSlice, GeometryArenaError> {

    if (command_buffer == VK_NULL_HANDLE || data.empty() || vertex_stride == 0 ||
        data.size_bytes() % vertex_stride != 0) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        }};
    }

    auto const vertex_count = data.size_bytes() / vertex_stride;

    if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::size_overflow,
                .cause = std::nullopt,
        }};
    }

    auto const checkpoint = allocator_.checkpoint();

    auto allocation = allocator_.allocate(static_cast<VkDeviceSize>(data.size_bytes()), alignment);

    if (!allocation) {
        return std::unexpected(allocation.error());
    }

    auto written = write(command_buffer, *allocation, data);

    if (!written) {
        allocator_.rollback(checkpoint);

        return std::unexpected(written.error());
    }

    return VertexSlice{
            .bytes = *allocation,
            .vertex_count = static_cast<std::uint32_t>(vertex_count),
            .vertex_stride = vertex_stride,
            .alignment = alignment,
    };
}

template<GeometryAllocatorPolicy Allocator>
auto GeometryArenaT<Allocator>::allocate_indices(VkCommandBuffer command_buffer, std::span<const std::byte> data,
                                                 VkIndexType index_type) -> std::expected<IndexSlice, GeometryArenaError> {

    auto const element_size = index_element_size(index_type);

    if (element_size == 0 || command_buffer == VK_NULL_HANDLE || data.empty() ||
        data.size_bytes() % element_size != 0) {

        return std::unexpected{GeometryArenaError{
                .type = element_size == 0 ? GeometryArenaErrorType::unsupported_index_type
                                          : GeometryArenaErrorType::invalid_argument,
                .cause = std::nullopt,
        }};
    }

    auto const index_count = data.size_bytes() / element_size;

    if (index_count > std::numeric_limits<std::uint32_t>::max()) {

        return std::unexpected{GeometryArenaError{
                .type = GeometryArenaErrorType::size_overflow,
                .cause = std::nullopt,
        }};
    }

    auto const checkpoint = allocator_.checkpoint();

    auto allocation = allocator_.allocate(static_cast<VkDeviceSize>(data.size_bytes()), element_size);

    if (!allocation) {
        return std::unexpected(allocation.error());
    }

    auto written = write(command_buffer, *allocation, data);

    if (!written) {
        allocator_.rollback(checkpoint);

        return std::unexpected(written.error());
    }

    return IndexSlice{
            .bytes = *allocation,
            .index_count = static_cast<std::uint32_t>(index_count),
            .index_type = index_type,
    };
}

template struct GeometryArenaT<BumpAllocator>;
template struct GeometryArenaT<FreeListAllocator>;
