#pragma once

#include <volk.h>

#include "gpu/buffer.hxx"
#include "gpu/device_error.hxx"
#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "assets/geometry.hxx"
#include "assets/geometry_allocator.hxx"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string_view>
#include <type_traits>

struct GeometryArenaCreateInfo {
    VkDeviceSize capacity = 0;
    std::string_view debug_name = "geometry_arena";
};

// Backed by an Allocator satisfying GeometryAllocatorPolicy (see
// geometry_allocator.hxx), which owns every offset decision; this class
// owns the GPU buffers and the upload-then-copy-then-barrier mechanics.
// `using GeometryArena = GeometryArenaT<BumpAllocator>` below is the type
// every other file names -- swapping the allocator is a one-line change
// here with zero call-site impact.
template<GeometryAllocatorPolicy Allocator>
struct GeometryArenaT {
    GeometryArenaT() = default;

    GeometryArenaT(GeometryArenaT const &) = delete;
    auto operator=(GeometryArenaT const &) -> GeometryArenaT & = delete;

    GeometryArenaT(GeometryArenaT &&) noexcept = default;
    auto operator=(GeometryArenaT &&) noexcept -> GeometryArenaT & = default;

    auto destroy(VulkanContext &) -> void;

    [[nodiscard]]
    static auto create(VulkanContext &ctx, GeometryArenaCreateInfo const &create_info)
            -> std::expected<GeometryArenaT, GeometryArenaError>;

    [[nodiscard]]
    auto allocate_vertices(VkCommandBuffer command_buffer, std::span<const std::byte> data, std::uint32_t vertex_stride,
                           VkDeviceSize alignment) -> std::expected<VertexSlice, GeometryArenaError>;

    template<typename Vertex>
        requires std::is_trivially_copyable_v<Vertex>
    [[nodiscard]]
    auto allocate_vertices(VkCommandBuffer command_buffer, std::span<const Vertex> vertices)
            -> std::expected<VertexSlice, GeometryArenaError> {

        return allocate_vertices(command_buffer, std::as_bytes(vertices), static_cast<std::uint32_t>(sizeof(Vertex)),
                                 alignof(Vertex));
    }

    [[nodiscard]]
    auto allocate_indices(VkCommandBuffer command_buffer, std::span<const std::byte> data, VkIndexType index_type)
            -> std::expected<IndexSlice, GeometryArenaError>;

    template<typename Index>
        requires(std::same_as<Index, std::uint16_t> || std::same_as<Index, std::uint32_t>)
    [[nodiscard]]
    auto allocate_indices(VkCommandBuffer command_buffer, std::span<const Index> indices)
            -> std::expected<IndexSlice, GeometryArenaError> {

        constexpr auto index_type = [] {
            if constexpr (std::same_as<Index, std::uint16_t>) {
                return VK_INDEX_TYPE_UINT16;
            } else {
                return VK_INDEX_TYPE_UINT32;
            }
        }();

        return allocate_indices(command_buffer, std::as_bytes(indices), index_type);
    }

    template<typename Vertex, typename Index>
        requires(std::is_trivially_copyable_v<Vertex> &&
                 (std::same_as<Index, std::uint16_t> || std::same_as<Index, std::uint32_t>) )
    [[nodiscard]]
    auto allocate_mesh(VkCommandBuffer command_buffer, std::span<const Vertex> vertices, std::span<const Index> indices)
            -> std::expected<MeshGeometry, GeometryArenaError> {

        auto const checkpoint = allocator_.checkpoint();

        auto vertex_slice = allocate_vertices(command_buffer, vertices);

        if (!vertex_slice) {
            return std::unexpected(vertex_slice.error());
        }

        auto index_slice = allocate_indices(command_buffer, indices);

        if (!index_slice) {
            allocator_.rollback(checkpoint);

            return std::unexpected(index_slice.error());
        }

        return MeshGeometry{
                .vertices = *vertex_slice,
                .indices = *index_slice,
        };
    }

    // Overwrites an already-allocated slice in place -- `data` must be
    // exactly `slice.size` bytes, and this never touches allocator state
    // (no allocate/deallocate). The caller owns the in-flight discipline:
    // the GPU must be done reading whatever was last submitted against this
    // range (frames_in_flight frames since its last use in a draw), and
    // this must be recorded before any draw in `command_buffer` that reads
    // it -- the barrier this records is write-after-write/read, not the
    // other direction. See TerrainSlotPool::tick_retirement() for the
    // deferred-release discipline this depends on.
    //
    // Safe against staging-buffer aliasing only because `upload_buffer` is
    // currently a full 1:1 mirror of the device buffer (see create()) --
    // every slice has a dedicated staging offset. If that ever changes to a
    // rotating ring (docs/engine_review_followups.md), a rewrite through a
    // reused ring slot needs its own in-flight deferral on the staging side
    // too.
    [[nodiscard]]
    auto rewrite_slice(VkCommandBuffer command_buffer, GeometrySlice const &slice, std::span<const std::byte> data)
            -> std::expected<void, GeometryArenaError> {

        return write(command_buffer, slice, data);
    }

    [[nodiscard]]
    auto device_address(GeometrySlice const &slice) const noexcept -> VkDeviceAddress {

        return buffer.device_address + slice.offset;
    }

    [[nodiscard]]
    auto vertex_address(VertexSlice const &slice) const noexcept -> VkDeviceAddress {

        return device_address(slice.bytes);
    }

    [[nodiscard]]
    auto used_size() const noexcept -> VkDeviceSize {

        return allocator_.used_size();
    }

    [[nodiscard]]
    auto remaining_size() const noexcept -> VkDeviceSize {

        return allocator_.capacity() - allocator_.used_size();
    }

    auto bindable_buffer() const -> VkBuffer { return buffer.buffer; }

private:
    [[nodiscard]]
    auto write(VkCommandBuffer command_buffer, GeometrySlice const &slice, std::span<const std::byte> data)
            -> std::expected<void, GeometryArenaError>;

    Buffer upload_buffer{};
    Buffer buffer{};

    Allocator allocator_{};
};

// Default backing allocator: bump, never frees -- identical behavior to
// this class before the allocator policy was factored out. See
// geometry_allocator.hxx for FreeListAllocator, the streaming-capable
// alternative that can be substituted here with no other call-site change.
using GeometryArena = GeometryArenaT<BumpAllocator>;
