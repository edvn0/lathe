#pragma once

#include <volk.h>

#include "buffer.hxx"
#include "device_error.hxx"
#include "error_context.hxx"
#include "forward.hxx"
#include "geometry.hxx"

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

enum class GeometryArenaErrorType : std::uint8_t {
    invalid_argument,
    unsupported_index_type,
    out_of_memory,
    size_overflow,
    device_error,
};

struct GeometryArenaError {
    GeometryArenaErrorType type = GeometryArenaErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<GeometryArenaErrorType> : std::formatter<std::string_view> {
    constexpr auto format(GeometryArenaErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case GeometryArenaErrorType::invalid_argument:
                    return "invalid_argument";
                case GeometryArenaErrorType::unsupported_index_type:
                    return "unsupported_index_type";
                case GeometryArenaErrorType::out_of_memory:
                    return "out_of_memory";
                case GeometryArenaErrorType::size_overflow:
                    return "size_overflow";
                case GeometryArenaErrorType::device_error:
                    return "device_error";
            }

            return "unknown_geometry_arena_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};



struct GeometryArenaCreateInfo {
    VkDeviceSize capacity = 0;
    std::string_view debug_name = "geometry_arena";
};

struct GeometryArena {
    GeometryArena() = default;

    GeometryArena(GeometryArena const &) = delete;
    auto operator=(GeometryArena const &) -> GeometryArena & = delete;

    GeometryArena(GeometryArena &&) noexcept = default;
    auto operator=(GeometryArena &&) noexcept -> GeometryArena & = default;

    auto destroy(VulkanContext &) -> void;

    [[nodiscard]]
    static auto create(VulkanContext &ctx, GeometryArenaCreateInfo const &create_info)
            -> std::expected<GeometryArena, GeometryArenaError>;

    [[nodiscard]]
    auto allocate_vertices(std::span<const std::byte> data, std::uint32_t vertex_stride, VkDeviceSize alignment)
            -> std::expected<VertexSlice, GeometryArenaError>;

    template<typename Vertex>
        requires std::is_trivially_copyable_v<Vertex>
    [[nodiscard]]
    auto allocate_vertices(std::span<const Vertex> vertices) -> std::expected<VertexSlice, GeometryArenaError> {
        return allocate_vertices(std::as_bytes(vertices), static_cast<std::uint32_t>(sizeof(Vertex)), alignof(Vertex));
    }

    [[nodiscard]]
    auto allocate_indices(std::span<const std::byte> data, VkIndexType index_type)
            -> std::expected<IndexSlice, GeometryArenaError>;

    template<typename Index>
        requires(std::same_as<Index, std::uint16_t> || std::same_as<Index, std::uint32_t>)
    [[nodiscard]]
    auto allocate_indices(std::span<const Index> indices) -> std::expected<IndexSlice, GeometryArenaError> {
        constexpr auto index_type = [] {
            if constexpr (std::same_as<Index, std::uint16_t>) {
                return VK_INDEX_TYPE_UINT16;
            } else {
                return VK_INDEX_TYPE_UINT32;
            }
        }();

        return allocate_indices(std::as_bytes(indices), index_type);
    }

    template<typename Vertex, typename Index>
        requires(std::is_trivially_copyable_v<Vertex> &&
                 (std::same_as<Index, std::uint16_t> || std::same_as<Index, std::uint32_t>) )
    [[nodiscard]]
    auto allocate_mesh(std::span<const Vertex> vertices, std::span<const Index> indices)
            -> std::expected<MeshGeometry, GeometryArenaError> {
        auto const checkpoint = next_offset;

        auto vertex_slice = allocate_vertices(vertices);
        if (!vertex_slice) {
            return std::unexpected(vertex_slice.error());
        }

        auto index_slice = allocate_indices(indices);
        if (!index_slice) {
            next_offset = checkpoint;
            return std::unexpected(index_slice.error());
        }

        return MeshGeometry{
                .vertices = *vertex_slice,
                .indices = *index_slice,
        };
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
        return next_offset;
    }

    [[nodiscard]]
    auto remaining_size() const noexcept -> VkDeviceSize {
        return capacity - next_offset;
    }

    Buffer buffer{};

private:
    [[nodiscard]]
    auto allocate_bytes(VkDeviceSize size, VkDeviceSize alignment) -> std::expected<GeometrySlice, GeometryArenaError>;

    [[nodiscard]]
    auto write(GeometrySlice const &slice, std::span<const std::byte> data) -> std::expected<void, GeometryArenaError>;

    VkDeviceSize capacity = 0;
    VkDeviceSize next_offset = 0;
};
