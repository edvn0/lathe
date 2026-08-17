#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>

struct GeometrySlice {
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkDeviceSize reserved_size = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return size != 0;
    }
};

struct VertexSlice {
    GeometrySlice bytes{};
    std::uint32_t vertex_count = 0;
    std::uint32_t vertex_stride = 0;
    VkDeviceSize alignment = 1;
};

struct IndexSlice {
    GeometrySlice bytes{};
    std::uint32_t index_count = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT32;
};

struct MeshGeometry {
    VertexSlice vertices{};
    IndexSlice indices{};
};