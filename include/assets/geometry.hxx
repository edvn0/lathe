#pragma once

#include <volk.h>

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

struct MeshletSlice {
    GeometrySlice descriptors{};
    GeometrySlice data{};
    std::uint32_t meshlet_count = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return meshlet_count != 0 && descriptors.valid() && data.valid();
    }
};

struct MeshGeometry {
    VertexSlice vertices{};
    IndexSlice indices{};
    MeshletSlice meshlets{};

    GeometrySlice skin{};

    [[nodiscard]]
    auto skinned() const noexcept -> bool {
        return skin.valid();
    }
};
