#pragma once

#include <volk.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/vec3.hpp>

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

// CPU copy of a meshlet's bounds, kept for debug visualisation of the normal cones.
struct MeshletCone {
    glm::vec3 centre{0.0F};
    float radius = 0.0F;
    glm::vec3 axis{0.0F, 0.0F, 1.0F};
    float cutoff = 1.0F;
};

struct MeshletSlice {
    std::shared_ptr<std::vector<MeshletCone> const> cones;

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
