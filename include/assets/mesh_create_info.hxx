#pragma once

#include <array>
#include <span>

#include <glm/vec3.hpp>

#include "core/config.hxx"
#include "assets/geometry.hxx"
#include "assets/material.hxx"

// One submesh (an LOD chain plus material) for Renderer::create_mesh.
struct SubmeshCreateInfo {
    // lods[0] is full detail and required. Levels without their own simplification may alias an earlier one.
    std::array<MeshGeometry, lod_count> lods{};
    MaterialHandle material{};

    // Local-space AABB of the submesh, shared by all LODs; used for GPU culling.
    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

struct MeshCreateInfo {
    std::span<const SubmeshCreateInfo> submeshes;
};
