#pragma once

#include <array>
#include <span>

#include <glm/vec3.hpp>

#include "core/config.hxx"
#include "assets/geometry.hxx"
#include "assets/material.hxx"

// Description of one submesh (one LOD chain + material) passed to
// Renderer::create_mesh / IMeshSink::create_mesh.
struct SubmeshCreateInfo {
    // One MeshGeometry per LOD level (lods[0] is full detail, always
    // required). Levels without a distinct simplification may alias an
    // earlier level's geometry.
    std::array<MeshGeometry, lod_count> lods{};
    MaterialHandle material{};

    // Local-space (untransformed) AABB over this submesh's own vertices,
    // used as the culling volume for GPU frustum culling. Shared across all
    // LODs.
    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

struct MeshCreateInfo {
    std::span<const SubmeshCreateInfo> submeshes;
};
