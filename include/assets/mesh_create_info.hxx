#pragma once

#include <array>
#include <span>

#include <glm/vec3.hpp>

#include "assets/geometry.hxx"
#include "assets/material.hxx"
#include "core/config.hxx"

struct SubmeshCreateInfo {
    std::array<MeshGeometry, lod_count> lods{};
    MaterialHandle material{};

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

struct MeshCreateInfo {
    std::span<const SubmeshCreateInfo> submeshes;
};
