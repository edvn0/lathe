#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include <glm/vec3.hpp>

#include "assets/load_model.hxx"

struct PrimitiveMeshData {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;

    std::array<std::optional<std::vector<std::uint32_t>>, lod_count - 1> lod_indices{};
};

[[nodiscard]] auto to_model_cpu_data(PrimitiveMeshData mesh) -> ModelCpuData;

[[nodiscard]] auto make_cube_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_sphere_mesh(std::uint32_t rings = 16, std::uint32_t segments = 32)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError>;

inline constexpr std::uint32_t grass_clump_card_lod = 2;

[[nodiscard]] auto make_capsule_mesh(std::uint32_t segments = 16, std::uint32_t rings = 8)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

[[nodiscard]] auto make_ribbon_mesh(std::span<glm::vec3 const> grid, std::uint32_t columns, float uv_scale = 0.25F)
        -> std::expected<PrimitiveMeshData, ModelLoadError>;

// One axis of a water grid: `cells` cells across [-0.5, 0.5], packed densest at `focus` (0 = the -0.5 end, 0.5 = the
// middle, 1 = the +0.5 end) and widening exponentially away from it by `curve` (0 = evenly spaced).
struct GridAxis {
    std::uint32_t cells = 64;
    float curve = 0.0F;
    float focus = 0.5F;
};

// A flat, upward-facing grid of `x.cells` by `z.cells` quads in the unit square of the XZ plane (y = 0), for surfaces
// the vertex stage displaces (water). Every LOD keeps every triangle: a simplifier would flatten it to two.
[[nodiscard]] auto make_water_grid_mesh(GridAxis x, GridAxis z) -> std::expected<PrimitiveMeshData, ModelLoadError>;
