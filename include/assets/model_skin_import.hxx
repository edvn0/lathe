#pragma once

#include <fastgltf/core.hpp>

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/model_skin.hxx"

// Internal to the glTF loader (and tests): skin + animation extraction from a parsed asset.
struct GltfSkinImport {
    std::shared_ptr<ModelAnimationData const> data;
    std::size_t skin_index{0};
    // glTF skin joint index (what JOINTS_0 holds) -> Skeleton joint index.
    std::vector<std::uint32_t> joint_remap;
    // Per glTF mesh index: true when a node instancing it uses `skin_index`.
    std::vector<bool> mesh_is_skinned;
};

// Imports the first skin of `asset`. nullopt when the asset has no skin (the common case, behaviour unchanged).
// `mirror_z`: apply the same Z mirror the loader applies to geometry (right-handed glTF -> left-handed renderer).
[[nodiscard]]
auto import_gltf_skin(fastgltf::Asset const &asset, bool mirror_z)
        -> std::expected<std::optional<GltfSkinImport>, ModelLoadError>;

// Reads JOINTS_0/WEIGHTS_0 of `primitive` (empty vector when it has none). Weights are normalised to sum 65535
// (unorm16), at most 4 influences, joints remapped through `joint_remap`.
[[nodiscard]]
auto read_skin_vertices(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive,
                        std::span<std::uint32_t const> joint_remap, std::size_t vertex_count)
        -> std::expected<std::vector<SkinVertex>, ModelLoadError>;

// Quantises 4 float weights (any non-negative scale) to unorm16 summing exactly to 65535.
[[nodiscard]] auto quantise_skin_weights(std::array<float, 4> weights) -> std::array<std::uint16_t, 4>;
