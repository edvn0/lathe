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

struct GltfSkinImport {
    std::shared_ptr<ModelAnimationData const> data;
    std::size_t skin_index{0};
    std::vector<std::uint32_t> joint_remap;
    std::vector<bool> mesh_is_skinned;
};

[[nodiscard]]
auto import_gltf_skin(fastgltf::Asset const &asset, bool mirror_z)
        -> std::expected<std::optional<GltfSkinImport>, ModelLoadError>;

[[nodiscard]]
auto read_skin_vertices(fastgltf::Asset const &asset, fastgltf::Primitive const &primitive,
                        std::span<std::uint32_t const> joint_remap, std::size_t vertex_count)
        -> std::expected<std::vector<SkinVertex>, ModelLoadError>;

[[nodiscard]] auto quantise_skin_weights(std::array<float, 4> weights) -> std::array<std::uint16_t, 4>;
