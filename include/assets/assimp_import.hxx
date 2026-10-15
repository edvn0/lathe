#pragma once

#include <expected>
#include <memory>

#include "assets/load_model.hxx"

// Reads a non-glTF model (.fbx, .obj, .dae, .ply, .stl, .usd/.usda/.usdc/.usdz) into the same unfinalized CPU form
// load_model_cpu_unfinalized returns for glTF, so tangents, LODs, meshlets and the GPU upload are shared.
//
// Static geometry, materials, textures and the node hierarchy only: skeletons, animation and lights are not imported.
[[nodiscard]]
auto load_model_cpu_assimp(AssetPath const &path, SamplerStorage &sampler_storage,
                           std::shared_ptr<ModelLoadProfile> profile = nullptr)
        -> std::expected<ModelCpuData, ModelLoadError>;
