#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "assets/load_model.hxx"
#include "core/fly_string.hxx"
#include "gpu/sampler.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/lbf_error.hxx"

inline constexpr std::uint16_t cooked_model_version = 3;
inline constexpr std::uint16_t cooked_model_oldest_readable_version = 1;

struct CookedImageRef {
    AssetId texture{};
    ModelTextureSlot slot = ModelTextureSlot::base_colour;
    FlyString debug_name;
};

struct CookedModel {
    ModelCpuData cpu_data;
    std::vector<CookedImageRef> images;
    std::vector<DefaultSampler> material_samplers;
};

[[nodiscard]]
auto encode_cooked_model(ModelCpuData const &cpu_data, std::span<CookedImageRef const> images,
                         std::span<DefaultSampler const> material_samplers)
        -> std::expected<std::vector<std::byte>, LbfError>;

[[nodiscard]]
auto decode_cooked_model(std::span<std::byte const> payload, std::uint16_t version = cooked_model_version)
        -> std::expected<CookedModel, LbfError>;
