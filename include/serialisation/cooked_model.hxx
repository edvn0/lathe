#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "assets/load_model.hxx"
#include "gpu/sampler.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/lbf_error.hxx"

// MODL chunk payload: a finalized ModelCpuData -- node hierarchy, materials, lights, and per primitive the packed
// vertices, every LOD's index buffer and every LOD's meshlets. Loading one skips glTF parsing, tangent generation,
// LOD simplification, vertex packing and meshlet building; what's left is decode + GPU upload.
//
// Vertex and index buffers go through meshoptimizer's codecs (lossless, decode at several GB/s) before the
// container's zstd, which together typically beat zstd alone by 2-3x on geometry.
//
// Textures aren't embedded: each image is a reference to a TEXR chunk by AssetId, so models sharing a texture
// share the chunk.
//
// Versioning: the encoder always writes cooked_model_version; the decoder accepts
// [cooked_model_oldest_readable_version, cooked_model_version]. The payload embeds GPU-facing layouts
// (CompressedModelVertex, GpuMeshlet, meshlet limits), so changing any of those must bump the version; the
// static_asserts in cooked_model.cxx trip when one changes. When a bump makes old payloads unusable (e.g. a new
// vertex format), raise oldest_readable too, and re-saving re-cooks those assets from source.
inline constexpr std::uint16_t cooked_model_version = 1;
inline constexpr std::uint16_t cooked_model_oldest_readable_version = 1;

struct CookedImageRef {
    AssetId texture{};
    ModelTextureSlot slot = ModelTextureSlot::base_colour;
    std::string debug_name;
};

struct CookedModel {
    // Primitives have compressed_vertices, indices, reduced_indices, meshlets and bounds, but no `vertices`.
    // Material samplers and image sources are left for the caller to resolve from `images` and
    // `material_samplers`.
    ModelCpuData cpu_data;
    std::vector<CookedImageRef> images; // parallel to cpu_data.image_sources
    std::vector<DefaultSampler> material_samplers; // parallel to cpu_data.materials
};

// Every primitive must be finalized (see prepare_primitive_gpu_data()). `images` is parallel to
// cpu_data.image_sources and `material_samplers` to cpu_data.materials.
[[nodiscard]]
auto encode_cooked_model(ModelCpuData const &cpu_data, std::span<CookedImageRef const> images,
                         std::span<DefaultSampler const> material_samplers)
        -> std::expected<std::vector<std::byte>, LbfError>;

[[nodiscard]]
auto decode_cooked_model(std::span<std::byte const> payload, std::uint16_t version = cooked_model_version)
        -> std::expected<CookedModel, LbfError>;
