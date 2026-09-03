#pragma once

#include <volk.h>

#include <cstdint>
#include <string>
#include <vector>

// How a texture's source pixels should be interpreted, which determines both
// the Basis Universal encode parameters and the block-compressed format it
// ultimately transcodes to.
enum class TextureRole : std::uint8_t {
    colour, // sRGB albedo/base-colour/emissive -> BC7, sRGB-tagged.
    generic, // Linear LDR data (metallic-roughness, occlusion, roughness) -> BC7, UNORM.
    normal_map, // Tangent-space XY normal -> BC5 (2 channels); shaders reconstruct Z.
};

struct CompressedMipLevel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t byte_offset = 0;
    std::uint32_t byte_length = 0;
};

// Fully CPU-side, block-compressed, fully-mipped texture, ready to hand to
// ImageStorage for GPU upload. Holds no Vulkan handles, so it's safe to
// build on a background thread concurrently with rendering. Split out of
// texture_pipeline.hxx (which does the CPU decode/encode/cache work that
// produces one of these) since ImageStorage::upgrade_pending_image needs
// the type itself without needing any of that.
struct CompressedTexture {
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<CompressedMipLevel> mips;
    std::vector<std::byte> data; // every mip concatenated, see CompressedMipLevel offsets
    std::string debug_name;
};
