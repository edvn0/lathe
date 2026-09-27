#pragma once

#include <volk.h>

#include <cstdint>
#include <string>
#include <vector>

// How source pixels are interpreted; decides the encode parameters and the final block format.
enum class TextureRole : std::uint8_t {
    colour, // sRGB albedo/emissive -> BC7 sRGB.
    generic, // Linear LDR data (metallic-roughness, occlusion) -> BC7 UNORM.
    normal_map, // Tangent-space XY normal -> BC5; shaders reconstruct Z.
};

struct CompressedMipLevel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t byte_offset = 0;
    std::uint32_t byte_length = 0;
};

// CPU-side, block-compressed, mipped texture ready for ImageStorage. No Vulkan handles, so it can be built on
// any thread.
struct CompressedTexture {
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<CompressedMipLevel> mips;
    std::vector<std::byte> data; // all mips concatenated; see CompressedMipLevel
    std::string debug_name;
};
