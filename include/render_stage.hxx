#pragma once

#include <cstdint>
#include <string_view>


enum class RenderStage : std::uint32_t {
    FullFrame = 0,
    Culling,
    ShadowPass,
    DepthPrepass,
    ForwardPass,
    Composition,
    BloomPass,
    Count
};

constexpr auto to_string(RenderStage stage) -> std::string_view {
    switch (stage) {
        using enum RenderStage;
        case FullFrame:
            return "Full Frame";
        case Culling:
            return "GPU Culling";
        case ShadowPass:
            return "Shadow Pass";
        case DepthPrepass:
            return "Depth prepass";
        case ForwardPass:
            return "Forward Pass";
        case Composition:
            return "Composition Pass";
        case BloomPass:
            return "Bloom pass";
        default:
            return "Unknown";
    }
}
inline constexpr std::uint32_t stage_count = static_cast<std::uint32_t>(RenderStage::Count);
inline constexpr std::uint32_t query_count = stage_count * 2;
