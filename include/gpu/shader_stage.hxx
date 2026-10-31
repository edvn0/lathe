#pragma once

#include <string>

inline constexpr VkPushConstantRange global_push_constant_range{
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = 256,
};

struct ShaderStageInfo {
    VkShaderStageFlagBits stage{};
    std::span<const std::uint32_t> spirv;
    FlyString entry_point = FlyString{"main"};
    VkPipelineShaderStageCreateFlags flags = 0;
    VkSpecializationInfo const *specialization_info = nullptr;

    // Owned: callers build it on the spot, and a view would dangle.
    std::string cache_key{};
};
