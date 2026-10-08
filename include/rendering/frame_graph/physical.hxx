#pragma once

#include <volk.h>

#include <cstdint>

namespace frame_graph {

    struct PhysicalImage {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent3D extent{};
        std::uint32_t mip_levels = 1;
        std::uint32_t array_layers = 1;
    };

    struct PhysicalBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        VkDeviceSize size = 0;
    };

}
