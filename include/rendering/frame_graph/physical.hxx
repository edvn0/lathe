#pragma once

#include <volk.h>

#include <cstdint>

namespace frame_graph {

    // The Vulkan handles behind an image or buffer the graph touches. Opaque to the compiler, which never reads them
    // and leaves them out of the declaration hash: the executor turns barriers and attachments into Vulkan structures
    // with them. Plain handles only, so the compiler still builds without a device.
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

} // namespace frame_graph
