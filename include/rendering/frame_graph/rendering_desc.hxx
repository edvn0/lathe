#pragma once

#include <volk.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    struct AttachmentResolve {
        std::uint32_t resource = 0;
        VkResolveModeFlagBits mode = VK_RESOLVE_MODE_AVERAGE_BIT;
    };

    struct AttachmentDesc {
        std::uint32_t resource = 0;
        LoadOp load = LoadOp::load;
        StoreOp store = StoreOp::store;
        VkClearValue clear{};
        std::optional<AttachmentResolve> resolve;
    };

    struct RenderingDesc {
        std::vector<AttachmentDesc> colors;
        std::optional<AttachmentDesc> depth;
        VkRect2D render_area{};
        std::uint32_t layer_count = 1;
        std::uint32_t view_mask = 0;
    };

}
