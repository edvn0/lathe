#pragma once

#include <volk.h>

#include <cstdint>

#include "rendering/frame_graph/types.hxx"
#include "rendering/frame_graph/vk_translate.hxx"

namespace frame_graph {

    // What a pass's record function receives while the executor runs it.
    struct PassContext {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::uint32_t frame_index = 0;
        LogicalQueue queue = LogicalQueue::graphics;
        PhysicalResources const *resources = nullptr;

        // The handles behind an id the pass declared. The slot must hold one.
        [[nodiscard]] auto image(ImageId id) const -> PhysicalImage const & { return resources->images[id.index]; }
        [[nodiscard]] auto buffer(BufferId id) const -> PhysicalBuffer const & { return resources->buffers[id.index]; }
    };

} // namespace frame_graph
