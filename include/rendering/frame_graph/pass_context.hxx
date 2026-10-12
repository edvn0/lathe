#pragma once

#include <volk.h>

#include <algorithm>
#include <cstdint>
#include <span>

#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/types.hxx"
#include "rendering/frame_graph/vk_translate.hxx"

namespace frame_graph {

    struct PassContext {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::uint32_t frame_index = 0;
        LogicalQueue queue = LogicalQueue::graphics;
        PhysicalResources const *resources = nullptr;

        // What the running pass declared. Empty for a context built outside the executor, which declares nothing.
        std::span<AccessDesc const> accesses;

        [[nodiscard]] auto image(ImageId id) const -> PhysicalImage const & { return resources->images[id.index]; }
        [[nodiscard]] auto buffer(BufferId id) const -> PhysicalBuffer const & { return resources->buffers[id.index]; }

        // Whether the running pass declared an access to the resource, and for `writable` one that writes it.
        // Versions are not compared: a handle from before the pass's own write still names the same resource.
        [[nodiscard]] auto declares(std::uint32_t resource, bool writable) const -> bool {
            return std::ranges::any_of(accesses, [&](AccessDesc const &access) {
                return access.resource == resource && (!writable || access.produces);
            });
        }
        [[nodiscard]] auto declares(ImageId id, bool writable = false) const -> bool {
            return declares(id.index, writable);
        }
        [[nodiscard]] auto declares(BufferId id, bool writable = false) const -> bool {
            return declares(id.index, writable);
        }
    };

}
