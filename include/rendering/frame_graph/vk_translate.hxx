#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/physical.hxx"
#include "rendering/frame_graph/rendering_desc.hxx"

// Pure translation from a compiled plan to Vulkan structures (docs/frame-graph.md, phase 3). No Vulkan calls, so it is
// unit-tested without a device; the executor records the results.
namespace frame_graph {

    // The handles behind each resource slot, by slot index. A slot the graph does not touch can stay empty.
    struct PhysicalResources {
        std::vector<PhysicalImage> images;
        std::vector<PhysicalBuffer> buffers;

        [[nodiscard]] auto image(std::uint32_t resource) const noexcept -> PhysicalImage const * {
            return resource < images.size() && images[resource].image != VK_NULL_HANDLE ? &images[resource] : nullptr;
        }

        [[nodiscard]] auto buffer(std::uint32_t resource) const noexcept -> PhysicalBuffer const * {
            return resource < buffers.size() && buffers[resource].buffer != VK_NULL_HANDLE ? &buffers[resource]
                                                                                           : nullptr;
        }
    };

    // The handles of every imported resource in `graph`. Transients get theirs from the allocator.
    [[nodiscard]] auto physical_resources_of(GraphDesc const &graph) -> PhysicalResources;

    enum class TranslateFailureKind : std::uint8_t { missing_image, missing_buffer };

    struct TranslateFailure {
        TranslateFailureKind kind = TranslateFailureKind::missing_image;
        std::uint32_t resource = 0;
    };

    // The aspects a barrier or view on an image of this format covers: depth, depth and stencil, stencil or colour.
    [[nodiscard]] auto image_aspect(VkFormat format) noexcept -> VkImageAspectFlags;

    // The arrays one vkCmdPipelineBarrier2 points into. info() returns pointers into this object, so keep it alive and
    // in place until the command is recorded.
    struct DependencyStorage {
        std::vector<VkMemoryBarrier2> memory;
        std::vector<VkImageMemoryBarrier2> images;
        std::vector<VkBufferMemoryBarrier2> buffers;

        [[nodiscard]] auto empty() const noexcept -> bool {
            return memory.empty() && images.empty() && buffers.empty();
        }

        [[nodiscard]] auto info() const noexcept -> VkDependencyInfo;
    };

    // Image barriers cover every mip and layer. Buffer barriers are always the whole buffer. Ownership release and
    // acquire halves keep their queue family indices; everything else uses VK_QUEUE_FAMILY_IGNORED.
    [[nodiscard]] auto translate(BarrierSet const &barriers, PhysicalResources const &resources)
            -> std::expected<DependencyStorage, TranslateFailure>;

    struct RenderingStorage {
        std::vector<VkRenderingAttachmentInfo> colors;
        VkRenderingAttachmentInfo depth{};
        bool has_depth = false;
        VkRect2D render_area{};
        std::uint32_t layer_count = 1;
        std::uint32_t view_mask = 0;

        // Pointers into this object, as for DependencyStorage.
        [[nodiscard]] auto info() const noexcept -> VkRenderingInfo;
    };

    // Colour attachments are in COLOR_ATTACHMENT_OPTIMAL and depth in DEPTH_ATTACHMENT_OPTIMAL, the layouts the use
    // table gives them, with the resolve target in the same layout.
    [[nodiscard]] auto translate(RenderingDesc const &rendering, PhysicalResources const &resources)
            -> std::expected<RenderingStorage, TranslateFailure>;

} // namespace frame_graph
