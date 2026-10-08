#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/physical.hxx"
#include "rendering/frame_graph/rendering_desc.hxx"

namespace frame_graph {

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

    [[nodiscard]] auto physical_resources_of(GraphDesc const &graph) -> PhysicalResources;

    enum class TranslateFailureKind : std::uint8_t { missing_image, missing_buffer };

    struct TranslateFailure {
        TranslateFailureKind kind = TranslateFailureKind::missing_image;
        std::uint32_t resource = 0;
    };

    [[nodiscard]] auto image_aspect(VkFormat format) noexcept -> VkImageAspectFlags;

    struct DependencyStorage {
        std::vector<VkMemoryBarrier2> memory;
        std::vector<VkImageMemoryBarrier2> images;
        std::vector<VkBufferMemoryBarrier2> buffers;

        [[nodiscard]] auto empty() const noexcept -> bool {
            return memory.empty() && images.empty() && buffers.empty();
        }

        [[nodiscard]] auto info() const noexcept -> VkDependencyInfo;
    };

    [[nodiscard]] auto translate(BarrierSet const &barriers, PhysicalResources const &resources)
            -> std::expected<DependencyStorage, TranslateFailure>;

    struct RenderingStorage {
        std::vector<VkRenderingAttachmentInfo> colors;
        VkRenderingAttachmentInfo depth{};
        bool has_depth = false;
        VkRect2D render_area{};
        std::uint32_t layer_count = 1;
        std::uint32_t view_mask = 0;

        [[nodiscard]] auto info() const noexcept -> VkRenderingInfo;
    };

    [[nodiscard]] auto translate(RenderingDesc const &rendering, PhysicalResources const &resources)
            -> std::expected<RenderingStorage, TranslateFailure>;

}
