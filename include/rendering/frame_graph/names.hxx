#pragma once

#include <string_view>

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    [[nodiscard]] constexpr auto use_name(Use use) noexcept -> std::string_view {
        switch (use) {
            case Use::color_attachment:
                return "color attachment";
            case Use::color_resolve:
                return "color resolve";
            case Use::depth_attachment:
                return "depth attachment";
            case Use::depth_resolve:
                return "depth resolve";
            case Use::sampled:
                return "sampled";
            case Use::storage_read:
                return "storage read";
            case Use::storage_write:
                return "storage write";
            case Use::storage_read_write:
                return "storage read/write";
            case Use::transfer_src:
                return "transfer src";
            case Use::transfer_dst:
                return "transfer dst";
            case Use::present:
                return "present";
            case Use::indirect_read:
                return "indirect read";
            case Use::index_read:
                return "index read";
            case Use::shader_read:
                return "shader read";
            case Use::shader_write:
                return "shader write";
            case Use::shader_read_write:
                return "shader read/write";
            case Use::transfer_read:
                return "transfer read";
            case Use::transfer_write:
                return "transfer write";
            case Use::host_read:
                return "host read";
            case Use::token_write:
                return "token write";
            case Use::token_read:
                return "token read";
        }
        return "unknown";
    }

    [[nodiscard]] constexpr auto layout_name(VkImageLayout layout) noexcept -> std::string_view {
        switch (layout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return "undefined";
            case VK_IMAGE_LAYOUT_GENERAL:
                return "general";
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return "color attachment";
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                return "depth attachment";
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                return "depth read-only";
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return "shader read-only";
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return "transfer src";
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return "transfer dst";
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                return "present";
            default:
                return "other";
        }
    }

}
