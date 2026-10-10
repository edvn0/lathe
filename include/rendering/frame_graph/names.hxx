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

    [[nodiscard]] constexpr auto format_name(VkFormat format) noexcept -> std::string_view {
        switch (format) {
            case VK_FORMAT_UNDEFINED:
                return "undefined";
            case VK_FORMAT_R8_UNORM:
                return "R8 UNORM";
            case VK_FORMAT_R8G8B8A8_UNORM:
                return "RGBA8 UNORM";
            case VK_FORMAT_R8G8B8A8_SRGB:
                return "RGBA8 SRGB";
            case VK_FORMAT_B8G8R8A8_UNORM:
                return "BGRA8 UNORM";
            case VK_FORMAT_B8G8R8A8_SRGB:
                return "BGRA8 SRGB";
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return "RGBA16 SFLOAT";
            case VK_FORMAT_R32_SFLOAT:
                return "R32 SFLOAT";
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return "RGBA32 SFLOAT";
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
                return "B10G11R11 UFLOAT";
            case VK_FORMAT_D16_UNORM:
                return "D16 UNORM";
            case VK_FORMAT_D32_SFLOAT:
                return "D32 SFLOAT";
            case VK_FORMAT_D24_UNORM_S8_UINT:
                return "D24S8";
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return "D32S8";
            default:
                return "other format";
        }
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
