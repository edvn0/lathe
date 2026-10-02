#pragma once

#include "rendering/frame_graph/types.hxx"

namespace frame_graph {

    // Every access bit that writes memory. A barrier's source scope only needs these made available.
    inline constexpr VkAccessFlags2 write_access_mask =
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
            VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

    struct UseInfo {
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; // UNDEFINED for buffers and tokens
        bool is_image = false;
        bool is_token = false;
        bool reads = false;
        bool writes = false;
        bool needs_shader_stages = false;
    };

    [[nodiscard]] constexpr auto is_image_use(Use use) noexcept -> bool { return use <= Use::present; }

    [[nodiscard]] constexpr auto is_token_use(Use use) noexcept -> bool {
        return use == Use::token_write || use == Use::token_read;
    }

    [[nodiscard]] constexpr auto is_buffer_use(Use use) noexcept -> bool {
        return !is_image_use(use) && !is_token_use(use);
    }

    [[nodiscard]] constexpr auto is_attachment_use(Use use) noexcept -> bool {
        return use == Use::color_attachment || use == Use::color_resolve || use == Use::depth_attachment ||
               use == Use::depth_resolve;
    }

    [[nodiscard]] constexpr auto shader_stage_flags(ShaderStages stages) noexcept -> VkPipelineStageFlags2 {
        auto flags = VkPipelineStageFlags2{VK_PIPELINE_STAGE_2_NONE};
        if ((stages & static_cast<ShaderStages>(ShaderStage::vertex)) != 0) {
            flags |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        }
        if ((stages & static_cast<ShaderStages>(ShaderStage::task)) != 0) {
            flags |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        }
        if ((stages & static_cast<ShaderStages>(ShaderStage::mesh)) != 0) {
            flags |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
        }
        if ((stages & static_cast<ShaderStages>(ShaderStage::fragment)) != 0) {
            flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        }
        if ((stages & static_cast<ShaderStages>(ShaderStage::compute)) != 0) {
            flags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        }
        return flags;
    }

    // The single source of truth for what a Use means to Vulkan. `stages` only matters for uses with
    // needs_shader_stages. Whether an attachment use discards its contents depends on its load op; see
    // `discards_contents`.
    [[nodiscard]] constexpr auto use_info(Use use, ShaderStages stages) noexcept -> UseInfo {
        auto const shader = shader_stage_flags(stages);
        switch (use) {
            case Use::color_attachment:
                return {
                        .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        .access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .is_image = true,
                        .reads = true,
                        .writes = true,
                };
            case Use::color_resolve:
                return {
                        .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .is_image = true,
                        .writes = true,
                };
            case Use::depth_attachment:
                return {
                        .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        .is_image = true,
                        .reads = true,
                        .writes = true,
                };
            case Use::depth_resolve:
                // The union of stages the existing code relies on: depth resolves run in COLOR_ATTACHMENT_OUTPUT.
                return {
                        .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                        .access =
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        .is_image = true,
                        .writes = true,
                };
            case Use::sampled:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        .is_image = true,
                        .reads = true,
                        .needs_shader_stages = true,
                };
            case Use::storage_read:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        .layout = VK_IMAGE_LAYOUT_GENERAL,
                        .is_image = true,
                        .reads = true,
                        .needs_shader_stages = true,
                };
            case Use::storage_write:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_GENERAL,
                        .is_image = true,
                        .writes = true,
                        .needs_shader_stages = true,
                };
            case Use::storage_read_write:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_GENERAL,
                        .is_image = true,
                        .reads = true,
                        .writes = true,
                        .needs_shader_stages = true,
                };
            case Use::transfer_src:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                        .access = VK_ACCESS_2_TRANSFER_READ_BIT,
                        .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .is_image = true,
                        .reads = true,
                };
            case Use::transfer_dst:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                        .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .is_image = true,
                        .writes = true,
                };
            case Use::present:
                return {
                        .stages = VK_PIPELINE_STAGE_2_NONE,
                        .access = VK_ACCESS_2_NONE,
                        .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        .is_image = true,
                        .reads = true,
                };
            case Use::indirect_read:
                return {
                        .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                        .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                        .reads = true,
                };
            case Use::index_read:
                return {
                        .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                        .access = VK_ACCESS_2_INDEX_READ_BIT,
                        .reads = true,
                };
            case Use::shader_read:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                        .reads = true,
                        .needs_shader_stages = true,
                };
            case Use::shader_write:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .writes = true,
                        .needs_shader_stages = true,
                };
            case Use::shader_read_write:
                return {
                        .stages = shader,
                        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        .reads = true,
                        .writes = true,
                        .needs_shader_stages = true,
                };
            case Use::transfer_read:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                        .access = VK_ACCESS_2_TRANSFER_READ_BIT,
                        .reads = true,
                };
            case Use::transfer_write:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                        .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .writes = true,
                };
            case Use::host_read:
                return {
                        .stages = VK_PIPELINE_STAGE_2_HOST_BIT,
                        .access = VK_ACCESS_2_HOST_READ_BIT,
                        .reads = true,
                };
            case Use::token_write:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .access = VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .is_token = true,
                        .writes = true,
                };
            case Use::token_read:
                return {
                        .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        .access = VK_ACCESS_2_MEMORY_READ_BIT,
                        .is_token = true,
                        .reads = true,
                };
        }
        return {};
    }

    // Whether an access throws away the previous contents. Pure writes through transfer_dst and storage_write discard;
    // attachments discard when they do not load. Buffers never discard implicitly.
    [[nodiscard]] constexpr auto discards_contents(Use use, LoadOp load) noexcept -> bool {
        switch (use) {
            case Use::color_attachment:
            case Use::depth_attachment:
                return load != LoadOp::load;
            case Use::color_resolve:
            case Use::depth_resolve:
            case Use::storage_write:
            case Use::transfer_dst:
                return true;
            default:
                return false;
        }
    }

} // namespace frame_graph
