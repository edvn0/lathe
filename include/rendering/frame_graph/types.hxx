#pragma once

#include <cstdint>
#include <string_view>

#include <volk.h>

#include "core/handle.hxx"

namespace frame_graph {

    enum class LogicalQueue : std::uint8_t { graphics, compute };
    inline constexpr std::size_t logical_queue_count = 2;

    enum class QueueAffinity : std::uint8_t { graphics, compute_preferred, compute_required };

    enum class PassType : std::uint8_t { raster, compute, transfer };

    enum class Use : std::uint8_t {
        color_attachment,
        color_resolve,
        depth_attachment,
        depth_resolve,
        sampled,
        storage_read,
        storage_write,
        storage_read_write,
        transfer_src,
        transfer_dst,
        present,
        indirect_read,
        index_read,
        shader_read,
        shader_write,
        shader_read_write,
        transfer_read,
        transfer_write,
        host_read,
        token_write,
        token_read,
    };

    enum class LoadOp : std::uint8_t { load, clear, dont_care };
    enum class StoreOp : std::uint8_t { store, dont_care };

    enum class ShaderStage : std::uint8_t {
        vertex = 1U << 0U,
        task = 1U << 1U,
        mesh = 1U << 2U,
        fragment = 1U << 3U,
        compute = 1U << 4U,
    };
    using ShaderStages = std::uint8_t;

    [[nodiscard]] constexpr auto stages_of(ShaderStage first) noexcept -> ShaderStages {
        return static_cast<ShaderStages>(first);
    }

    [[nodiscard]] constexpr auto operator|(ShaderStage lhs, ShaderStage rhs) noexcept -> ShaderStages {
        return static_cast<ShaderStages>(static_cast<ShaderStages>(lhs) | static_cast<ShaderStages>(rhs));
    }

    [[nodiscard]] constexpr auto operator|(ShaderStages lhs, ShaderStage rhs) noexcept -> ShaderStages {
        return static_cast<ShaderStages>(lhs | static_cast<ShaderStages>(rhs));
    }

    struct ImageTag;
    struct BufferTag;
    struct PassTag;

    using ImageId = Handle<ImageTag, 0>;
    using BufferId = Handle<BufferTag, 0>;
    using PassId = Handle<PassTag, 0>;

    struct ResourceState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
        LogicalQueue queue = LogicalQueue::graphics;

        auto operator==(ResourceState const &) const -> bool = default;
    };

    enum class Sharing : std::uint8_t { exclusive, concurrent };

}
