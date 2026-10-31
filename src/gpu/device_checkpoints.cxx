#include "gpu/device_checkpoints.hxx"

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/logger.hxx"
#include "gpu/context.hxx"

namespace {
    std::atomic_bool is_enabled{false}; // NOLINT: process-wide, set once at device creation.

    std::mutex names_mutex;
    std::deque<std::string> names;               // NOLINT: stable addresses, guarded by names_mutex.
    std::unordered_set<std::string_view> interned; // NOLINT: views into 'names', guarded by names_mutex.

    auto intern(std::string_view name) -> char const * {
        std::scoped_lock const lock{names_mutex};

        if (auto const it = interned.find(name); it != interned.end()) {
            return it->data();
        }

        auto const &stored = names.emplace_back(name);
        interned.insert(std::string_view{stored});

        return stored.c_str();
    }

    auto stage_name(VkPipelineStageFlagBits stage) -> char const * {
        switch (stage) {
            case VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:
                return "top of pipe";
            case VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT:
                return "bottom of pipe";
            case VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT:
                return "compute shader";
            case VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:
                return "fragment shader";
            case VK_PIPELINE_STAGE_VERTEX_SHADER_BIT:
                return "vertex shader";
            case VK_PIPELINE_STAGE_TRANSFER_BIT:
                return "transfer";
            default:
                return "other stage";
        }
    }

    auto report_queue(char const *queue_name, VkQueue queue) -> void {
        std::uint32_t count = 0;
        vkGetQueueCheckpointDataNV(queue, &count, nullptr);

        if (count == 0) {
            info("Checkpoints, {} queue: none reached the GPU", queue_name);
            return;
        }

        std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV{.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
        vkGetQueueCheckpointDataNV(queue, &count, data.data());

        for (std::uint32_t index = 0; index < count; ++index) {
            auto const *marker = static_cast<char const *>(data[index].pCheckpointMarker);

            error("Checkpoints, {} queue: last reached '{}' ({})", queue_name, marker != nullptr ? marker : "?",
                  stage_name(data[index].stage));
        }
    }
}

namespace device_checkpoints {

    auto enable(bool supported) noexcept -> void {
        is_enabled.store(supported && vkCmdSetCheckpointNV != nullptr && vkGetQueueCheckpointDataNV != nullptr,
                         std::memory_order_release);
    }

    auto enabled() noexcept -> bool { return is_enabled.load(std::memory_order_acquire); }

    auto set(VkCommandBuffer commands, std::string_view name) -> void {
        if (enabled()) {
            vkCmdSetCheckpointNV(commands, intern(name));
        }
    }

    auto report(VulkanContext const &context) -> void {
        if (!enabled()) {
            info("Checkpoints are unavailable (VK_NV_device_diagnostic_checkpoints is not enabled)");
            return;
        }

        report_queue("graphics", context.graphics_queue);

        if (context.compute_queue != context.graphics_queue) {
            report_queue("compute", context.compute_queue);
        }

        if (context.background_queue != VK_NULL_HANDLE) {
            report_queue("background", context.background_queue);
        }
    }

}
