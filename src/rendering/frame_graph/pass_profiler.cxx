#include "rendering/frame_graph/pass_profiler.hxx"

#include "core/logger.hxx"

namespace frame_graph {
    namespace {

        constexpr auto queue_index(LogicalQueue queue) noexcept -> std::size_t {
            return static_cast<std::size_t>(queue);
        }

    } // namespace

    PassProfiler::~PassProfiler() { destroy(); }

    auto PassProfiler::initialize(PassProfilerCreateInfo const &create_info) -> bool {
        destroy();

        if (create_info.device == VK_NULL_HANDLE || create_info.slots == 0 || create_info.max_passes == 0) {
            error("Invalid pass profiler create info");
            return false;
        }

        device_ = create_info.device;
        timestamp_period_ = create_info.timestamp_period;
        max_passes_ = create_info.max_passes;

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(create_info.physical_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(create_info.physical_device, &family_count, families.data());

        for (auto queue = std::size_t{0}; queue < logical_queue_count; ++queue) {
            auto const family = create_info.queue_family[queue];
            valid_bits_[queue] = family < families.size() ? families[family].timestampValidBits : 0;
        }

        pools_.resize(create_info.slots);

        for (auto &slot: pools_) {
            for (auto queue = std::size_t{0}; queue < logical_queue_count; ++queue) {
                if (valid_bits_[queue] == 0) {
                    continue;
                }

                VkQueryPoolCreateInfo const pool_info{
                        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .queryType = VK_QUERY_TYPE_TIMESTAMP,
                        .queryCount = 2 * max_passes_,
                        .pipelineStatistics = 0,
                };

                auto const result = vkCreateQueryPool(device_, &pool_info, nullptr, &slot[queue].pool);
                if (result != VK_SUCCESS) {
                    error("vkCreateQueryPool(pass profiler) failed with VkResult {}", static_cast<int>(result));
                    destroy();
                    return false;
                }

                vkResetQueryPool(device_, slot[queue].pool, 0, 2 * max_passes_);
            }
        }

        return true;
    }

    auto PassProfiler::destroy() -> void {
        if (device_ != VK_NULL_HANDLE) {
            for (auto &slot: pools_) {
                for (auto &queue: slot) {
                    if (queue.pool != VK_NULL_HANDLE) {
                        vkDestroyQueryPool(device_, queue.pool, nullptr);
                        queue.pool = VK_NULL_HANDLE;
                    }
                }
            }
        }

        pools_.clear();
        timings_.clear();
        valid_bits_ = {};
        device_ = VK_NULL_HANDLE;
    }

    auto PassProfiler::begin_slot(std::uint32_t slot) -> void {
        if (slot >= pools_.size()) {
            return;
        }

        timings_.clear();

        for (auto queue = std::size_t{0}; queue < logical_queue_count; ++queue) {
            auto &entry = pools_[slot][queue];

            auto const logical = queue == 0 ? LogicalQueue::graphics : LogicalQueue::compute;

            if (entry.pool == VK_NULL_HANDLE) {
                // No timestamps on this queue: still list the passes, without a time.
                for (auto const &written: entry.written) {
                    timings_.push_back(PassTiming{.name_id = written.name_id,
                                                  .label = written.label,
                                                  .queue = logical,
                                                  .milliseconds = std::nullopt});
                }
                entry.written.clear();
                continue;
            }

            if (!entry.written.empty()) {
                std::vector<std::uint64_t> ticks(2 * entry.written.size());
                auto const result =
                        vkGetQueryPoolResults(device_, entry.pool, 0, static_cast<std::uint32_t>(ticks.size()),
                                              ticks.size() * sizeof(std::uint64_t), ticks.data(), sizeof(std::uint64_t),
                                              VK_QUERY_RESULT_64_BIT);

                auto const bits = valid_bits_[queue];
                auto const mask = bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;

                for (auto index = std::size_t{0}; index < entry.written.size(); ++index) {
                    auto const &written = entry.written[index];
                    auto timing = PassTiming{.name_id = written.name_id,
                                             .label = written.label,
                                             .queue = logical,
                                             .milliseconds = std::nullopt};

                    if (result == VK_SUCCESS) {
                        auto const begin = ticks[2 * static_cast<std::size_t>(written.timestamp_slot)];
                        auto const end = ticks[2 * static_cast<std::size_t>(written.timestamp_slot) + 1];
                        // The counter may wrap within its valid bits.
                        auto const delta = (end - begin) & mask;
                        timing.milliseconds = static_cast<float>(delta) * timestamp_period_ / 1'000'000.0F;
                    }

                    timings_.push_back(std::move(timing));
                }
            }

            entry.written.clear();
            vkResetQueryPool(device_, entry.pool, 0, 2 * max_passes_);
        }
    }

    auto PassProfiler::write_begin(VkCommandBuffer command_buffer, LogicalQueue queue, std::uint32_t slot,
                                   std::uint32_t timestamp_slot, std::string_view name_id, std::string_view label)
            -> void {
        if (slot >= pools_.size()) {
            return;
        }

        auto &entry = pools_[slot][queue_index(queue)];

        // Without timestamps, remember the pass so it is still listed.
        if (timestamp_slot < max_passes_) {
            entry.written.push_back(Written{
                    .name_id = std::string{name_id}, .label = std::string{label}, .timestamp_slot = timestamp_slot});
        }

        if (entry.pool == VK_NULL_HANDLE || timestamp_slot >= max_passes_) {
            return;
        }

        // ALL_COMMANDS, not TOP_OF_PIPE: the begin timestamp is written once every earlier command on the queue has
        // finished. A TOP_OF_PIPE begin is written as soon as the pass is reached, while its end waits for all earlier
        // work, so a cheap pass recorded behind an expensive one reported that one's remaining time as its own (e.g.
        // cluster_stats_clear reading exactly gpu_culling's time). Now passes on a queue don't overlap: a pass's time
        // is from the queue draining before it to its own end, and the times of one queue add up to at most its span.
        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, entry.pool, 2 * timestamp_slot);
    }

    auto PassProfiler::write_end(VkCommandBuffer command_buffer, LogicalQueue queue, std::uint32_t slot,
                                 std::uint32_t timestamp_slot) -> void {
        if (slot >= pools_.size() || timestamp_slot >= max_passes_) {
            return;
        }

        auto const &entry = pools_[slot][queue_index(queue)];
        if (entry.pool == VK_NULL_HANDLE) {
            return;
        }

        vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, entry.pool, 2 * timestamp_slot + 1);
    }

    auto PassProfiler::source_location([[maybe_unused]] std::string_view label, [[maybe_unused]] std::uint32_t color)
            -> tracy::SourceLocationData const * {
#ifdef TRACY_ENABLE
        auto key = std::pair<std::string, std::uint32_t>{std::string{label}, color};
        if (auto const found = interned_.find(key); found != interned_.end()) {
            return found->second;
        }

        // Tracy keeps these pointers for the life of the process, so the strings and the location are never freed.
        auto const &name = names_.emplace_back(label);
        auto const &location = locations_.emplace_back(tracy::SourceLocationData{
                .name = name.c_str(),
                .function = "frame_graph",
                .file = "frame_graph",
                .line = 0,
                .color = color,
        });
        interned_.emplace(std::move(key), &location);
        return &location;
#else
        return nullptr;
#endif
    }

} // namespace frame_graph
