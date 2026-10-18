#include "gpu/submission_plan.hxx"

#include <algorithm>

namespace {

    constexpr auto queue_index(LogicalQueue queue) noexcept -> std::size_t {
        return static_cast<std::size_t>(queue);
    }

}

auto plan_submissions(std::span<SubmitBatch const> batches, std::array<std::uint64_t, gpu_queue_count> timeline_values,
                      std::array<std::size_t, gpu_queue_count> timeline_of_queue)
        -> std::expected<SubmissionPlan, SubmissionPlanError> {
    auto plan = SubmissionPlan{};
    auto planned = timeline_values;

    auto values = std::array<std::vector<std::uint64_t>, gpu_queue_count>{};

    // Waits each queue has already issued: a later wait on a value at or below one of these, for stages the earlier wait
    // covered, adds nothing, because everything submitted after a wait is ordered behind it.
    auto issued = std::array<std::array<std::vector<TimelineWait>, gpu_queue_count>, gpu_queue_count>{};

    for (auto index = std::size_t{0}; index < batches.size(); ++index) {
        auto const &batch = batches[index];
        auto const q = queue_index(batch.queue);
        auto const timeline = timeline_of_queue[q];

        if (batch.signal_index != values[q].size()) {
            return std::unexpected(SubmissionPlanError::signal_out_of_order);
        }

        auto submit = PlannedSubmit{
                .batch = index,
                .waits_acquire = batch.waits_swapchain_acquire,
                .acquire_stages = batch.swapchain_wait_stages,
                .signal_timeline = timeline,
                .signals_render_finished = batch.signals_render_finished,
        };

        for (auto const &wait: batch.waits) {
            auto const waited = queue_index(wait.queue);
            if (wait.signal_index >= values[waited].size()) {
                return std::unexpected(SubmissionPlanError::wait_before_signal);
            }
            auto const timeline_wait = TimelineWait{
                    .timeline = timeline_of_queue[waited],
                    .value = values[waited][wait.signal_index],
                    .stages = wait.stages != 0 ? wait.stages : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            };
            auto &earlier = issued[q][waited];
            auto const redundant = std::ranges::any_of(earlier, [&](TimelineWait const &prior) {
                auto const covers_stages = (prior.stages & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0 ||
                                           (timeline_wait.stages & ~prior.stages) == 0;
                return prior.value >= timeline_wait.value && covers_stages;
            });
            if (redundant) {
                continue;
            }
            earlier.push_back(timeline_wait);
            submit.waits.push_back(timeline_wait);
        }

        planned[timeline] += 1;
        values[q].push_back(planned[timeline]);
        submit.signal_value = planned[timeline];
        plan.submits.push_back(std::move(submit));
    }

    plan.timeline_values = planned;
    return plan;
}
