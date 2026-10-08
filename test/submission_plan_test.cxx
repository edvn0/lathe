#include <doctest/doctest.h>

#include <array>

#include "gpu/submission_plan.hxx"

using frame_graph::LogicalQueue;
using frame_graph::SemaphoreWait;

namespace {

    constexpr auto separate_timelines = std::array<std::size_t, gpu_queue_count>{0, 1};
    constexpr auto aliased_timeline = std::array<std::size_t, gpu_queue_count>{0, 0};
    constexpr auto no_values = std::array<std::uint64_t, gpu_queue_count>{0, 0};

    auto graphics_batch(std::uint32_t signal_index, std::span<SemaphoreWait const> waits = {}) -> SubmitBatch {
        return SubmitBatch{.queue = LogicalQueue::graphics, .waits = waits, .signal_index = signal_index};
    }

    auto compute_batch(std::uint32_t signal_index, std::span<SemaphoreWait const> waits = {}) -> SubmitBatch {
        return SubmitBatch{.queue = LogicalQueue::compute, .waits = waits, .signal_index = signal_index};
    }

}

TEST_SUITE("unit") {
    TEST_CASE("one graphics batch signals the next value on its timeline") {
        auto batch = graphics_batch(0);
        batch.waits_swapchain_acquire = true;
        batch.signals_render_finished = true;
        auto const batches = std::array{batch};

        auto const plan = plan_submissions(batches, no_values, separate_timelines);
        REQUIRE(plan.has_value());
        REQUIRE(plan->submits.size() == 1);

        auto const &submit = plan->submits.front();
        CHECK(submit.signal_timeline == 0);
        CHECK(submit.signal_value == 1);
        CHECK(submit.waits.empty());
        CHECK(submit.waits_acquire);
        CHECK(submit.acquire_stages == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        CHECK(submit.signals_render_finished);
        CHECK(plan->timeline_values == std::array<std::uint64_t, gpu_queue_count>{1, 0});
    }

    TEST_CASE("values keep counting across frames") {
        auto const batches = std::array{graphics_batch(0)};
        auto const first = plan_submissions(batches, no_values, separate_timelines);
        REQUIRE(first.has_value());
        auto const second = plan_submissions(batches, first->timeline_values, separate_timelines);
        REQUIRE(second.has_value());
        auto const third = plan_submissions(batches, second->timeline_values, separate_timelines);
        REQUIRE(third.has_value());

        CHECK(first->submits.front().signal_value == 1);
        CHECK(second->submits.front().signal_value == 2);
        CHECK(third->submits.front().signal_value == 3);
        CHECK(third->timeline_values[0] == 3);
    }

    TEST_CASE("graphics, compute, graphics: each wait names the absolute value it needs") {
        auto const compute_waits = std::array{SemaphoreWait{
                .queue = LogicalQueue::graphics, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT}};
        auto const graphics_waits = std::array{SemaphoreWait{
                .queue = LogicalQueue::compute, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT}};
        auto const batches =
                std::array{graphics_batch(0), compute_batch(0, compute_waits), graphics_batch(1, graphics_waits)};

        auto const plan = plan_submissions(batches, {10, 4}, separate_timelines);
        REQUIRE(plan.has_value());
        REQUIRE(plan->submits.size() == 3);

        CHECK(plan->submits[0].signal_timeline == 0);
        CHECK(plan->submits[0].signal_value == 11);

        CHECK(plan->submits[1].signal_timeline == 1);
        CHECK(plan->submits[1].signal_value == 5);
        REQUIRE(plan->submits[1].waits.size() == 1);
        CHECK(plan->submits[1].waits.front() ==
              TimelineWait{.timeline = 0, .value = 11, .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

        CHECK(plan->submits[2].signal_timeline == 0);
        CHECK(plan->submits[2].signal_value == 12);
        REQUIRE(plan->submits[2].waits.size() == 1);
        CHECK(plan->submits[2].waits.front() ==
              TimelineWait{.timeline = 1, .value = 5, .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT});

        CHECK(plan->timeline_values == std::array<std::uint64_t, gpu_queue_count>{12, 5});
    }

    TEST_CASE("one physical queue: both logical queues count on the same timeline") {
        auto const compute_waits = std::array{SemaphoreWait{
                .queue = LogicalQueue::graphics, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
        auto const graphics_waits = std::array{SemaphoreWait{
                .queue = LogicalQueue::compute, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
        auto const batches =
                std::array{graphics_batch(0), compute_batch(0, compute_waits), graphics_batch(1, graphics_waits)};

        auto const plan = plan_submissions(batches, no_values, aliased_timeline);
        REQUIRE(plan.has_value());

        CHECK(plan->submits[0].signal_value == 1);
        CHECK(plan->submits[1].signal_value == 2);
        CHECK(plan->submits[1].waits.front().value == 1);
        CHECK(plan->submits[2].signal_value == 3);
        CHECK(plan->submits[2].waits.front().value == 2);
        for (auto const &submit: plan->submits) {
            CHECK(submit.signal_timeline == 0);
        }
        CHECK(plan->timeline_values == std::array<std::uint64_t, gpu_queue_count>{3, 0});
    }

    TEST_CASE("a wait with no stages waits for everything") {
        auto const waits = std::array{
                SemaphoreWait{.queue = LogicalQueue::graphics, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_NONE}};
        auto const batches = std::array{graphics_batch(0), compute_batch(0, waits)};

        auto const plan = plan_submissions(batches, no_values, separate_timelines);
        REQUIRE(plan.has_value());
        CHECK(plan->submits[1].waits.front().stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }

    TEST_CASE("a batch with no commands is still planned") {
        auto batch = graphics_batch(0);
        batch.command_buffer = VK_NULL_HANDLE;
        auto const batches = std::array{batch};
        auto const plan = plan_submissions(batches, no_values, separate_timelines);
        REQUIRE(plan.has_value());
        CHECK(plan->submits.size() == 1);
    }

    TEST_CASE("bad batch lists are rejected") {
        SUBCASE("a signal index that skips ahead") {
            auto const batches = std::array{graphics_batch(1)};
            auto const plan = plan_submissions(batches, no_values, separate_timelines);
            REQUIRE_FALSE(plan.has_value());
            CHECK(plan.error() == SubmissionPlanError::signal_out_of_order);
        }
        SUBCASE("a signal index that repeats") {
            auto const batches = std::array{graphics_batch(0), graphics_batch(0)};
            auto const plan = plan_submissions(batches, no_values, separate_timelines);
            REQUIRE_FALSE(plan.has_value());
            CHECK(plan.error() == SubmissionPlanError::signal_out_of_order);
        }
        SUBCASE("a wait for a signal that comes later") {
            auto const waits = std::array{SemaphoreWait{
                    .queue = LogicalQueue::compute, .signal_index = 0, .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
            auto const batches = std::array{graphics_batch(0, waits), compute_batch(0)};
            auto const plan = plan_submissions(batches, no_values, separate_timelines);
            REQUIRE_FALSE(plan.has_value());
            CHECK(plan.error() == SubmissionPlanError::wait_before_signal);
        }
        SUBCASE("a wait for a signal that never happens") {
            auto const waits = std::array{SemaphoreWait{.queue = LogicalQueue::graphics,
                                                        .signal_index = 7,
                                                        .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
            auto const batches = std::array{graphics_batch(0), compute_batch(0, waits)};
            auto const plan = plan_submissions(batches, no_values, separate_timelines);
            REQUIRE_FALSE(plan.has_value());
            CHECK(plan.error() == SubmissionPlanError::wait_before_signal);
        }
    }

    TEST_CASE("an empty frame plans nothing and leaves the timelines alone") {
        auto const plan = plan_submissions({}, {3, 2}, separate_timelines);
        REQUIRE(plan.has_value());
        CHECK(plan->submits.empty());
        CHECK(plan->timeline_values == std::array<std::uint64_t, gpu_queue_count>{3, 2});
    }
}
