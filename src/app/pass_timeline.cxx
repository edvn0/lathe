#include "app/pass_timeline.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string_view>

#include <imgui.h>

#include "rendering/renderer.hxx"

namespace gui {
    namespace {

        constexpr auto lane_height = 38.0F;
        constexpr auto lane_gap = 10.0F;
        constexpr auto label_width = 70.0F;
        constexpr auto ruler_height = 22.0F;

        auto lane_label(frame_graph::LogicalQueue queue) -> char const * {
            return queue == frame_graph::LogicalQueue::graphics ? "graphics" : "compute";
        }

        // A stable per-pass tint so the same pass keeps its colour from frame to frame.
        auto bar_colour(frame_graph::LogicalQueue queue, std::string_view name_id) -> ImU32 {
            auto const hash = std::hash<std::string_view>{}(name_id);
            auto const shade = static_cast<float>(hash % 5U) * 0.07F;
            return queue == frame_graph::LogicalQueue::graphics
                           ? ImGui::GetColorU32(ImVec4{0.30F + shade, 0.50F + shade, 0.85F, 1.0F})
                           : ImGui::GetColorU32(ImVec4{0.30F + shade, 0.75F, 0.45F + shade, 1.0F});
        }

        auto nice_step(float span) -> float {
            auto const raw = span / 8.0F;
            auto const magnitude = std::pow(10.0F, std::floor(std::log10(std::max(raw, 0.0001F))));
            for (auto const multiple: std::array{1.0F, 2.0F, 5.0F, 10.0F}) {
                if (raw <= multiple * magnitude) {
                    return multiple * magnitude;
                }
            }
            return 10.0F * magnitude;
        }

    }

    auto draw_pass_timeline(Renderer const &renderer) -> void {
        auto const timings = renderer.frame_graph_timings();

        auto span = 0.0F;
        auto busy = std::array<float, frame_graph::logical_queue_count>{};
        for (auto const &timing: timings) {
            if (timing.milliseconds && timing.start_milliseconds) {
                span = std::max(span, *timing.start_milliseconds + *timing.milliseconds);
                busy[static_cast<std::size_t>(timing.queue)] += *timing.milliseconds;
            }
        }

        if (span <= 0.0F) {
            ImGui::TextDisabled("No GPU timings yet");
            return;
        }

        auto const total_busy = busy[0] + busy[1];
        ImGui::Text("Frame span %.3f ms \xC2\xB7 graphics busy %.3f ms \xC2\xB7 compute busy %.3f ms",
                    static_cast<double>(span), static_cast<double>(busy[0]), static_cast<double>(busy[1]));
        ImGui::TextDisabled("Overlap hides %.3f ms of work", static_cast<double>(std::max(total_busy - span, 0.0F)));

        auto const origin = ImGui::GetCursorScreenPos();
        auto const width = std::max(ImGui::GetContentRegionAvail().x, label_width + 50.0F);
        auto const track_width = width - label_width;
        auto const scale = track_width / span;
        auto const height = ruler_height + (2.0F * lane_height) + lane_gap;
        auto *draw = ImGui::GetWindowDrawList();

        ImGui::InvisibleButton("##timeline", ImVec2{width, height});

        auto const text_colour = ImGui::GetColorU32(ImGuiCol_TextDisabled);
        auto const grid_colour = ImGui::GetColorU32(ImGuiCol_Border);

        auto const step = nice_step(span);
        for (auto tick = 0.0F; tick <= span; tick += step) {
            auto const x = origin.x + label_width + (tick * scale);
            draw->AddLine(ImVec2{x, origin.y + ruler_height - 4.0F}, ImVec2{x, origin.y + height}, grid_colour);
            auto const label = std::format("{:g} ms", static_cast<double>(tick));
            draw->AddText(ImVec2{x + 3.0F, origin.y}, text_colour, label.c_str());
        }

        for (auto lane = std::size_t{0}; lane < frame_graph::logical_queue_count; ++lane) {
            auto const queue = lane == 0 ? frame_graph::LogicalQueue::graphics : frame_graph::LogicalQueue::compute;
            auto const top = origin.y + ruler_height + (static_cast<float>(lane) * (lane_height + lane_gap));
            draw->AddText(ImVec2{origin.x, top + 10.0F}, text_colour, lane_label(queue));
            draw->AddRectFilled(ImVec2{origin.x + label_width, top}, ImVec2{origin.x + width, top + lane_height},
                                ImGui::GetColorU32(ImGuiCol_FrameBg));
        }

        auto const mouse = ImGui::GetMousePos();
        for (auto const &timing: timings) {
            if (!timing.milliseconds || !timing.start_milliseconds) {
                continue;
            }

            auto const lane = static_cast<float>(static_cast<std::size_t>(timing.queue));
            auto const top = origin.y + ruler_height + (lane * (lane_height + lane_gap));
            auto const left = origin.x + label_width + (*timing.start_milliseconds * scale);
            auto const right = left + std::max(*timing.milliseconds * scale, 1.0F);
            auto const min = ImVec2{left, top + 2.0F};
            auto const max = ImVec2{right, top + lane_height - 2.0F};

            draw->AddRectFilled(min, max, bar_colour(timing.queue, timing.name_id), 3.0F);

            auto const label_size = ImGui::CalcTextSize(timing.label.c_str());
            if (label_size.x + 6.0F < right - left) {
                draw->AddText(ImVec2{left + 3.0F, top + ((lane_height - label_size.y) * 0.5F)},
                              ImGui::GetColorU32(ImVec4{0.05F, 0.05F, 0.08F, 1.0F}), timing.label.c_str());
            }

            if (mouse.x >= min.x && mouse.x <= max.x && mouse.y >= min.y && mouse.y <= max.y) {
                draw->AddRect(min, max, ImGui::GetColorU32(ImVec4{1.0F, 1.0F, 1.0F, 0.9F}), 3.0F, 0, 2.0F);
                ImGui::SetTooltip("%s\n%.3f ms, starts at %.3f ms\n%s queue", timing.label.c_str(),
                                  static_cast<double>(*timing.milliseconds),
                                  static_cast<double>(*timing.start_milliseconds), lane_label(timing.queue));
            }
        }
    }

}
