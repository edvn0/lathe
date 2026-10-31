#include "app/gpu_timings_view.hxx"

#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <imgui.h>

namespace gui {
    namespace {

        constexpr auto bar_height = 14.0F;

        auto lane_name(std::uint8_t lane) -> char const * {
            switch (lane) {
                case 0: return "graphics";
                case 1: return "compute";
                default: return "overlay";
            }
        }

        auto series_colour(std::size_t index) -> ImVec4 {
            constexpr auto rgb = std::array<std::uint32_t, 8>{0x3987e5, 0xd95926, 0x199e70, 0xc98500,
                                                              0xd55181, 0x008300, 0x9085e9, 0xe66767};
            auto const c = rgb[index % rgb.size()];
            return {static_cast<float>((c >> 16U) & 0xffU) / 255.0F, static_cast<float>((c >> 8U) & 0xffU) / 255.0F,
                    static_cast<float>(c & 0xffU) / 255.0F, 1.0F};
        }

        auto cell_text(char const *text, bool dim) -> void {
            if (dim) {
                ImGui::TextDisabled("%s", text);
            } else {
                ImGui::TextUnformatted(text);
            }
        }

        auto tooltip(GpuTimings::Entry const &entry) -> void {
            if (!ImGui::BeginTooltip()) {
                return;
            }
            ImGui::TextUnformatted(entry.label.c_str());
            ImGui::TextDisabled("%s (%s)", entry.id.c_str(), lane_name(entry.lane));
            ImGui::Separator();
            ImGui::Text("average  %.4f ms", static_cast<double>(entry.ema_ms));
            ImGui::Text("last     %.4f ms", static_cast<double>(entry.last_ms));
            ImGui::Text("min      %.4f ms", static_cast<double>(entry.min_ms()));
            ImGui::Text("max      %.4f ms", static_cast<double>(entry.max_ms()));
            ImGui::EndTooltip();
        }

    }

    auto draw_gpu_timings(GpuTimings const &live, float frame_ms, GpuTimingsViewState &state) -> void {
        auto const &timings = state.paused ? state.frozen : live;
        auto const shown_frame_ms = state.paused ? state.frozen_frame_ms : frame_ms;

        if (ImGui::Button(state.paused ? "Resume" : "Pause")) {
            state.paused = !state.paused;
            if (state.paused) {
                state.frozen = live;
                state.frozen_frame_ms = frame_ms;
            }
        }
        if (shown_frame_ms > 0.0F) {
            ImGui::SameLine();
            ImGui::TextDisabled("GPU frame %.2f ms (%.0f fps)", static_cast<double>(shown_frame_ms),
                                static_cast<double>(1000.0F / shown_frame_ms));
        }

        ImGui::Spacing();

        if (timings.entries().empty()) {
            ImGui::TextDisabled("Waiting for GPU timings...");
            return;
        }

        constexpr auto flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit |
                               ImGuiTableFlags_ScrollY;
        if (!ImGui::BeginTable("##gpu_timings", 3, flags)) {
            return;
        }

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("GPU pass");
        ImGui::TableSetupColumn("ms (EMA)");
        ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto const total = timings.total_ms();
        auto const entries = timings.entries();

        for (auto index = std::size_t{0}; index < entries.size(); ++index) {
            auto const &entry = entries[index];
            auto const idle = !entry.active;
            char text[32];

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            cell_text(entry.label.c_str(), idle);
            auto const hovered = ImGui::IsItemHovered();

            ImGui::TableSetColumnIndex(1);
            if (idle) {
                std::snprintf(text, sizeof text, "-");
            } else {
                std::snprintf(text, sizeof text, "%.3f", static_cast<double>(entry.ema_ms));
            }
            cell_text(text, idle);

            ImGui::TableSetColumnIndex(2);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, series_colour(index));
            ImGui::ProgressBar(!idle && total > 0.0F ? entry.ema_ms / total : 0.0F, ImVec2{-FLT_MIN, bar_height}, "");
            ImGui::PopStyleColor();

            if (hovered && !idle) {
                tooltip(entry);
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Total");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", static_cast<double>(total));

        ImGui::EndTable();
    }

}
