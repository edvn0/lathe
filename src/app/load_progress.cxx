// A persistent card over the viewport while models parse and their textures stream in. The streamers only report how
// many requests are pending, not how many there were, so the texture bar is measured against the most that were
// pending at once since the work began; a model's parse has no progress to report and shows a moving bar instead.

#include "app/application.hxx"

#include <algorithm>
#include <cstddef>
#include <format>
#include <string>

#include <imgui.h>

#include "rendering/renderer.hxx"
#include "rendering/toast.hxx"

namespace {
    // Loads shorter than this never show the card, so a quick one does not flash it.
    constexpr auto min_visible_seconds = 0.3F;
    constexpr auto card_margin = 12.0F;
    constexpr auto bar_width = 260.0F;
}

auto Application::draw_load_progress() -> void {
    auto const textures = renderer->texture_streamer().pending_count();
    auto const models = renderer->model_streamer().pending_count();

    if (textures == 0 && models == 0) {
        // Only a load long enough to have shown the card announces that it finished.
        if (load_progress.busy_seconds >= min_visible_seconds) {
            gui::toast_success("Loading complete", "All models and textures are loaded.");
        }
        load_progress = {};
        return;
    }

    load_progress.busy_seconds += ImGui::GetIO().DeltaTime;
    load_progress.peak_textures = std::max(load_progress.peak_textures, textures);

    auto const &pane = viewports[0];
    if (load_progress.busy_seconds < min_visible_seconds || pane.content_size.x <= 0.0F || pane.content_size.y <= 0.0F) {
        return;
    }

    // The bottom-left of the viewport; the toasts stack in the bottom-right.
    ImGui::SetNextWindowPos(ImVec2{pane.screen_pos.x + card_margin, pane.screen_pos.y + pane.content_size.y - card_margin},
                            ImGuiCond_Always, ImVec2{0.0F, 1.0F});
    ImGui::SetNextWindowBgAlpha(0.88F);

    constexpr auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                           ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoDocking;

    if (ImGui::Begin("##load_progress", nullptr, flags)) {
        ImGui::TextDisabled("Loading");

        if (models > 0) {
            auto name = std::string{};
            auto unsettled = std::size_t{0};
            for (auto const &load: model_loads) {
                if (!load.settled) {
                    name = load.file_name;
                    ++unsettled;
                }
            }

            ImGui::TextUnformatted(unsettled == 1  ? name.c_str()
                                   : models == 1 ? "1 model"
                                                 : std::format("{} models", models).c_str());
            // A negative fraction is ImGui's moving bar, for work with no progress to show.
            ImGui::ProgressBar(-1.0F * static_cast<float>(ImGui::GetTime()), ImVec2{bar_width, 0.0F}, "Reading");
        }

        if (textures > 0) {
            auto const peak = std::max(load_progress.peak_textures, textures);
            auto const done = peak - textures;

            ImGui::TextUnformatted("Textures");
            ImGui::ProgressBar(static_cast<float>(done) / static_cast<float>(peak), ImVec2{bar_width, 0.0F},
                               std::format("{} / {}", done, peak).c_str());
        }
    }
    ImGui::End();
}
