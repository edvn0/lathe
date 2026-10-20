#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <imgui.h>

namespace ui {

    // How many assets a picker lists before the filter narrows it down.
    inline constexpr auto picker_unfiltered_limit = std::size_t{40};

    // The body of a picker popup for named assets (`entries` hold a `name` and a `handle`, as NamedAssetTable's do):
    // those in `used` first, then the rest, narrowed by a filter. Assets without a name are left out, as they have
    // nothing to show and ImGui rejects an item without a label. Returns the asset clicked, if any. Call it inside
    // an open popup.
    template<typename Entry, typename Handle>
    auto draw_asset_choices(std::span<Entry const> entries, std::span<Handle const> used, Handle const current)
            -> std::optional<Handle> {
        static ImGuiTextFilter filter;

        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }
        filter.Draw("##asset_filter", ImGui::GetFontSize() * 14.0F);

        auto const is_used = [&](Handle handle) { return std::ranges::find(used, handle) != used.end(); };
        auto const listed = [&](Entry const &entry) {
            return !entry.name.empty() && entry.handle.valid() && filter.PassFilter(entry.name.c_str());
        };

        auto picked = std::optional<Handle>{};
        auto const draw = [&](Entry const &entry, std::size_t index) {
            // The index keeps equal names (or names that look like "##id" syntax) apart.
            ImGui::PushID(static_cast<int>(index));
            if (ImGui::Selectable(entry.name.c_str(), entry.handle == current)) {
                picked = entry.handle;
            }
            ImGui::PopID();
        };

        auto any_used = false;
        for (auto index = std::size_t{0}; index < entries.size(); ++index) {
            if (listed(entries[index]) && is_used(entries[index].handle)) {
                if (!std::exchange(any_used, true)) {
                    ImGui::SeparatorText("In this scene");
                }
                draw(entries[index], index);
            }
        }

        ImGui::SeparatorText("Loaded");
        auto shown = std::size_t{0};
        auto hidden = std::size_t{0};
        for (auto index = std::size_t{0}; index < entries.size(); ++index) {
            if (!listed(entries[index]) || is_used(entries[index].handle)) {
                continue;
            }
            if (!filter.IsActive() && shown >= picker_unfiltered_limit) {
                ++hidden;
                continue;
            }
            draw(entries[index], index);
            ++shown;
        }
        if (hidden > 0) {
            ImGui::TextDisabled("%zu more -- type to filter", hidden);
        }

        return picked;
    }

}
