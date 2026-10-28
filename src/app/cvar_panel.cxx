#include "app/application.hxx"

#include <algorithm>
#include <cctype>
#include <string>

#include "core/cvar.hxx"

namespace {

    [[nodiscard]] auto contains_ignoring_case(std::string_view text, std::string_view needle) -> bool {
        auto const match = std::ranges::search(text, needle, [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right));
        });

        return !match.empty();
    }

}

// Every CVar with its value, default and where the value came from. A change applies as a run-time set; a setting
// that is only read at startup (the window mode, say) shows the new value but takes effect on the next run.
auto Application::draw_cvar_panel() -> void {
    ImGui::SetNextItemWidth(-1.0F);
    ImGui::InputTextWithHint("##cvar_filter", "Filter", cvar_filter.data(), cvar_filter.size());

    if (!cvar_status.empty()) {
        ImGui::TextColored(ImVec4{1.0F, 0.5F, 0.4F, 1.0F}, "%s", cvar_status.c_str());
    }

    constexpr auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable |
                           ImGuiTableFlags_ScrollY;

    if (!ImGui::BeginTable("##cvars", 4, flags)) {
        return;
    }

    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 1.2F);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 1.0F);
    ImGui::TableSetupColumn("Default", ImGuiTableColumnFlags_WidthStretch, 0.7F);
    ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 0.7F);
    ImGui::TableHeadersRow();

    std::string_view const filter{cvar_filter.data()};
    auto cvars = CVarRegistry::global().all();

    std::ranges::sort(cvars, [](CVarBase const *left, CVarBase const *right) { return left->name() < right->name(); });

    for (auto *const cvar: cvars) {
        if (!filter.empty() && !contains_ignoring_case(cvar->name(), filter)) {
            continue;
        }

        ImGui::PushID(cvar);
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::TextUnformatted(std::string{cvar->name()}.c_str());
        if (ImGui::BeginItemTooltip()) {
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0F);
            ImGui::TextWrapped("%s", cvar->help().c_str());
            ImGui::TextDisabled("%s=<%s>   %s", cvar->cli_name().c_str(), cvar->hint().c_str(),
                                cvar->cli_only() ? "command line only" : cvar->env_name().c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }

        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1.0F);

        if (cvar->is_list()) {
            ImGui::TextDisabled("%s", cvar->text().c_str());
        } else if (cvar->is_flag()) {
            auto on = cvar->text() == "on";

            if (ImGui::Checkbox("##value", &on)) {
                if (auto const set = CVarRegistry::global().set(cvar->name(), on ? "on" : "off"); !set) {
                    cvar_status = std::format("{}: {}", cvar->name(), set.error());
                } else {
                    cvar_status.clear();
                }
            }
        } else {
            std::array<char, 256> buffer{};
            auto const current = cvar->text();

            std::ranges::copy_n(current.begin(), static_cast<std::ptrdiff_t>(std::min(current.size(), buffer.size() - 1)),
                                buffer.begin());

            if (ImGui::InputText("##value", buffer.data(), buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue)) {
                if (auto const set = CVarRegistry::global().set(cvar->name(), buffer.data()); !set) {
                    cvar_status = std::format("{}: {}", cvar->name(), set.error());
                } else {
                    cvar_status.clear();
                }
            }
        }

        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", cvar->default_text().c_str());

        ImGui::TableNextColumn();
        ImGui::TextUnformatted(std::string{to_string(cvar->source())}.c_str());

        ImGui::PopID();
    }

    ImGui::EndTable();
}
