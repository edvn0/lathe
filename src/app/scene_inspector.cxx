#include "app/scene_inspector.hxx"

#include <algorithm>
#include <cstdio>
#include <format>

#include <imgui.h>

#include "core/error_describe.hxx"
#include "rendering/imgui_widget.hxx"
#include "serialisation/lbf_container.hxx"

namespace {

    [[nodiscard]] auto human_size(std::uint64_t bytes) -> std::string {
        if (bytes < 1024) {
            return std::format("{} B", bytes);
        }

        auto value = static_cast<double>(bytes) / 1024.0;
        char const *unit = "KiB";

        if (value >= 1024.0) {
            value /= 1024.0;
            unit = "MiB";
        }

        return std::format("{:.1f} {}", value, unit);
    }

    constexpr ImVec4 good{0.45F, 0.85F, 0.45F, 1.0F};
    constexpr ImVec4 warn{0.95F, 0.75F, 0.25F, 1.0F};
    constexpr ImVec4 bad{0.95F, 0.35F, 0.35F, 1.0F};

    auto status_text(InspectedSection const &section) -> void {
        switch (section.status) {
            case InspectedSectionStatus::current:
                ImGui::TextColored(good, "current");
                break;
            case InspectedSectionStatus::older:
                ImGui::TextColored(warn, "older (reads fine)");
                break;
            case InspectedSectionStatus::too_old:
                ImGui::TextColored(bad, "too old");
                break;
            case InspectedSectionStatus::newer:
                ImGui::TextColored(warn, "newer (skipped)");
                break;
            case InspectedSectionStatus::unknown:
                ImGui::TextColored(warn, "unknown (skipped)");
                break;
        }
    }

} // namespace

namespace gui {

    auto SceneInspector::inspect(std::filesystem::path const &path) -> void {
        result_ = inspect_scene_file(path);

        auto const text = path_to_utf8(path);
        auto const length = std::min(text.size(), path_buffer_.size() - 1);
        std::copy_n(text.begin(), length, path_buffer_.begin());
        path_buffer_[length] = '\0';
    }

    auto SceneInspector::draw(EditorIcons const *icons, std::filesystem::path const &current_scene, bool &open) -> void {
        panel("Scene Inspector", open, [&] {
            ImGui::SetNextItemWidth(-1.0F);
            if (ImGui::InputText("##path", path_buffer_.data(), path_buffer_.size(),
                                 ImGuiInputTextFlags_EnterReturnsTrue)) {
                inspect(utf8_to_path(path_buffer_.data()));
            }

            ImGui::BeginDisabled(browser_.is_open());
            if (ImGui::Button("Browse...")) {
                browser_.open("Inspect Scene", {{.label = "Lathe scenes (*.lbf)", .extensions = {".lbf"}},
                                                {.label = "All files", .extensions = {}}});
            }
            ImGui::EndDisabled();

            ImGui::SameLine();
            ImGui::BeginDisabled(current_scene.empty());
            if (ImGui::Button("Inspect open scene")) {
                inspect(current_scene);
            }
            ImGui::EndDisabled();

            ImGui::SameLine();
            ImGui::BeginDisabled(path_buffer_[0] == '\0');
            if (ImGui::Button("Reload")) {
                inspect(utf8_to_path(path_buffer_.data()));
            }
            ImGui::EndDisabled();

            ImGui::Separator();

            if (!result_.has_value()) {
                ImGui::TextDisabled("Pick a .lbf file to look inside it.");
            } else if (!result_->has_value()) {
                ImGui::TextColored(bad, "Cannot read file: %s [%s]",
                                   describe(result_->error().cause.value_or(ErrorCause{})).c_str(),
                                   std::format("{}", result_->error().type).c_str());
            } else {
                draw_info(**result_);
            }
        });

        if (auto const picked = browser_.draw(icons)) {
            inspect(*picked);
        }
    }

    auto SceneInspector::draw_info(SceneFileInfo const &info) -> void {
        auto const &header = info.header;

        ImGui::Text("LBF %u.%u, %s, %s, %u chunks", static_cast<unsigned>(header.version_major),
                    static_cast<unsigned>(header.version_minor),
                    header.kind == LbfFileKind::scene ? "scene file" : "asset pack",
                    human_size(header.file_size).c_str(), header.chunk_count);

        if (header.kind != LbfFileKind::scene) {
            ImGui::TextColored(warn, "This is not a scene file; only its chunks are listed.");
        }

        if (ImGui::CollapsingHeader("Chunks", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::BeginTable("##chunks", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Type");
                ImGui::TableSetupColumn("Version");
                ImGui::TableSetupColumn("Id");
                ImGui::TableSetupColumn("Stored");
                ImGui::TableSetupColumn("Raw");
                ImGui::TableHeadersRow();

                for (auto const &chunk: info.chunks) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(chunk.type_name.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("v%u", static_cast<unsigned>(chunk.version));
                    ImGui::TableNextColumn();
                    ImGui::Text("%llx", static_cast<unsigned long long>(chunk.id));
                    ImGui::TableNextColumn();
                    ImGui::Text("%s%s", human_size(chunk.stored_size).c_str(), chunk.compressed ? " (zstd)" : "");
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(human_size(chunk.raw_size).c_str());
                }

                ImGui::EndTable();
            }
        }

        if (!info.scene_problem.empty()) {
            ImGui::TextColored(bad, "Scene chunk unreadable: %s", info.scene_problem.c_str());
        } else if (info.scene.has_value()) {
            draw_scene(*info.scene);
        } else if (header.kind == LbfFileKind::scene) {
            ImGui::TextColored(bad, "No SCEN chunk in this scene file.");
        }
    }

    auto SceneInspector::draw_scene(InspectedScene const &scene) -> void {
        auto const inline_count = std::ranges::count_if(scene.sections, [](auto const &s) { return s.inline_strings; });
        auto const skipped = std::ranges::count_if(scene.sections, [](auto const &s) {
            return s.status == InspectedSectionStatus::newer || s.status == InspectedSectionStatus::unknown;
        });

        ImGui::Text("Scene chunk v%u, %s, %zu sections", static_cast<unsigned>(scene.chunk_version),
                    human_size(scene.payload_size).c_str(), scene.sections.size());

        if (scene.decode_error.has_value()) {
            ImGui::TextColored(bad, "Does not decode: %s", scene.decode_error->c_str());
        } else {
            ImGui::TextColored(good, "Decodes cleanly%s", skipped > 0 ? " (some sections skipped)" : "");
        }

        if (!scene.layout_problem.empty()) {
            ImGui::TextColored(bad, "Layout: %s", scene.layout_problem.c_str());
        }

        if (scene.strings.has_value()) {
            ImGui::Text("Strings: shared table, %zu strings, %s%s", scene.strings->size(),
                        human_size(scene.strings_size).c_str(),
                        inline_count > 0 ? ", but some sections are still inline" : "");
        } else if (inline_count > 0) {
            ImGui::TextColored(warn, "Strings: stored inline (written before the strings section existed)");
        } else {
            ImGui::TextDisabled("Strings: no strings section");
        }

        if (ImGui::CollapsingHeader("Sections", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::BeginTable("##sections", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Section");
                ImGui::TableSetupColumn("Version");
                ImGui::TableSetupColumn("Size");
                ImGui::TableSetupColumn("Status");
                ImGui::TableSetupColumn("Strings");
                ImGui::TableHeadersRow();

                for (auto const &section: scene.sections) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s (%u)", section.name.c_str(), section.type);
                    ImGui::TableNextColumn();
                    if (section.current_version != 0) {
                        ImGui::Text("v%u / v%u", static_cast<unsigned>(section.version),
                                    static_cast<unsigned>(section.current_version));
                    } else {
                        ImGui::Text("v%u", static_cast<unsigned>(section.version));
                    }
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(human_size(section.size).c_str());
                    ImGui::TableNextColumn();
                    status_text(section);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(section.inline_strings ? "inline" : "-");
                }

                ImGui::EndTable();
            }

            ImGui::TextDisabled("Version column: in the file / written by this build.");
        }

        if (!scene.counts.empty() && ImGui::CollapsingHeader("Contents", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto const &[name, count]: scene.counts) {
                if (count > 0) {
                    ImGui::Text("%zu %s", count, name.c_str());
                }
            }
        }

        if (scene.strings.has_value() && ImGui::CollapsingHeader("Strings table", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto filter = string_filter_;
            filter.resize(std::max<std::size_t>(filter.size(), 128));
            ImGui::SetNextItemWidth(-1.0F);
            if (ImGui::InputTextWithHint("##filter", "filter", filter.data(), filter.size())) {
                string_filter_ = filter.c_str();
            }

            if (ImGui::BeginChild("##strings", ImVec2{0.0F, 220.0F}, ImGuiChildFlags_Borders)) {
                for (auto const &text: *scene.strings) {
                    if (string_filter_.empty() || text.find(string_filter_) != std::string::npos) {
                        ImGui::TextUnformatted(text.c_str());
                    }
                }
            }
            ImGui::EndChild();
        }
    }

} // namespace gui
