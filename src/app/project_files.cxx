#include "app/application.hxx"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

#include "core/logger.hxx"
#include "core/project_file.hxx"
#include "rendering/imgui_widget.hxx"

namespace {

    constexpr auto project_switch_popup = "Open project";
    constexpr auto new_project_popup = "New project";

    template<std::size_t N>
    auto fill_buffer(std::array<char, N> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), N - 1);
        std::copy_n(text.begin(), length, buffer.begin());
        buffer[length] = '\0';
    }

}

auto Application::request_open_project(std::filesystem::path root) -> void {
    if (scene_load_job.has_value() || scene_save_job.has_value()) {
        scene_status = "A scene is loading or saving; wait for it to finish.";
        return;
    }

    if (source_scene_dirty()) {
        pending_project_open = std::move(root);
        project_switch_popup_requested = true;
        return;
    }

    switch_project(std::move(root));
}

auto Application::request_new_project_dialog() -> void {
    if (new_project_directory[0] == '\0') {
        fill_buffer(new_project_directory, gui::path_to_utf8(std::filesystem::current_path().parent_path() / "my_project"));
        fill_buffer(new_project_name, "My Project");
    }

    new_project_status.clear();
    new_project_popup_requested = true;
}

auto Application::create_project() -> bool {
    auto const root = std::filesystem::absolute(gui::utf8_to_path(new_project_directory.data())).lexically_normal();
    std::string const name{new_project_name.data()};

    if (name.empty()) {
        new_project_status = "Give the project a name.";
        return false;
    }

    std::error_code error;

    if (std::filesystem::exists(root / ProjectFile::file_name, error)) {
        new_project_status = std::format("'{}' already holds a project; open it instead.", root.string());
        return false;
    }

    std::filesystem::create_directories(root / "assets", error);

    if (error) {
        new_project_status = std::format("Could not create '{}': {}", root.string(), error.message());
        return false;
    }

    ProjectFile const created{.manifest = {.name = name, .title = name, .game = "empty", .version = "0.1.0"}, .root = root};

    if (auto const saved = created.save(); !saved) {
        new_project_status = saved.error();
        return false;
    }

    switch_project(root);

    return true;
}

auto Application::draw_project_ui() -> void {
    if (auto const picked = project_browser.draw(editor_icons.get())) {
        request_open_project(picked->parent_path());
    }

    if (project_switch_popup_requested) {
        ImGui::OpenPopup(project_switch_popup);
        project_switch_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(project_switch_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto const target = pending_project_open.has_value() ? gui::path_to_utf8(pending_project_open->filename()) : "";

        ImGui::Text("The scene has unsaved changes. Opening '%s' replaces it.", target.c_str());

        if (ImGui::Button("Discard changes and open")) {
            if (pending_project_open.has_value()) {
                switch_project(std::move(*pending_project_open));
            }
            pending_project_open.reset();
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            pending_project_open.reset();
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    if (new_project_popup_requested) {
        ImGui::OpenPopup(new_project_popup);
        new_project_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(new_project_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Name", new_project_name.data(), new_project_name.size());
        ImGui::InputText("Directory", new_project_directory.data(), new_project_directory.size());

        if (!new_project_status.empty()) {
            ImGui::TextColored(ImVec4{1.0F, 0.5F, 0.4F, 1.0F}, "%s", new_project_status.c_str());
        }

        // A new project replaces the open scene the same way opening one does.
        ImGui::BeginDisabled(source_scene_dirty());
        if (ImGui::Button("Create")) {
            if (create_project()) {
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndDisabled();

        if (source_scene_dirty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Save or discard the scene's changes first.");
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}
