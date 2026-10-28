#include "app/application.hxx"

#include <algorithm>
#include <cstring>
#include <format>
#include <string>

#include <GLFW/glfw3.h>

#include "core/project_file.hxx"
#include "rendering/imgui_widget.hxx"

namespace {

    template<std::size_t N>
    auto fill_buffer(std::array<char, N> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), N - 1);
        std::copy_n(text.begin(), length, buffer.begin());
        buffer[length] = '\0';
    }

    template<std::size_t N>
    [[nodiscard]] auto text_of(std::array<char, N> const &buffer) -> std::string {
        return std::string{buffer.data()};
    }

}

auto Application::draw_project_settings_page() -> void {
    if (!project) {
        ImGui::TextWrapped("No project is open. A project holds your scenes, scripts and assets, and the game that "
                           "starts with it; the editor's own shaders and built-in assets stay available.");

        if (ImGui::Button("New Project...")) {
            request_new_project_dialog();
        }

        ImGui::SameLine();

        if (ImGui::Button("Open Project...")) {
            project_browser.open("Open Project", {{.label = "Lathe projects (lathe.project)", .extensions = {".project"}}});
        }

        return;
    }

    auto &form = project_form;

    if (form.root != project->root) {
        auto const &manifest = project->manifest;

        form.root = project->root;
        form.status.clear();
        fill_buffer(form.name, manifest.name);
        fill_buffer(form.title, manifest.title);
        fill_buffer(form.version, manifest.version);
        fill_buffer(form.game, manifest.game);
        fill_buffer(form.entry, manifest.entry);
        fill_buffer(form.scene, manifest.scene);
    }

    ImGui::TextDisabled("%s", gui::path_to_utf8(project->root / ProjectFile::file_name).c_str());
    ImGui::Spacing();

    ImGui::InputText("Name", form.name.data(), form.name.size());
    ImGui::InputText("Title", form.title.data(), form.title.size());
    ImGui::InputText("Version", form.version.data(), form.version.size());

    if (ImGui::BeginCombo("Game", form.game[0] == '\0' ? "(default)" : form.game.data())) {
        if (ImGui::Selectable("(default)", form.game[0] == '\0')) {
            fill_buffer(form.game, "");
        }

        for (auto const name: available_games) {
            if (ImGui::Selectable(std::string{name}.c_str(), name == std::string_view{form.game.data()})) {
                fill_buffer(form.game, name);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::InputText("Script", form.entry.data(), form.entry.size());
    ImGui::SetItemTooltip("The Lua script a scripted game starts with, relative to the project.");

    ImGui::InputText("Main scene", form.scene.data(), form.scene.size());
    ImGui::SetItemTooltip("The scene (.lbf) opened with the project, relative to the project.");

    if (!scene_path.empty()) {
        ImGui::SameLine();

        if (ImGui::Button("Use open scene")) {
            if (auto const relative = scene_path.lexically_normal().lexically_relative(project->root);
                !relative.empty() && *relative.begin() != "..") {
                fill_buffer(form.scene, relative.generic_string());
            } else {
                form.status = "The open scene is outside the project; save it inside the project first.";
            }
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Save")) {
        auto updated = *project;

        updated.manifest.name = text_of(form.name);
        updated.manifest.title = text_of(form.title);
        updated.manifest.version = text_of(form.version);
        updated.manifest.game = text_of(form.game);
        updated.manifest.entry = text_of(form.entry);
        updated.manifest.scene = text_of(form.scene);

        if (auto const saved = updated.save()) {
            project = std::move(updated);
            form.status = "Saved. The game and script apply the next time the project opens.";

            auto const &title = project->manifest.title.empty() ? project->manifest.name : project->manifest.title;
            glfwSetWindowTitle(context.window, title.c_str());
        } else {
            form.status = saved.error();
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Package Game...")) {
        request_package_dialog();
    }

    if (!form.status.empty()) {
        ImGui::TextWrapped("%s", form.status.c_str());
    }
}
