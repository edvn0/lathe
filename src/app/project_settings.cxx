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

auto Application::project_form_dirty() const -> bool {
    auto const &form = project_form;

    if (!project || form.root != project->root) {
        return false;
    }

    auto const &manifest = project->manifest;

    return text_of(form.name) != manifest.name || text_of(form.title) != manifest.title ||
           text_of(form.version) != manifest.version || text_of(form.game) != manifest.game ||
           text_of(form.entry) != manifest.entry || text_of(form.scene) != manifest.scene || form.gi != manifest.gi ||
           form.diagnostics != manifest.diagnostics;
}

auto Application::save_project_form() -> bool {
    auto &form = project_form;

    if (!project || form.root != project->root) {
        return false;
    }

    auto updated = *project;

    updated.manifest.name = text_of(form.name);
    updated.manifest.title = text_of(form.title);
    updated.manifest.version = text_of(form.version);
    updated.manifest.game = text_of(form.game);
    updated.manifest.entry = text_of(form.entry);
    updated.manifest.scene = text_of(form.scene);
    updated.manifest.gi = form.gi;
    updated.manifest.diagnostics = form.diagnostics;

    auto const saved = updated.save();

    if (!saved) {
        form.status = saved.error();
        return false;
    }

    project = std::move(updated);
    form.status = "Saved. The game, script and global illumination apply the next time the project opens.";

    auto const &title = project->manifest.title.empty() ? project->manifest.name : project->manifest.title;
    glfwSetWindowTitle(context.window, title.c_str());

    return true;
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
        form.gi = manifest.gi;
        form.diagnostics = manifest.diagnostics;
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

    if (ImGui::CollapsingHeader("Global illumination")) {
        auto &gi = form.gi;
        ImGui::Checkbox("Enabled##gi", &gi.enabled);
        ImGui::SetItemTooltip("Probe GI lights the scene with its emissive surfaces, bounce light and the sky. Needs a device "
                              "with VK_KHR_ray_query; the setting applies the next time the project opens.");
        ImGui::BeginDisabled(!gi.enabled);
        auto rays = static_cast<int>(gi.rays_per_probe / 64U);
        if (ImGui::SliderInt("Rays per probe", &rays, 1, 4, "%d x 64")) {
            gi.rays_per_probe = static_cast<std::uint32_t>(rays) * 64U;
        }
        ImGui::DragFloat("Probe spacing (m)", &gi.probe_spacing, 0.05F, 0.1F, 16.0F, "%.2f");
        ImGui::SetItemTooltip("Distance between the probes of the near cascade; the coarse cascade is four times that.");
        ImGui::DragFloat("Intensity", &gi.intensity, 0.01F, 0.0F, 16.0F, "%.2f");
        ImGui::DragFloat("Emitter intensity", &gi.emitter_intensity, 1.0F, 0.0F, 100000.0F, "%.0f");
        ImGui::SetItemTooltip("How strongly small emissive pieces (bulbs, lantern glass) light their surroundings.");
        ImGui::Checkbox("Far cascade", &gi.far_cascade);
        constexpr std::array<char const *, 3> choices{"auto", "on", "off"};
        auto current = 0;
        for (auto i = 0; i < 3; ++i) {
            if (gi.synthesize_lights == choices[static_cast<std::size_t>(i)]) {
                current = i;
            }
        }
        if (ImGui::Combo("Lights from emissive meshes", &current, "auto\0on\0off\0")) {
            gi.synthesize_lights = choices[static_cast<std::size_t>(current)];
        }
        ImGui::SetItemTooltip("The point lights a model gets from its emissive meshes when it loads. auto: none while the "
                              "probes light them themselves.");
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Diagnostics")) {
        constexpr std::array<char const *, 4> choices{"auto", "standard", "full", "off"};
        auto current = 0;
        for (auto i = 0; i < 4; ++i) {
            if (form.diagnostics.gpu_crash_dumps == choices[static_cast<std::size_t>(i)]) {
                current = i;
            }
        }
        if (ImGui::Combo("GPU crash dumps", &current, "auto\0standard\0full\0off\0")) {
            form.diagnostics.gpu_crash_dumps = choices[static_cast<std::size_t>(current)];
        }
        ImGui::SetItemTooltip("When the GPU faults, the driver writes a crash dump (Nsight Aftermath, docs/gpu-crash-dumps.md). "
                              "auto: the cheap collection, and everything after a device-lost restart. Applies the next time "
                              "the project opens.");
    }

    ImGui::Spacing();

    if (ImGui::Button("Save")) {
        static_cast<void>(save_project_form());
    }

    ImGui::SameLine();

    if (ImGui::Button("Package Game...")) {
        request_package_dialog();
    }

    if (!form.status.empty()) {
        ImGui::TextWrapped("%s", form.status.c_str());
    }
}
