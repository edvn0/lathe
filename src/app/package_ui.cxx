#include "app/application.hxx"

#include <algorithm>
#include <cstring>
#include <format>
#include <string>
#include <utility>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/renderer.hxx"

namespace {

    constexpr auto package_dialog = "Package game";
    constexpr auto progress_dialog = "Packaging game";
    constexpr auto package_scene_file = "scenes/main.lbf";

    template<std::size_t N>
    auto fill(std::array<char, N> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), N - 1);
        std::copy_n(text.begin(), length, buffer.begin());
        buffer[length] = '\0';
    }

    template<std::size_t N>
    [[nodiscard]] auto text_of(std::array<char, N> const &buffer) -> std::string {
        return std::string{buffer.data()};
    }

    [[nodiscard]] auto valid_name(std::string_view name) -> bool {
        return !name.empty() && name != "." && name != ".." && name.find_first_of("/\\") == std::string_view::npos;
    }

}

auto Application::request_package_dialog() -> void {
    if (package_form.name[0] == '\0') {
        fill(package_form.name, game_name.empty() ? "game" : game_name);
        fill(package_form.title, game_name.empty() ? "Lathe Game" : game_name);
        fill(package_form.version, "0.1.0");
        fill(package_form.output, "packages");
    }

    package_popup_requested = true;
}

auto Application::start_package() -> void {
    if (packaging_active() || scene_save_job.has_value() || scene_load_job.has_value()) {
        return;
    }

    if (is_playing) {
        stop();
    }

    auto const name = text_of(package_form.name);
    auto const output_dir = std::filesystem::absolute(gui::utf8_to_path(text_of(package_form.output)));
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);

    if (error) {
        package_status = std::format("Could not create '{}': {}", output_dir.string(), error.message());
        return;
    }

    auto const executable = std::filesystem::read_symlink("/proc/self/exe", error);

    if (error) {
        package_status = "Could not locate the running executable.";
        return;
    }

    PackageOptions options;
    options.manifest = GameManifest{.name = name,
                                    .title = text_of(package_form.title),
                                    .game = game_name,
                                    .version = text_of(package_form.version),
                                    .entry = script_entry,
                                    .scene = package_scene_file};
    options.output_dir = output_dir;
    options.data_root = Paths::current().data_root();
    options.executable = executable;
    options.compiler = &Renderer::compiler();
    options.archive = package_form.archive;

    package_staged_scene = output_dir / (name + ".scene.tmp.lbf");
    options.cooked_scene = package_staged_scene;

    SceneSaveOptions save_options;

    if (scene_pack != nullptr) {
        save_options.source_packs.push_back(scene_pack);
    }

    package_pending = std::move(options);
    package_status = "Cooking the scene...";
    package_scene_job = SceneSaveJob::start(*editor_scene, *renderer, engine_models, package_staged_scene,
                                            std::move(save_options));
    info("Packaging '{}' into '{}'", name, output_dir.string());
}

auto Application::update_package_jobs() -> void {
    if (package_scene_job.has_value() && package_scene_job->ready()) {
        auto saved = package_scene_job->take();
        package_scene_job.reset();

        if (!saved) {
            error("Cooking the scene for packaging failed: {}", describe(saved.error()));
            package_status = "Packaging failed while cooking the scene; see the console.";
            package_pending.reset();
        } else if (package_pending.has_value()) {
            package_status = "Packaging...";
            package_job = GamePackageJob::start(*std::move(package_pending));
            package_pending.reset();
        }
    }

    if (package_job.has_value() && package_job->ready()) {
        auto result = package_job->take();
        package_job.reset();

        if (result) {
            auto const location = result->archive.value_or(result->directory);
            package_status = std::format("Packaged {} files ({:.1f} MiB) in {:.1f}s: {}", result->files,
                                         static_cast<double>(result->bytes) / (1024.0 * 1024.0), result->seconds,
                                         location.string());
            info("{}", package_status);
        } else if (result.error() == "cancelled") {
            package_status = "Packaging cancelled.";
        } else {
            package_status = std::format("Packaging failed: {}", result.error());
            error("{}", package_status);
        }

        std::error_code ignored;
        std::filesystem::remove(package_staged_scene, ignored);
    }
}

auto Application::draw_package_ui() -> void {
    if (package_popup_requested) {
        ImGui::OpenPopup(package_dialog);
        package_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(package_dialog, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Builds a self-contained game folder: this executable, the cooked scene, assets and "
                           "precompiled shaders. The C++ game code is the one already built into this editor.");
        ImGui::Separator();

        ImGui::InputText("Folder name", package_form.name.data(), package_form.name.size());
        ImGui::InputText("Window title", package_form.title.data(), package_form.title.size());
        ImGui::InputText("Version", package_form.version.data(), package_form.version.size());
        ImGui::InputText("Output directory", package_form.output.data(), package_form.output.size());
        ImGui::Checkbox("Also write a .tar archive", &package_form.archive);

        auto const name = text_of(package_form.name);
        auto const valid = valid_name(name) && package_form.output[0] != '\0';

        if (!game_name.empty() && game_name == "lua" && script_entry.empty()) {
            ImGui::TextColored(ImVec4{1.0F, 0.75F, 0.3F, 1.0F}, "This is a Lua game but no --script is set.");
        }

        if (valid) {
            std::error_code exists_error;
            auto const target = gui::utf8_to_path(text_of(package_form.output)) / name;

            if (std::filesystem::exists(target, exists_error)) {
                ImGui::TextColored(ImVec4{1.0F, 0.75F, 0.3F, 1.0F}, "'%s' exists and will be replaced.",
                                   target.string().c_str());
            }
        }

        ImGui::Separator();

        ImGui::BeginDisabled(!valid);
        if (ImGui::Button("Package")) {
            start_package();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    if (packaging_active() && !package_progress_popup_open) {
        ImGui::OpenPopup(progress_dialog);
        package_progress_popup_open = true;
    }

    ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, ImVec4{0.0F, 0.0F, 0.0F, 0.65F});
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2{0.5F, 0.5F});
    ImGui::SetNextWindowSize(ImVec2{420.0F, 0.0F});

    if (ImGui::BeginPopupModal(progress_dialog, nullptr,
                               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
        if (!packaging_active()) {
            package_progress_popup_open = false;
            ImGui::CloseCurrentPopup();
        } else if (package_scene_job.has_value()) {
            ImGui::TextUnformatted("Cooking the scene");
            ImGui::ProgressBar(-1.0F * static_cast<float>(ImGui::GetTime()), ImVec2{-1.0F, 0.0F});
        } else {
            auto const progress = package_job->progress();
            ImGui::TextUnformatted(progress.step.c_str());
            ImGui::ProgressBar(progress.fraction, ImVec2{-1.0F, 0.0F});

            if (ImGui::Button("Cancel")) {
                package_job->cancel();
            }
        }

        ImGui::EndPopup();
    } else if (!packaging_active()) {
        package_progress_popup_open = false;
    }

    ImGui::PopStyleColor();
}
