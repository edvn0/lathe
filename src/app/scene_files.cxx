// Application's scene file handling: opening and saving .lbf scenes in the background, the unsaved-changes prompt,
// and OS drag-and-drop.

#include "app/application.hxx"

#include <algorithm>
#include <cctype>
#include <format>
#include <string>
#include <utility>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/imgui_widget.hxx"
#include "scene/selection_context.hxx"

namespace {

    constexpr auto unsaved_changes_popup = "Unsaved changes";
    constexpr auto save_as_popup = "Save scene as";
    constexpr std::string_view default_scene_path = "scenes/untitled.lbf";

    [[nodiscard]] auto lowercase_extension(std::filesystem::path const &path) -> std::string {
        auto extension = gui::path_to_utf8(path.extension());
        std::ranges::transform(extension, extension.begin(),
                               [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        return extension;
    }

    [[nodiscard]] auto with_scene_extension(std::filesystem::path path) -> std::filesystem::path {
        if (lowercase_extension(path) != scene_file_extension) {
            path += scene_file_extension;
        }

        return path;
    }

    [[nodiscard]] auto scene_file_filters() -> std::vector<gui::FileBrowser::Filter> {
        return {
                {.label = "Lathe scenes (*.lbf)", .extensions = {std::string{scene_file_extension}}},
                {.label = "All files", .extensions = {}},
        };
    }

    auto fill_buffer(std::array<char, 512> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), buffer.size() - 1);
        std::ranges::copy(text.substr(0, length), buffer.begin());
        buffer[length] = '\0';
    }

} // namespace

auto Application::editor_scene_fingerprint() -> std::uint64_t {
    return scene_fingerprint(capture_scene(*editor_scene, *renderer, engine_models));
}

auto Application::editor_scene_dirty() -> bool { return editor_scene_fingerprint() != scene_clean_fingerprint; }

auto Application::mark_editor_scene_clean() -> void { scene_clean_fingerprint = editor_scene_fingerprint(); }

auto Application::request_open_scene(std::filesystem::path path) -> void {
    if (scene_load_job.has_value()) {
        scene_status = "A scene is already opening; wait for it to finish.";
        return;
    }

    if (editor_scene_dirty()) {
        pending_scene_open = std::move(path);
        unsaved_changes_popup_requested = true;
        return;
    }

    start_open_scene(std::move(path));
}

auto Application::start_open_scene(std::filesystem::path path) -> void {
    if (is_playing) {
        stop();
    }

    // The selection and the model-load list name entities that are about to be destroyed.
    selection_context().clear();
    model_loads.clear();

    scene_status = std::format("Opening '{}'...", gui::path_to_utf8(path.filename()));
    scene_load_job = SceneLoadJob::start(std::move(path));
}

auto Application::start_save_scene(std::filesystem::path path) -> void {
    if (path.empty()) {
        fill_buffer(save_as_buffer, scene_path.empty() ? default_scene_path : gui::path_to_utf8(scene_path));
        save_as_popup_requested = true;
        return;
    }

    if (scene_save_job.has_value()) {
        scene_status = "A save is already running.";
        return;
    }

    path = with_scene_extension(std::move(path));

    SceneSaveOptions options;

    if (scene_pack != nullptr) {
        options.source_packs.push_back(scene_pack);
    }

    scene_status = std::format("Saving '{}'...", gui::path_to_utf8(path.filename()));
    scene_save_job = SceneSaveJob::start(*editor_scene, *renderer, engine_models, std::move(path), std::move(options));
}

auto Application::update_scene_jobs() -> void {
    if (scene_save_job.has_value() && scene_save_job->ready()) {
        auto const path = scene_save_job->path();
        auto const fingerprint = scene_save_job->fingerprint();
        auto saved = scene_save_job->take();
        scene_save_job.reset();

        if (saved) {
            scene_path = path;
            scene_clean_fingerprint = fingerprint;

            // The new file is now the most complete source of cooked assets, and if it replaced the file scene_pack
            // was reading, the old table of contents no longer matches what's on disk.
            if (auto reopened = AssetPack::open(path)) {
                scene_pack = std::move(*reopened);
            } else {
                scene_pack.reset();
            }

            scene_status = std::format("Saved '{}' ({:.1f} MiB, {:.2f}s){}", gui::path_to_utf8(path.filename()),
                                       static_cast<double>(saved->file_size) / (1024.0 * 1024.0), saved->seconds,
                                       saved->warnings.empty() && saved->cook.failures.empty()
                                               ? ""
                                               : " with warnings; see the console");

            if (open_after_save.has_value()) {
                auto next = std::move(*open_after_save);
                open_after_save.reset();
                start_open_scene(std::move(next));
            }
        } else {
            error("Saving '{}' failed: {}", path.string(), describe(saved.error()));
            scene_status = std::format("Saving '{}' failed; see the console. Nothing was opened.",
                                       gui::path_to_utf8(path.filename()));
            open_after_save.reset();
        }
    }

    if (scene_load_job.has_value()) {
        auto const path = scene_load_job->path();
        auto finished = scene_load_job->step(*editor_scene, *renderer, engine_models);

        if (!finished.has_value()) {
            if (scene_load_job->phase() == SceneLoadJob::Phase::streaming) {
                auto const total = scene_load_job->models_total();
                scene_status = std::format("Opening '{}': streaming models {}/{}", gui::path_to_utf8(path.filename()),
                                           total - scene_load_job->models_remaining(*renderer), total);
            }
            return;
        }

        scene_load_job.reset();

        if (*finished) {
            scene_path = path;
            scene_pack = std::move((*finished)->pack);
            mark_editor_scene_clean();
            scene_status = std::format("Opened '{}' in {:.2f}s{}", gui::path_to_utf8(path.filename()),
                                       (*finished)->seconds,
                                       (*finished)->instantiate.warnings.empty() ? "" : " with warnings; see the console");
        } else {
            error("Opening '{}' failed: {}", path.string(), describe(finished->error()));
            scene_status = std::format("Opening '{}' failed; see the console.", gui::path_to_utf8(path.filename()));
        }
    }
}

auto Application::on_files_dropped(std::span<std::filesystem::path const> paths) -> void {
    std::optional<std::filesystem::path> scene;

    for (auto const &path: paths) {
        auto const extension = lowercase_extension(path);

        if (extension == scene_file_extension) {
            if (!scene.has_value()) {
                scene = path;
            } else {
                warn("Dropped several scenes; opening only '{}'", scene->string());
            }
        } else if (extension == ".gltf" || extension == ".glb") {
            spawn_streamed_model(path);
        } else {
            warn("Dropped '{}': only .lbf scenes and .gltf/.glb models can be dropped", path.string());
        }
    }

    if (scene.has_value()) {
        request_open_scene(std::move(*scene));
    }
}

auto Application::draw_scene_file_ui() -> void {
    bool const busy = scene_save_job.has_value() || scene_load_job.has_value();

    gui::widget("Scene", [&] {
        auto const title = scene_path.empty() ? std::string{"(untitled)"} : gui::path_to_utf8(scene_path);
        ImGui::Text("File: %s", title.c_str());

        ImGui::BeginDisabled(busy || scene_browser.is_open());
        if (ImGui::Button("Open...")) {
            scene_browser.open("Open Scene", scene_file_filters());
        }
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            start_save_scene(scene_path);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save As...")) {
            start_save_scene({});
        }
        ImGui::EndDisabled();

        if (!scene_status.empty()) {
            ImGui::TextWrapped("%s", scene_status.c_str());
        }

        ImGui::TextDisabled("Ctrl+S save, Ctrl+Shift+S save as, Ctrl+O open.");
        ImGui::TextDisabled("Drop a .lbf onto the window to open it, or a .gltf/.glb to add it.");
    });

    if (auto const picked = scene_browser.draw(editor_icons.get())) {
        request_open_scene(*picked);
    }

    if (unsaved_changes_popup_requested) {
        ImGui::OpenPopup(unsaved_changes_popup);
        unsaved_changes_popup_requested = false;

        if (scene_path.empty()) {
            fill_buffer(save_as_buffer, default_scene_path);
        }
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(unsaved_changes_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto const next = pending_scene_open.has_value() ? gui::path_to_utf8(pending_scene_open->filename()) : "";
        auto const current = scene_path.empty() ? std::string{"The current scene"}
                                                : std::format("'{}'", gui::path_to_utf8(scene_path.filename()));

        ImGui::Text("%s has unsaved changes.", current.c_str());
        ImGui::Text("Save before opening '%s'?", next.c_str());

        if (scene_path.empty()) {
            ImGui::InputText("Save to", save_as_buffer.data(), save_as_buffer.size());
        }

        ImGui::Separator();

        auto const close = [&] {
            pending_scene_open.reset();
            ImGui::CloseCurrentPopup();
        };

        ImGui::BeginDisabled(scene_save_job.has_value());
        if (ImGui::Button(scene_path.empty() ? "Save and open" : "Save, overwrite and open")) {
            auto target = scene_path.empty() ? gui::utf8_to_path(save_as_buffer.data()) : scene_path;
            open_after_save = pending_scene_open;
            start_save_scene(std::move(target));
            close();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        if (ImGui::Button("Discard changes and open")) {
            if (pending_scene_open.has_value()) {
                start_open_scene(*pending_scene_open);
            }
            close();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            close();
        }

        ImGui::EndPopup();
    }

    if (save_as_popup_requested) {
        ImGui::OpenPopup(save_as_popup);
        save_as_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(save_as_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Models and textures are cooked into the file, so it opens without the sources.");

        if (ImGui::IsWindowAppearing()) {
            ImGui::SetKeyboardFocusHere();
        }

        bool const confirmed = ImGui::InputText("Path", save_as_buffer.data(), save_as_buffer.size(),
                                                ImGuiInputTextFlags_EnterReturnsTrue);

        auto const target = with_scene_extension(gui::utf8_to_path(save_as_buffer.data()));
        std::error_code exists_error;

        if (target != scene_path && std::filesystem::exists(target, exists_error)) {
            ImGui::TextColored(ImVec4{1.0F, 0.75F, 0.3F, 1.0F}, "'%s' exists and will be overwritten.",
                               gui::path_to_utf8(target).c_str());
        }

        if (confirmed || ImGui::Button("Save")) {
            start_save_scene(target);
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}
