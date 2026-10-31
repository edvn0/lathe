
#include "app/application.hxx"

#include <algorithm>
#include <GLFW/glfw3.h>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>
#include <format>
#include <string>
#include <utility>

#include "assets/model_formats.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/imgui_widget.hxx"
#include "scene/selection_context.hxx"

namespace {

    constexpr auto unsaved_changes_popup = "Unsaved changes";
    constexpr auto save_as_popup = "Save scene as";
    constexpr auto game_switch_popup = "Load game";
    constexpr auto exit_popup = "Exit";
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

    auto fill_buffer(std::array<char, 512> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), buffer.size() - 1);
        std::ranges::copy(text.substr(0, length), buffer.begin());
        buffer[length] = '\0';
    }

}

auto Application::source_scene_fingerprint() -> std::uint64_t {
    return scene_fingerprint(capture_scene(*source_scene, *renderer, engine_models));
}

auto Application::source_scene_dirty() -> bool { return source_scene_fingerprint() != scene_clean_fingerprint; }

auto Application::mark_source_scene_clean() -> void { scene_clean_fingerprint = source_scene_fingerprint(); }

auto Application::request_open_scene(std::filesystem::path path) -> void {
    if (scene_load_job.has_value()) {
        scene_status = "A scene is already opening; wait for it to finish.";
        return;
    }

    if (source_scene_dirty()) {
        pending_scene_open = std::move(path);
        unsaved_changes_popup_requested = true;
        return;
    }

    open_scene(std::move(path));
}

auto Application::request_switch_game(std::string name) -> void {
    if (scene_load_job.has_value() || scene_save_job.has_value()) {
        scene_status = "A scene is loading or saving; wait for it to finish.";
        return;
    }

    if (source_scene_dirty()) {
        pending_game_switch = std::move(name);
        game_switch_popup_requested = true;
        return;
    }

    switch_game(std::move(name));
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
    scene_save_job = SceneSaveJob::start(*source_scene, *renderer, engine_models, std::move(path), std::move(options));
}

auto Application::recover_work() -> std::optional<std::filesystem::path> {
    if (!source_scene_dirty()) {
        return std::nullopt;
    }

    auto path = std::filesystem::path{};

    if (!scene_path.empty()) {
        path = scene_path;
        path.replace_filename(scene_path.stem().string() + ".recovered" + scene_path.extension().string());
    } else {
        auto const stamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        path = Paths::user_cache_directory() / "recovered" / std::format("untitled-{:%Y%m%d-%H%M%S}", stamp);
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
    }

    path = with_scene_extension(std::move(path));

    // A save that was already running owns the scene file; a second one would race it.
    if (scene_save_job.has_value()) {
        scene_save_job.reset();
    }

    // Reference the sources instead of cooking them into the file: cooking a big model and its textures takes minutes,
    // and the GPU is gone, so a recovery save has to be quick and light.
    SceneSaveOptions options;
    options.embed_assets = false;

    if (scene_pack != nullptr) {
        options.source_packs.push_back(scene_pack);
    }

    auto job = std::make_unique<SceneSaveJob>(SceneSaveJob::start(*source_scene, *renderer, engine_models, path, std::move(options)));

    // The save reads no GPU state, but give up rather than hang the exit if something it touches is stuck. The job is
    // leaked then, since destroying a running one would block.
    constexpr auto timeout = std::chrono::seconds{20};
    auto const deadline = std::chrono::steady_clock::now() + timeout;

    while (!job->ready()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            error("Recovery save of '{}' did not finish in {} s", gui::path_to_utf8(path), timeout.count());
            static_cast<void>(job.release());
            return std::nullopt;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }

    auto saved = job->take();

    if (!saved) {
        error("Recovery save of '{}' failed: {}", gui::path_to_utf8(path), describe(saved.error()));
        return std::nullopt;
    }

    info("Recovered the unsaved scene to '{}'", gui::path_to_utf8(path));
    return path;
}

auto Application::update_scene_save() -> void {
    if (scene_save_job.has_value() && scene_save_job->ready()) {
        auto const path = scene_save_job->path();
        auto const fingerprint = scene_save_job->fingerprint();
        auto saved = scene_save_job->take();
        scene_save_job.reset();

        if (saved) {
            scene_path = path;
            scene_clean_fingerprint = fingerprint;

            if (auto reopened = AssetPack::open(path)) {
                scene_pack = std::move(*reopened);
            } else {
                scene_pack.reset();
            }

            scene_status = std::format(
                    "Saved '{}' ({:.1f} MiB, {:.2f}s){}", gui::path_to_utf8(path.filename()),
                    static_cast<double>(saved->file_size) / (1024.0 * 1024.0), saved->seconds,
                    saved->warnings.empty() && saved->cook.failures.empty() ? "" : " with warnings; see the console");

            if (exit_after_save) {
                exit_after_save = false;
                exit_confirmed = true;
                glfwSetWindowShouldClose(context.window, GLFW_TRUE);
            }

            if (open_after_save.has_value()) {
                auto next = std::move(*open_after_save);
                open_after_save.reset();
                open_scene(std::move(next));
            }
        } else {
            error("Saving '{}' failed: {}", path.string(), describe(saved.error()));
            scene_status = std::format("Saving '{}' failed; see the console. Nothing was opened.",
                                       gui::path_to_utf8(path.filename()));
            open_after_save.reset();
            exit_after_save = false;
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
        } else if (is_supported_model(path)) {
            spawn_streamed_model(path);
        } else {
            warn("Dropped '{}': only .lbf scenes and .gltf/.glb/.fbx/.obj/.dae/.ply/.stl/.usd/.usda/.usdc/.usdz models can be dropped", path.string());
        }
    }

    if (scene.has_value()) {
        request_open_scene(std::move(*scene));
    }
}

auto Application::draw_scene_file_ui() -> void {
    draw_package_ui();
    draw_project_ui();
    draw_exit_ui();

    scene_inspector.draw(editor_icons.get(), scene_path, panels.scene_inspector);

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
                open_scene(*pending_scene_open);
            }
            close();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            close();
        }

        ImGui::EndPopup();
    }

    if (game_switch_popup_requested) {
        ImGui::OpenPopup(game_switch_popup);
        game_switch_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (ImGui::BeginPopupModal(game_switch_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("The scene has unsaved changes. Loading '%s' replaces it.",
                    pending_game_switch.value_or("").c_str());

        if (ImGui::Button("Discard changes and load")) {
            if (pending_game_switch.has_value()) {
                switch_game(std::move(*pending_game_switch));
            }
            pending_game_switch.reset();
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            pending_game_switch.reset();
            ImGui::CloseCurrentPopup();
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

auto Application::unsaved_changes() -> std::vector<std::string> {
    std::vector<std::string> changes;

    if (source_scene_dirty()) {
        changes.emplace_back(scene_path.empty() ? "The scene has unsaved changes and no file yet."
                                                : std::format("'{}' has unsaved changes.",
                                                              gui::path_to_utf8(scene_path.filename())));
    }

    if (project_form_dirty()) {
        changes.emplace_back("The project settings have edits that are not saved.");
    }

    if (scene_save_job.has_value()) {
        changes.emplace_back("A scene is still being saved.");
    }

    if (packaging_active()) {
        changes.emplace_back("A game package is still being built.");
    }

    return changes;
}

auto Application::confirm_exit() -> bool {
    if (exit_confirmed) {
        return true;
    }

    exit_scene_dirty = source_scene_dirty();
    exit_changes = unsaved_changes();

    if (exit_changes.empty()) {
        return true;
    }

    exit_popup_requested = true;

    return false;
}

auto Application::draw_exit_ui() -> void {
    if (exit_popup_requested) {
        ImGui::OpenPopup(exit_popup);
        exit_popup_requested = false;
    }

    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

    if (!ImGui::BeginPopupModal(exit_popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    ImGui::TextUnformatted("Exit with changes that will be lost?");

    for (auto const &change: exit_changes) {
        ImGui::BulletText("%s", change.c_str());
    }

    ImGui::Separator();

    auto const scene_unsavable = exit_scene_dirty && scene_path.empty();

    ImGui::BeginDisabled(scene_save_job.has_value() || packaging_active() || scene_unsavable);
    if (ImGui::Button("Save and exit")) {
        if (project_form_dirty()) {
            static_cast<void>(save_project_form());
        }

        if (exit_scene_dirty) {
            exit_after_save = true;
            start_save_scene(scene_path);
        } else {
            exit_confirmed = true;
            glfwSetWindowShouldClose(context.window, GLFW_TRUE);
        }

        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();

    if (scene_unsavable) {
        ImGui::SetItemTooltip("Save the scene to a file first (File > Save As).");
    }

    ImGui::SameLine();
    if (ImGui::Button("Exit without saving")) {
        exit_confirmed = true;
        glfwSetWindowShouldClose(context.window, GLFW_TRUE);
        ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
}
