#include "app/application.hxx"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "rendering/debug_renderer.hxx"
#include "rendering/file_browser.hxx"

namespace {
    constexpr float resize_border = 5.0F;
    constexpr float caption_button_width = 46.0F;

    // GLFW's glfwStartWindowDrag edge bits (see cmake/patches/glfw-window-drag.patch).
    constexpr int edge_top = 1;
    constexpr int edge_bottom = 2;
    constexpr int edge_left = 4;
    constexpr int edge_right = 8;

    enum class CaptionButton : std::uint8_t { minimise, maximise, close };

    // Draws one title-bar button with a hand-drawn glyph so it does not depend on the icon font.
    [[nodiscard]] auto caption_button(CaptionButton kind, bool maximised, float height) -> bool {
        auto const origin = ImGui::GetCursorScreenPos();
        ImGui::PushID(static_cast<int>(kind));
        bool const pressed = ImGui::InvisibleButton("##caption", ImVec2{caption_button_width, height});
        ImGui::PopID();

        bool const hovered = ImGui::IsItemHovered();
        auto *draw = ImGui::GetWindowDrawList();
        auto const max = ImVec2{origin.x + caption_button_width, origin.y + height};

        if (hovered) {
            draw->AddRectFilled(origin, max,
                                kind == CaptionButton::close ? IM_COL32(196, 43, 28, 255)
                                                             : ImGui::GetColorU32(ImGuiCol_ButtonHovered));
        }

        auto const colour = ImGui::GetColorU32(ImGuiCol_Text);
        auto const centre = ImVec2{(origin.x + max.x) * 0.5F, (origin.y + max.y) * 0.5F};
        constexpr float half = 5.0F;

        switch (kind) {
            case CaptionButton::minimise:
                draw->AddLine({centre.x - half, centre.y}, {centre.x + half, centre.y}, colour);
                break;
            case CaptionButton::maximise:
                if (maximised) {
                    draw->AddRect({centre.x - half, centre.y - half + 2.0F}, {centre.x + half - 2.0F, centre.y + half},
                                  colour);
                    draw->AddLine({centre.x - half + 2.0F, centre.y - half + 2.0F},
                                  {centre.x - half + 2.0F, centre.y - half}, colour);
                    draw->AddLine({centre.x - half + 2.0F, centre.y - half}, {centre.x + half, centre.y - half}, colour);
                    draw->AddLine({centre.x + half, centre.y - half}, {centre.x + half, centre.y + half - 2.0F}, colour);
                } else {
                    draw->AddRect({centre.x - half, centre.y - half}, {centre.x + half, centre.y + half}, colour);
                }
                break;
            case CaptionButton::close:
                draw->AddLine({centre.x - half, centre.y - half}, {centre.x + half, centre.y + half}, colour);
                draw->AddLine({centre.x - half, centre.y + half}, {centre.x + half, centre.y - half}, colour);
                break;
        }

        return pressed;
    }
} // namespace

auto Application::remove_window_decorations() -> void {
    // The window is created undecorated (run_engine, via EditorProfile::custom_title_bar) only when windowed; fullscreen
    // and headless runs keep what they have, and the menu bar then has no window buttons.
    custom_title_bar = context.undecorated_window;
}

// Edge resizing for the undecorated window: the compositor / window manager does the actual resize.
auto Application::handle_window_resize() -> void {
    auto *const window = context.window;
    if (!custom_title_bar || window == nullptr || glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE) {
        return;
    }

    auto const *viewport = ImGui::GetMainViewport();
    auto const mouse = ImGui::GetIO().MousePos;
    auto const min = viewport->Pos;
    auto const max = ImVec2{min.x + viewport->Size.x, min.y + viewport->Size.y};

    if (mouse.x < min.x || mouse.y < min.y || mouse.x >= max.x || mouse.y >= max.y) {
        return;
    }

    int edges = 0;
    edges |= mouse.x < min.x + resize_border ? edge_left : 0;
    edges |= mouse.x >= max.x - resize_border ? edge_right : 0;
    edges |= mouse.y < min.y + resize_border ? edge_top : 0;
    edges |= mouse.y >= max.y - resize_border ? edge_bottom : 0;

    if (edges == 0) {
        return;
    }

    switch (edges) {
        case edge_left:
        case edge_right:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            break;
        case edge_top:
        case edge_bottom:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            break;
        case edge_top | edge_left:
        case edge_bottom | edge_right:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
            break;
        default:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNESW);
            break;
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        glfwStartWindowDrag(window, edges);
    }
}

auto Application::draw_main_menu_bar() -> void {
    auto *const window = context.window;

    if (!ImGui::BeginMainMenuBar()) {
        return;
    }

    bool const busy = scene_save_job.has_value() || scene_load_job.has_value() || packaging_active();

    if (ImGui::BeginMenu("File")) {
        ImGui::BeginDisabled(busy);
        if (ImGui::MenuItem("New Scene", "Ctrl+N")) {
            request_switch_game("empty");
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(busy || scene_browser.is_open());
        if (ImGui::MenuItem("Open Scene...", "Ctrl+O")) {
            scene_browser.open("Open Scene", {{.label = "Lathe scenes (*.lbf)", .extensions = {".lbf"}}});
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(busy);
        if (ImGui::MenuItem("Save", "Ctrl+S")) {
            start_save_scene(scene_path);
        }
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S")) {
            start_save_scene({});
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        ImGui::BeginDisabled(busy);
        if (ImGui::MenuItem("New Project...")) {
            request_new_project_dialog();
        }
        if (ImGui::MenuItem("Open Project...")) {
            project_browser.open("Open Project", {{.label = "Lathe projects (lathe.project)", .extensions = {".project"}}});
        }
        if (ImGui::BeginMenu("Recent Projects")) {
            auto const recent = RecentProjects::user_default().list();

            for (auto const &root: recent) {
                if (ImGui::MenuItem(gui::path_to_utf8(root).c_str())) {
                    request_open_project(root);
                }
            }

            if (recent.empty()) {
                ImGui::TextDisabled("No recent projects");
            }
            ImGui::EndMenu();
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        ImGui::BeginDisabled(busy || model_browser.is_open());
        if (ImGui::MenuItem("Import Model...")) {
            request_import_model();
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        ImGui::BeginDisabled(busy);
        if (ImGui::BeginMenu("Load Game")) {
            for (auto const name: available_games) {
                if (ImGui::MenuItem(std::string{name}.c_str(), nullptr, name == game_name)) {
                    request_switch_game(std::string{name});
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Package Game...")) {
            request_package_dialog();
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        if (ImGui::MenuItem("Exit", "Alt+F4")) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
        ImGui::BeginDisabled(is_playing);
        if (ImGui::MenuItem("Translate", "1", gizmo_operation == ImGuizmo::TRANSLATE)) {
            gizmo_operation = ImGuizmo::TRANSLATE;
        }
        if (ImGui::MenuItem("Rotate", "2", gizmo_operation == ImGuizmo::ROTATE)) {
            gizmo_operation = ImGuizmo::ROTATE;
        }
        if (ImGui::MenuItem("Scale", "3", gizmo_operation == ImGuizmo::SCALE)) {
            gizmo_operation = ImGuizmo::SCALE;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Local Space", "4", gizmo_mode == ImGuizmo::LOCAL)) {
            gizmo_mode = gizmo_mode == ImGuizmo::WORLD ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Place Text", "T", text_placement.active())) {
            text_placement.set_active(!text_placement.active());
        }
        if (ImGui::MenuItem("Focus Selection", "F")) {
            focus_selection();
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        ImGui::MenuItem("Project Settings...", nullptr, &panels.project_settings);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        // Each viewport carries its own view mode; the menu only splits the area.
        if (ImGui::MenuItem("Split Viewport", nullptr, &second_viewport_open, !is_playing) && second_viewport_open) {
            split_viewport_requested = true;
        }

        ImGui::Separator();

        bool grid = renderer->editor_grid();
        if (ImGui::MenuItem("Grid and Axes", nullptr, &grid)) {
            renderer->set_editor_grid(grid);
        }

        bool cones = debug_renderer->meshlet_cones_debug_enabled();
        if (ImGui::MenuItem("Meshlet Normal Cones", nullptr, &cones)) {
            debug_renderer->set_meshlet_cones_debug_enabled(cones);
        }

        bool bounds = debug_renderer->model_bounds_debug_enabled();
        if (ImGui::MenuItem("Submesh Bounds", nullptr, &bounds)) {
            debug_renderer->set_model_bounds_debug_enabled(bounds);
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Render Scale")) {
            auto percent = static_cast<int>(std::lround(renderer->render_scale() * 100.0F));
            ImGui::SetNextItemWidth(180.0F);
            if (ImGui::SliderInt("##render_scale", &percent, static_cast<int>(Renderer::min_render_scale * 100.0F),
                                 static_cast<int>(Renderer::max_render_scale * 100.0F), "%d%%")) {
                renderer->set_render_scale(static_cast<float>(percent) / 100.0F);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Window")) {
        ImGui::MenuItem("Hierarchy", nullptr, &panels.hierarchy);
        ImGui::MenuItem("Inspector", nullptr, &panels.inspector);
        ImGui::MenuItem("Assets", nullptr, &panels.assets);
        ImGui::MenuItem("Console", nullptr, &panels.console);
        ImGui::MenuItem("Script", nullptr, &panels.script);
        ImGui::MenuItem("Game Panels", nullptr, &panels.game);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Layout")) {
            reset_layout_requested = true;
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Debug")) {
        ImGui::MenuItem("Frame Graph", nullptr, &panels.frame_graph);
        ImGui::MenuItem("Frame Timings", nullptr, &panels.frame_timings);
        ImGui::MenuItem("Scene Stats", nullptr, &panels.scene_stats);
        ImGui::MenuItem("Events", nullptr, &panels.events);
        ImGui::MenuItem("CVars", nullptr, &panels.cvars);
#if LATHE_TRACK_MEMORY
        ImGui::MenuItem("Memory", nullptr, &panels.memory);
#endif
        ImGui::Separator();
        ImGui::MenuItem("Scene File Inspector", nullptr, &panels.scene_inspector);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Play")) {
        if (ImGui::MenuItem(is_playing ? "Stop" : "Play")) {
            if (is_playing) {
                stop();
            } else {
                play();
            }
        }
        ImGui::MenuItem("Fullscreen", nullptr, &play_fullscreen);
        ImGui::Separator();
        if (ImGui::MenuItem("Screenshot", "F12")) {
            request_screenshot();
        }
        ImGui::MenuItem("Viewport Only", nullptr, &screenshot_viewport_only);
        ImGui::EndMenu();
    }

    // Compact toolbar: gizmo tools and Play/Stop, the things Unreal keeps in its main toolbar.
    ImGui::Dummy(ImVec2{8.0F, 0.0F});
    ImGui::SameLine();
    ImGui::BeginDisabled(is_playing);
    {
        auto const size = ImGui::GetFrameHeight() - 6.0F;
        auto const tool = [&](gui::EditorIcon icon, bool active, char const *id, char const *tooltip) {
            auto const bg = active ? ImVec4{0.18F, 0.50F, 0.93F, 0.55F} : ImVec4{0.0F, 0.0F, 0.0F, 0.0F};
            auto const tint = active ? ImVec4{1.0F, 1.0F, 1.0F, 1.0F} : ImVec4{0.68F, 0.68F, 0.72F, 1.0F};
            bool const pressed = ImGui::ImageButton(id, editor_icons->texture(icon), ImVec2{size, size}, ImVec2{0, 0},
                                                    ImVec2{1, 1}, bg, tint);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
                ImGui::SetTooltip("%s", tooltip);
            }
            ImGui::SameLine();
            return pressed;
        };

        if (tool(gui::EditorIcon::move, gizmo_operation == ImGuizmo::TRANSLATE, "##tb_move", "Translate (1)")) {
            gizmo_operation = ImGuizmo::TRANSLATE;
        }
        if (tool(gui::EditorIcon::rotate, gizmo_operation == ImGuizmo::ROTATE, "##tb_rotate", "Rotate (2)")) {
            gizmo_operation = ImGuizmo::ROTATE;
        }
        if (tool(gui::EditorIcon::scale, gizmo_operation == ImGuizmo::SCALE, "##tb_scale", "Scale (3)")) {
            gizmo_operation = ImGuizmo::SCALE;
        }
        bool const local = gizmo_mode == ImGuizmo::LOCAL;
        if (tool(local ? gui::EditorIcon::local : gui::EditorIcon::world, true, "##tb_space",
                 local ? "Local space (4)" : "World space (4)")) {
            gizmo_mode = local ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
        }
    }
    ImGui::EndDisabled();

    if (ImGui::Button(is_playing ? "  Stop  " : "  Play  ")) {
        if (is_playing) {
            stop();
        } else {
            play();
        }
    }

    // Everything to the right of the menus is the title bar: scene name, a drag area, then the window buttons.
    auto const bar_height = ImGui::GetFrameHeight();
    auto const buttons_width = custom_title_bar ? caption_button_width * 3.0F : 0.0F;
    auto const available = ImGui::GetContentRegionAvail().x - buttons_width;

    if (available > 0.0F) {
        auto const title = std::string{scene_path.empty() ? "(untitled)" : gui::path_to_utf8(scene_path.filename())} +
                           (source_scene_dirty() ? "*" : "") + "  -  " + game_name;
        ImGui::InvisibleButton("##title_drag", ImVec2{available, bar_height});

        auto const min = ImGui::GetItemRectMin();
        auto const max = ImGui::GetItemRectMax();
        auto const text_size = ImGui::CalcTextSize(title.c_str());
        if (text_size.x < available) {
            ImGui::GetWindowDrawList()->AddText(
                    {min.x + ((max.x - min.x) - text_size.x) * 0.5F, min.y + ((max.y - min.y) - text_size.y) * 0.5F},
                    ImGui::GetColorU32(ImGuiCol_TextDisabled), title.c_str());
        }

        if (custom_title_bar) {
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE) {
                    glfwRestoreWindow(window);
                } else {
                    glfwMaximizeWindow(window);
                }
            } else if (ImGui::IsItemActivated()) {
                glfwStartWindowDrag(window, 0);
            }
        }
    }

    if (custom_title_bar) {
        bool const maximised = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0.0F, 0.0F});
        if (caption_button(CaptionButton::minimise, false, bar_height)) {
            glfwIconifyWindow(window);
        }
        ImGui::SameLine();
        if (caption_button(CaptionButton::maximise, maximised, bar_height)) {
            if (maximised) {
                glfwRestoreWindow(window);
            } else {
                glfwMaximizeWindow(window);
            }
        }
        ImGui::SameLine();
        if (caption_button(CaptionButton::close, false, bar_height)) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        ImGui::PopStyleVar();
    }

    ImGui::EndMainMenuBar();
}
