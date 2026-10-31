// The editor's commands that have a shortcut: file and scene actions, gizmo modes, tools. The key handler, the menu bar
// and the Add menu all run these through Application::commands, so a shortcut and its menu entry cannot drift apart.

#include "app/application.hxx"

#include <string>
#include <string_view>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include "rendering/renderer.hxx"

auto Application::shortcut_label(std::string_view id) const -> std::string {
    auto const *command = commands.find(id);

    return command == nullptr ? std::string{} : editor::shortcut_text(command->shortcut);
}

auto Application::register_editor_commands() -> void {
    using editor::CommandContext;

    auto const not_busy = [this] {
        return !scene_save_job.has_value() && !scene_load_job.has_value() && !packaging_active();
    };

    commands.add({.id = "file.new", .label = "New Scene", .shortcut = {GLFW_KEY_N, GLFW_MOD_CONTROL},
                  .enabled = not_busy, .action = [this] { request_switch_game("empty"); }});

    commands.add({.id = "file.open", .label = "Open Scene...", .shortcut = {GLFW_KEY_O, GLFW_MOD_CONTROL},
                  .enabled = [this, not_busy] { return not_busy() && !scene_browser.is_open(); },
                  .action = [this] {
                      scene_browser.open("Open Scene", {{.label = "Lathe scenes (*.lbf)", .extensions = {".lbf"}}});
                  }});

    commands.add({.id = "file.save", .label = "Save", .shortcut = {GLFW_KEY_S, GLFW_MOD_CONTROL}, .enabled = not_busy,
                  .action = [this] { start_save_scene(scene_path); }});

    commands.add({.id = "file.save_as", .label = "Save As...",
                  .shortcut = {GLFW_KEY_S, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT}, .enabled = not_busy,
                  .action = [this] { start_save_scene({}); }});

    commands.add({.id = "scene.repopulate", .label = "Repopulate Scene",
                  .shortcut = {GLFW_KEY_R, GLFW_MOD_CONTROL}, .enabled = [this] { return !scene_load_job.has_value(); },
                  .action = [this] {
                      renderer->queue_render_thread_event([this] {
                          if (is_playing) {
                              stop();
                          }

                          game->on_populate(*source_scene, *renderer, engine_models);
                          scene_path.clear();
                          scene_pack.reset();
                          mark_source_scene_clean();
                      });
                  }});

    commands.add({.id = "gizmo.translate", .label = "Translate", .shortcut = {GLFW_KEY_1, 0},
                  .context = CommandContext::viewport, .action = [this] { gizmo_operation = ImGuizmo::TRANSLATE; }});
    commands.add({.id = "gizmo.rotate", .label = "Rotate", .shortcut = {GLFW_KEY_2, 0},
                  .context = CommandContext::viewport, .action = [this] { gizmo_operation = ImGuizmo::ROTATE; }});
    commands.add({.id = "gizmo.scale", .label = "Scale", .shortcut = {GLFW_KEY_3, 0},
                  .context = CommandContext::viewport, .action = [this] { gizmo_operation = ImGuizmo::SCALE; }});
    commands.add({.id = "gizmo.toggle_space", .label = "Local Space", .shortcut = {GLFW_KEY_4, 0},
                  .context = CommandContext::viewport, .action = [this] {
                      gizmo_mode = gizmo_mode == ImGuizmo::WORLD ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
                  }});

    commands.add({.id = "edit.focus_selection", .label = "Focus Selection", .shortcut = {GLFW_KEY_F, 0},
                  .context = CommandContext::viewport, .action = [this] { focus_selection(); }});

    commands.add({.id = "tool.place_text", .label = "Place Text", .shortcut = {GLFW_KEY_T, 0},
                  .context = CommandContext::viewport,
                  .action = [this] { text_placement.set_active(!text_placement.active()); }});
    commands.add({.id = "tool.cancel_text", .label = "Stop Placing Text", .shortcut = {GLFW_KEY_ESCAPE, 0},
                  .context = CommandContext::viewport, .action = [this] { text_placement.set_active(false); }});

    commands.add({.id = "viewport.add_menu", .label = "Add...", .shortcut = {GLFW_KEY_A, GLFW_MOD_SHIFT},
                  .context = CommandContext::viewport, .action = [this] { add_menu_requested = true; }});
}
