#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace editor {

    // A key and the modifiers held with it, as GLFW reports them (GLFW_KEY_*, GLFW_MOD_*). Key 0 is no shortcut.
    struct Shortcut {
        int key = 0;
        int modifiers = 0;
    };

    // Where a shortcut may fire. The caller decides what each one means (see Application::handle_event).
    enum class CommandContext : std::uint8_t {
        always,
        // Not playing and no text field wants the keyboard.
        editing,
        // As editing, with the pointer over a viewport (or a drag that began there): the viewport's tools and camera.
        viewport,
    };

    // One thing the editor can do by name: a menu entry, a popup item, later a palette entry.
    struct Command {
        // Stable and unique, e.g. "add.cube"; what bindings and scripts refer to.
        std::string id;
        std::string label;
        // The submenu or group the command is listed under, e.g. "Add".
        std::string category;
        Shortcut shortcut;
        CommandContext context = CommandContext::always;
        // Empty means always enabled.
        std::function<bool()> enabled;
        std::function<void()> action;
    };

    class CommandRegistry {
    public:
        // Registering an id that exists replaces the earlier command.
        auto add(Command command) -> void;

        [[nodiscard]] auto commands() const noexcept -> std::span<Command const> { return commands_; }
        [[nodiscard]] auto find(std::string_view id) const noexcept -> Command const *;

        // False if there is no such command or it is disabled right now.
        [[nodiscard]] auto is_enabled(std::string_view id) const -> bool;

        // Runs the command whatever its context; false if there is none with that id or it is disabled.
        auto run(std::string_view id) const -> bool;

        // Runs the command bound to this key if its context is `active`. True if a command is bound there and its
        // context is active, even if it is disabled, so the key does not also reach something else.
        auto run_for_key(int key, int modifiers, std::function<bool(CommandContext)> const &active) const -> bool;

    private:
        std::vector<Command> commands_;
    };

    // "Ctrl+Shift+S", "F", "1"; empty for no shortcut.
    [[nodiscard]] auto shortcut_text(Shortcut shortcut) -> std::string;

}
