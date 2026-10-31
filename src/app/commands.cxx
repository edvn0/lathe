#include "app/commands.hxx"

#include <algorithm>
#include <utility>

#include <GLFW/glfw3.h>

namespace editor {

    auto CommandRegistry::add(Command command) -> void {
        auto const existing = std::ranges::find(commands_, command.id, &Command::id);

        if (existing != commands_.end()) {
            *existing = std::move(command);
            return;
        }

        commands_.push_back(std::move(command));
    }

    auto CommandRegistry::find(std::string_view id) const noexcept -> Command const * {
        auto const found = std::ranges::find(commands_, id, &Command::id);

        return found == commands_.end() ? nullptr : &*found;
    }

    auto CommandRegistry::is_enabled(std::string_view id) const -> bool {
        auto const *command = find(id);

        return command != nullptr && command->action && (!command->enabled || command->enabled());
    }

    auto CommandRegistry::run(std::string_view id) const -> bool {
        if (!is_enabled(id)) {
            return false;
        }

        find(id)->action();
        return true;
    }

    auto CommandRegistry::run_for_key(int key, int modifiers, std::function<bool(CommandContext)> const &active) const
            -> bool {
        for (auto const &command: commands_) {
            if (command.shortcut.key != key || command.shortcut.modifiers != modifiers || !active(command.context)) {
                continue;
            }

            if (command.action && (!command.enabled || command.enabled())) {
                command.action();
            }
            return true;
        }

        return false;
    }

    auto shortcut_text(Shortcut shortcut) -> std::string {
        if (shortcut.key == 0) {
            return {};
        }

        auto text = std::string{};
        if ((shortcut.modifiers & GLFW_MOD_CONTROL) != 0) {
            text += "Ctrl+";
        }
        if ((shortcut.modifiers & GLFW_MOD_SHIFT) != 0) {
            text += "Shift+";
        }
        if ((shortcut.modifiers & GLFW_MOD_ALT) != 0) {
            text += "Alt+";
        }

        auto const key = shortcut.key;
        if (key >= GLFW_KEY_F1 && key <= GLFW_KEY_F25) {
            text += 'F' + std::to_string(key - GLFW_KEY_F1 + 1);
        } else if (key == GLFW_KEY_ESCAPE) {
            text += "Esc";
        } else if ((key >= GLFW_KEY_0 && key <= GLFW_KEY_9) || (key >= GLFW_KEY_A && key <= GLFW_KEY_Z)) {
            // These GLFW key codes are the characters themselves.
            text += static_cast<char>(key);
        } else {
            text += '?';
        }

        return text;
    }

}
