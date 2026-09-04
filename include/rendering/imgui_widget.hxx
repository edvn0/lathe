#pragma once

#include <string_view>
#include <type_traits>

#include <glm/vec2.hpp>
#include <imgui.h>

// Shared ImGui window helper -- used by both the engine's own panels
// (Application::on_ui()) and per-game panels (IGame::on_ui()), so every
// window in the app opens the same way rather than each call site
// hand-rolling its own Begin/End pair.
namespace gui {
    // Opens a window named `name`, then calls `f` with whatever subset of
    // (size, position) it accepts -- dispatched on f's arity since a generic
    // lambda has one signature, so which of these f can accept is a
    // compile-time property of f, not something callers select between at
    // the call site. Always calls ImGui::End(), even if Begin() returned
    // false, per ImGui's own contract.
    inline constexpr auto widget = [](std::string_view name, auto &&f) -> bool {
        if (!ImGui::Begin(name.data())) {
            ImGui::End();
            return false;
        }

        if constexpr (std::is_invocable_v<decltype(f), glm::vec2, glm::vec2>) {
            auto const size = ImGui::GetWindowSize();
            auto const position = ImGui::GetWindowPos();
            f(glm::vec2{size.x, size.y}, glm::vec2{position.x, position.y});
        } else if constexpr (std::is_invocable_v<decltype(f), glm::vec2>) {
            auto const size = ImGui::GetWindowSize();
            f(glm::vec2{size.x, size.y});
        } else {
            f();
        }

        ImGui::End();
        return false;
    };
} // namespace gui
