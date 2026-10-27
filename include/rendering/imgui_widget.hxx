#pragma once

#include <type_traits>

#include <glm/vec2.hpp>
#include <imgui.h>

namespace gui {
    inline constexpr auto widget = [](char const *name, auto &&f, bool *p_open = nullptr) -> bool {
        if (!ImGui::Begin(name, p_open)) {
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

    // A widget with a close button, driven by `open` so a menu can bring it back. Skipped entirely while closed.
    inline constexpr auto panel = [](char const *name, bool &open, auto &&f) -> bool {
        if (!open) {
            return false;
        }
        return widget(name, f, &open);
    };
}
