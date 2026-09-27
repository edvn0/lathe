#pragma once

#include <type_traits>

#include <glm/vec2.hpp>
#include <imgui.h>

namespace gui {
    // Opens window `name` and calls `f` with whichever of (size, position) it accepts. Always calls ImGui::End().
    inline constexpr auto widget = [](char const *name, auto &&f) -> bool {
        if (!ImGui::Begin(name)) {
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
