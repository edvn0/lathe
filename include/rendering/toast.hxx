#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

namespace gui {
    enum class ToastKind : std::uint8_t { info, success, warning, error };

    auto toast(ToastKind kind, std::string_view title, std::string_view message,
               std::chrono::milliseconds duration) -> void;
    auto toast_info(std::string_view title, std::string_view message) -> void;
    auto toast_success(std::string_view title, std::string_view message) -> void;
    auto toast_warn(std::string_view title, std::string_view message) -> void;
    auto toast_error(std::string_view title, std::string_view message) -> void;

    // Toasts stack in the bottom-right corner of this screen rectangle (the editor's viewport panel) until cleared,
    // when they go back to the window's corner.
    auto set_toast_anchor(float x, float y, float width, float height) -> void;
    auto clear_toast_anchor() -> void;

    auto render_toasts() -> void;
}
