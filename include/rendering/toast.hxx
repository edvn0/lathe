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

    auto render_toasts() -> void;
}
