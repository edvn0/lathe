#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

// Corner toast notifications. The only way the rest of the engine reaches ImGuiNotify, which stays private to
// toast.cxx.
namespace gui {
    enum class ToastKind : std::uint8_t { info, success, warning, error };

    // Queues a toast. Safe from any thread; shown from the next render_toasts().
    auto toast(ToastKind kind, std::string_view title, std::string_view message,
               std::chrono::milliseconds duration) -> void;
    // 3 s.
    auto toast_info(std::string_view title, std::string_view message) -> void;
    // 3 s.
    auto toast_success(std::string_view title, std::string_view message) -> void;
    // 4 s.
    auto toast_warn(std::string_view title, std::string_view message) -> void;
    // 6 s.
    auto toast_error(std::string_view title, std::string_view message) -> void;

    // Hands queued toasts to ImGuiNotify and draws them. Once per frame, inside the ImGui frame.
    auto render_toasts() -> void;
} // namespace gui
