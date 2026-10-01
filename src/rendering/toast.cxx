#include "rendering/toast.hxx"

// ImGuiNotify.hpp uses these without including them.
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <imgui.h>

#include "imgui_internal.h"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

// The monitor-corner placement reads ImGuiPlatformIO::Monitors, which is only filled with multi-viewport support.
#define NOTIFY_RENDER_OUTSIDE_MAIN_WINDOW false
#include "ImGuiNotify.hpp"

namespace gui {
    namespace {
        struct PendingToast {
            ToastKind kind = ToastKind::info;
            std::string title;
            std::string message;
            std::chrono::milliseconds duration{};
        };

        // toast() may be called from any thread; ImGuiNotify's own list is only touched in render_toasts().
        std::mutex pending_mutex;
        std::vector<PendingToast> pending_toasts;

        [[nodiscard]] auto notify_type(ToastKind kind) noexcept -> ImGuiToastType {
            switch (kind) {
                case ToastKind::info:
                    return ImGuiToastType::Info;
                case ToastKind::success:
                    return ImGuiToastType::Success;
                case ToastKind::warning:
                    return ImGuiToastType::Warning;
                case ToastKind::error:
                    return ImGuiToastType::Error;
            }
            return ImGuiToastType::Info;
        }
    } // namespace

    auto toast(ToastKind kind, std::string_view title, std::string_view message,
               std::chrono::milliseconds duration) -> void {
        std::scoped_lock const lock{pending_mutex};
        pending_toasts.push_back(PendingToast{
                .kind = kind,
                .title = std::string{title},
                .message = std::string{message},
                .duration = duration,
        });
    }

    auto toast_info(std::string_view title, std::string_view message) -> void {
        toast(ToastKind::info, title, message, std::chrono::milliseconds{3000});
    }

    auto toast_success(std::string_view title, std::string_view message) -> void {
        toast(ToastKind::success, title, message, std::chrono::milliseconds{3000});
    }

    auto toast_warn(std::string_view title, std::string_view message) -> void {
        toast(ToastKind::warning, title, message, std::chrono::milliseconds{4000});
    }

    auto toast_error(std::string_view title, std::string_view message) -> void {
        toast(ToastKind::error, title, message, std::chrono::milliseconds{6000});
    }

    auto render_toasts() -> void {
        std::vector<PendingToast> ready;
        {
            std::scoped_lock const lock{pending_mutex};
            ready.swap(pending_toasts);
        }

        for (auto const &pending: ready) {
            ImGuiToast notification{notify_type(pending.kind), static_cast<int>(pending.duration.count())};
            // Always "%s": the text is never a format string. vsnprintf cuts it at NOTIFY_MAX_MSG_LENGTH.
            notification.setTitle("%s", pending.title.c_str());
            notification.setContent("%s", pending.message.c_str());
            ImGui::InsertNotification(notification);
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0F);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.10F, 0.10F, 0.10F, 1.0F});
        ImGui::RenderNotifications();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }
} // namespace gui
