#include "rendering/toast.hxx"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <imgui.h>

#include "imgui_internal.h"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ImGuiNotify places toasts against the main viewport. A toast anchor stands in for it so they can sit in the editor's
// viewport panel instead.
namespace {
    ImGuiViewport toast_anchor_viewport;
    bool toast_anchored = false;
}

namespace ImGui {
    inline auto ToastAnchorViewport() -> ImGuiViewport * {
        return toast_anchored ? &toast_anchor_viewport : GetMainViewport();
    }
}

#define NOTIFY_RENDER_OUTSIDE_MAIN_WINDOW false
#define GetMainViewport ToastAnchorViewport
#include "ImGuiNotify.hpp"
#undef GetMainViewport

namespace gui {
    namespace {
        struct PendingToast {
            ToastKind kind = ToastKind::info;
            std::string title;
            std::string message;
            std::chrono::milliseconds duration{};
        };

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
    }

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

    auto set_toast_anchor(float x, float y, float width, float height) -> void {
        toast_anchor_viewport.Pos = ImVec2{x, y};
        toast_anchor_viewport.Size = ImVec2{width, height};
        toast_anchored = true;
    }

    auto clear_toast_anchor() -> void { toast_anchored = false; }

    auto render_toasts() -> void {
        std::vector<PendingToast> ready;
        {
            std::scoped_lock const lock{pending_mutex};
            ready.swap(pending_toasts);
        }

        for (auto const &pending: ready) {
            ImGuiToast notification{notify_type(pending.kind), static_cast<int>(pending.duration.count())};
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
}
