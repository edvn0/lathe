#include "rendering/toast.hxx"

// ImGuiNotify.hpp uses these without including them.
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <imgui.h>

#include "imgui_internal.h"

// The monitor-corner placement reads ImGuiPlatformIO::Monitors, which is only filled with multi-viewport support.
#define NOTIFY_RENDER_OUTSIDE_MAIN_WINDOW false
#include "ImGuiNotify.hpp"

namespace gui {
    auto render_toasts() -> void { ImGui::RenderNotifications(); }
} // namespace gui
