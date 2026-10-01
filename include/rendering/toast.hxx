#pragma once

namespace gui {
    // Draws the queued toasts. Once per frame, inside the ImGui frame.
    auto render_toasts() -> void;
} // namespace gui
