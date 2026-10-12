#pragma once

class Renderer;

namespace gui {

    // One lane per queue with every pass drawn at its measured start and duration, so async overlap is visible.
    auto draw_pass_timeline(Renderer const &renderer) -> void;

}
