#pragma once

#include "core/gpu_timings.hxx"

namespace gui {

    struct GpuTimingsViewState {
        bool paused = false;
        // What the table shows while paused.
        GpuTimings frozen;
        float frozen_frame_ms = 0.0F;
    };

    // GPU pass table: pass, smoothed ms and share of the total, with a Total row; idle passes are dimmed. Hovering a
    // pass shows its last/min/max and queue. `frame_ms` is the measured whole-GPU-frame time; 0 means unknown.
    auto draw_gpu_timings(GpuTimings const &live, float frame_ms, GpuTimingsViewState &state) -> void;

}
