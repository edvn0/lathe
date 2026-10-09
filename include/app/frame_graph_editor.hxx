#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "app/stage_registry.hxx"
#include "rendering/frame_graph/layout.hxx"

class Renderer;

namespace gui {

    // Node view of the last distinct frame graph. Read-only for now: it shows passes, the resources that connect
    // them and the stage each pass belongs to.
    class FrameGraphEditor {
    public:
        FrameGraphEditor();
        ~FrameGraphEditor();

        FrameGraphEditor(FrameGraphEditor const &) = delete;
        auto operator=(FrameGraphEditor const &) -> FrameGraphEditor & = delete;
        FrameGraphEditor(FrameGraphEditor &&) = delete;
        auto operator=(FrameGraphEditor &&) -> FrameGraphEditor & = delete;

        auto draw(Renderer const &renderer, stages::Registry const &registry) -> void;

    private:
        struct Context;

        auto relayout(frame_graph::FrameGraphView const &view) -> void;

        std::unique_ptr<Context> context_;
        std::uint64_t laid_out_revision_ = 0;
        frame_graph::GraphLayout layout_;
        std::vector<std::uintptr_t> node_ids_;
        std::unordered_set<std::uintptr_t> placed_;
        int fit_countdown_ = 0;
    };

}
