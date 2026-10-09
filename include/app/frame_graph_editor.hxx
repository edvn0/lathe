#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "app/stage_registry.hxx"
#include "rendering/frame_graph/layout.hxx"

class Renderer;

namespace gui {

    // Node view of the last distinct frame graph. Passes belonging to a registered stage collapse into one node
    // that hosts the stage's settings; stages the graph currently lacks appear as ghost nodes so they can be
    // switched back on.
    class FrameGraphEditor {
    public:
        FrameGraphEditor();
        ~FrameGraphEditor();

        FrameGraphEditor(FrameGraphEditor const &) = delete;
        auto operator=(FrameGraphEditor const &) -> FrameGraphEditor & = delete;
        FrameGraphEditor(FrameGraphEditor &&) = delete;
        auto operator=(FrameGraphEditor &&) -> FrameGraphEditor & = delete;

        auto draw(Renderer &renderer, stages::Registry &registry) -> void;

    private:
        struct Context;

        struct Ghost {
            std::size_t stage = 0;
            std::uintptr_t key = 0;
            float x = 0.0F;
            float y = 0.0F;
        };

        auto relayout(frame_graph::FrameGraphView const &view, stages::Registry const &registry) -> void;
        auto reset_layout() -> void;

        std::unique_ptr<Context> context_;
        std::uint64_t laid_out_revision_ = 0;
        bool expanded_ = false;
        bool layout_dirty_ = true;
        frame_graph::GraphLayout layout_;
        std::vector<std::uintptr_t> node_keys_;
        std::vector<std::uint32_t> pass_barriers_;
        std::vector<Ghost> ghosts_;
        std::unordered_set<std::uintptr_t> placed_;
        std::unordered_map<std::uintptr_t, float> measured_height_;
        int settle_countdown_ = 0;
        int fit_countdown_ = 0;
    };

}
