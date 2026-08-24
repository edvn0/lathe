#pragma once

#include <btBulletDynamicsCommon.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "buffer.hxx"
#include "pipeline_graph_repository.hxx"

#include <glm/fwd.hpp>

class Renderer;

namespace debug_draw {

    struct Vertex {
        float x, y, z;
        std::uint32_t rgba;
    };

    class DebugRenderer final : public btIDebugDraw {
    public:
        explicit DebugRenderer(Renderer &r);
        ~DebugRenderer() override = default;

        void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override;
        void drawContactPoint(const btVector3 &point_on_b, const btVector3 &normal_on_b, btScalar distance,
                              int /*life_time*/, const btVector3 &color) override;
        void reportErrorWarning(const char *warning_string) override;
        void draw3dText(const btVector3 &location, const char *text_string) override;
        void setDebugMode(int mode) override { debug_mode = mode; }
        int getDebugMode() const override { return debug_mode; }

        auto begin_frame() -> void;

        auto render(VkCommandBuffer cmd, const glm::mat4 &, std::uint32_t frame_index) -> void;
        auto render(VkCommandBuffer cmd, std::span<const float, 16>, std::uint32_t frame_index) -> void;

        void clearLines() override { pending_lines.clear(); }

    private:
        struct FrameBuffer {
            std::unique_ptr<Buffer> vertex;
            std::uint32_t capacity = 0;
        };

        Renderer &renderer;
        int debug_mode = DBG_DrawWireframe;

        std::vector<Vertex> pending_lines;
        std::vector<FrameBuffer> frame_buffers;
        std::uint32_t frame_cursor = 0;

        PipelineNodeHandle pipeline;
        bool force_recompile = true;

        auto ensure_capacity(std::uint32_t frame_index, std::uint32_t vertex_count) -> void;
    };

} // namespace debug_draw
