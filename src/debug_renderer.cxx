#include "debug_renderer.hxx"

#include <btBulletDynamicsCommon.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "buffer.hxx"
#include "context.hxx"
#include "logger.hxx"
#include "pipeline_graph_repository.hxx"
#include "renderer.hxx"

namespace debug_draw {

    namespace {

        [[nodiscard]]
        constexpr auto next_power_of_two(std::size_t value) noexcept -> std::size_t {
            if (value <= 1) {
                return 1;
            }

            --value;

            for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
                value |= value >> shift;
            }

            return value + 1;
        }

        [[nodiscard]]
        constexpr auto pack_rgba(btVector3 const &colour, float alpha = 1.0F) noexcept -> std::uint32_t {
            auto const to_byte = [](float value) -> std::uint32_t {
                return static_cast<std::uint32_t>(std::clamp(value, 0.0F, 1.0F) * 255.0F + 0.5F);
            };

            return to_byte(colour.x()) | (to_byte(colour.y()) << 8) | (to_byte(colour.z()) << 16) |
                   (to_byte(alpha) << 24);
        }

        struct Vertex {
            float x;
            float y;
            float z;
            std::uint32_t rgba;
        };

        struct PC {
            float view_proj[16];
            VkDeviceAddress vertices;
        };

        auto create_pipeline(Renderer &renderer, VkFormat framebuffer_format, VkFormat depth_format)
                -> std::expected<PipelineNodeHandle, RendererError> {
            return renderer.register_pipeline(PipelineRegisterInfo{
                    .stages =
                            {
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/debug_draw.slang",
                                            .entry_point = "vs_main",
                                            .stage = renderer::ShaderStage::vertex,
                                    },
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/debug_draw.slang",
                                            .entry_point = "fs_main",
                                            .stage = renderer::ShaderStage::fragment,
                                    },
                            },
                    .push_constant_ranges = {global_push_constant_range},
                    .colour_formats = {framebuffer_format},
                    .dynamic_states = {},
                    .depth_format = depth_format,
                    .stencil_format = VK_FORMAT_UNDEFINED,
                    .samples = renderer.samples(),
                    .blending = false,
                    .debug_name = "debug_draw.pipeline",
            });
        }

        class BulletDebugDraw final : public btIDebugDraw {
        public:
            explicit BulletDebugDraw(std::vector<Vertex> &pending_lines) noexcept : pending_lines_{pending_lines} {}

            void drawLine(btVector3 const &from, btVector3 const &to, btVector3 const &colour) override {
                auto const rgba = pack_rgba(colour);

                pending_lines_.push_back(Vertex{
                        .x = from.x(),
                        .y = from.y(),
                        .z = from.z(),
                        .rgba = rgba,
                });

                pending_lines_.push_back(Vertex{
                        .x = to.x(),
                        .y = to.y(),
                        .z = to.z(),
                        .rgba = rgba,
                });
            }

            void drawContactPoint(btVector3 const &point_on_b, btVector3 const &normal_on_b, btScalar distance,
                                  int /*life_time*/, btVector3 const &colour) override {
                drawLine(point_on_b, point_on_b + normal_on_b * distance, colour);

                constexpr auto marker_radius = btScalar{0.05F};

                drawLine(point_on_b - btVector3{marker_radius, 0, 0}, point_on_b + btVector3{marker_radius, 0, 0},
                         colour);

                drawLine(point_on_b - btVector3{0, marker_radius, 0}, point_on_b + btVector3{0, marker_radius, 0},
                         colour);

                drawLine(point_on_b - btVector3{0, 0, marker_radius}, point_on_b + btVector3{0, 0, marker_radius},
                         colour);
            }

            void reportErrorWarning(char const *warning_string) override { warn("(Bullet) {}", warning_string); }

            void draw3dText(btVector3 const &, char const *) override {}

            void setDebugMode(int mode) override { debug_mode_ = mode; }

            [[nodiscard]]
            auto getDebugMode() const -> int override {
                return debug_mode_;
            }

            void clearLines() override { pending_lines_.clear(); }

        private:
            std::vector<Vertex> &pending_lines_;
            int debug_mode_ = DBG_DrawWireframe;
        };

    } // namespace

    struct DebugRenderer::Impl {
        struct FrameBuffer {
            std::unique_ptr<Buffer> vertex;
            std::uint32_t capacity = 0;
        };

        explicit Impl(Renderer &r) : renderer{r}, bullet_debug_draw{pending_lines} {
            frame_buffers.resize(renderer.context().swapchain.frame_count());
        }

        [[nodiscard]]
        auto ensure_capacity(std::uint32_t frame_index, std::uint32_t vertex_count) -> bool {
            if (frame_index >= frame_buffers.size()) {
                error("[DebugDraw] Invalid frame index {} for {} frame buffers", frame_index, frame_buffers.size());
                return false;
            }

            auto &frame_buffer = frame_buffers[frame_index];

            if (frame_buffer.vertex != nullptr && frame_buffer.capacity >= vertex_count) {
                return true;
            }

            auto const size = next_power_of_two(static_cast<std::size_t>(vertex_count) * sizeof(Vertex));

            info("[DebugDraw] Reallocating line buffer to {} bytes", size);

            auto created =
                    Buffer::create(renderer.context(), BufferCreateInfo{
                                                               .size = size,
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                               .memory = BufferMemory::upload,
                                                               .debug_name = "debug_draw_vertex_buffer",
                                                       });

            if (!created) {
                error("[DebugDraw] Failed to allocate line buffer");
                return false;
            }

            frame_buffer.vertex = std::make_unique<Buffer>(std::move(*created));

            frame_buffer.capacity = static_cast<std::uint32_t>(size / sizeof(Vertex));

            return true;
        }

        Renderer &renderer;

        std::vector<Vertex> pending_lines;
        std::vector<FrameBuffer> frame_buffers;

        PipelineNodeHandle pipeline;
        bool force_recompile = true;

        BulletDebugDraw bullet_debug_draw;
    };

    DebugRenderer::DebugRenderer(Renderer &renderer) : impl_{std::make_unique<Impl>(renderer)} {}

    DebugRenderer::~DebugRenderer() = default;

    DebugRenderer::DebugRenderer(DebugRenderer &&) noexcept = default;

    auto DebugRenderer::operator=(DebugRenderer &&) noexcept -> DebugRenderer & = default;

    auto DebugRenderer::bullet_debug_draw() noexcept -> btIDebugDraw * { return &impl_->bullet_debug_draw; }

    auto DebugRenderer::begin_frame() -> void { impl_->pending_lines.clear(); }

    auto DebugRenderer::render(VkCommandBuffer cmd, glm::mat4 const &view_projection, std::uint32_t frame_index)
            -> void {
        render(cmd, std::span<const float, 16>{glm::value_ptr(view_projection), 16}, frame_index);
    }

    auto DebugRenderer::render(VkCommandBuffer cmd, std::span<const float, 16> view_projection,
                               std::uint32_t frame_index) -> void {
        if (impl_->pending_lines.empty()) {
            return;
        }

        if (impl_->force_recompile || !impl_->pipeline.valid()) {
            auto created =
                    create_pipeline(impl_->renderer, impl_->renderer.hdr_format(), impl_->renderer.depth_format());

            if (!created) {
                error("[DebugDraw] Failed to create pipeline");
                return;
            }

            impl_->pipeline = *created;
            impl_->force_recompile = false;
        }

        auto const vertex_count = static_cast<std::uint32_t>(impl_->pending_lines.size());

        if (!impl_->ensure_capacity(frame_index, vertex_count)) {
            return;
        }

        auto &frame_buffer = impl_->frame_buffers[frame_index];

        if (!frame_buffer.vertex->write(0, std::span<const Vertex>{impl_->pending_lines})) {
            error("[DebugDraw] Failed to write line buffer");
            return;
        }

        auto const *pipeline = impl_->renderer.resolve_pipeline(impl_->pipeline);

        if (pipeline == nullptr) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline());

        auto const &extent = impl_->renderer.context().swapchain.extent();

        auto const viewport = VkViewport{
                .x = 0.0F,
                .y = 0.0F,
                .width = static_cast<float>(extent.width),
                .height = static_cast<float>(extent.height),
                .minDepth = 0.0F,
                .maxDepth = 1.0F,
        };

        auto const scissor = VkRect2D{
                .offset = {0, 0},
                .extent =
                        {
                                static_cast<std::uint32_t>(extent.width),
                                static_cast<std::uint32_t>(extent.height),
                        },
        };

        vkCmdSetViewportWithCount(cmd, 1, &viewport);

        vkCmdSetScissorWithCount(cmd, 1, &scissor);

        vkCmdSetDepthTestEnable(cmd, VK_TRUE);

        vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

        vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);

        vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

        vkCmdSetLineWidth(cmd, 1.0F);

        impl_->renderer.resource_table().bind(cmd, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->layout());

        auto push_constants = PC{
                .view_proj = {},
                .vertices = frame_buffer.vertex->device_address,
        };

        std::memcpy(push_constants.view_proj, view_projection.data(), view_projection.size_bytes());

        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push_constants), &push_constants);

        vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    }

    auto DebugRenderer::clear_lines() -> void { impl_->bullet_debug_draw.clearLines(); }

} // namespace debug_draw
