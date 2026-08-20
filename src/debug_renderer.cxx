#include "debug_renderer.hxx"

#include <bit>

#include "context.hxx"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "logger.hxx"
#include "renderer.hxx"

#include <cstring>

namespace debug_draw {

    namespace {
        [[nodiscard]] constexpr auto next_power_of_two(std::size_t value) noexcept -> std::size_t {
            if (value <= 1)
                return 1;
            --value;
            for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1)
                value |= value >> shift;
            return value + 1;
        }

        [[nodiscard]] constexpr auto pack_rgba(const btVector3 &c, float alpha = 1.0f) noexcept -> std::uint32_t {
            auto to_byte = [](float v) {
                return static_cast<std::uint32_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            return to_byte(c.x()) | (to_byte(c.y()) << 8) | (to_byte(c.z()) << 16) | (to_byte(alpha) << 24);
        }

        struct PC {
            float view_proj[16];
            VkDeviceAddress vertices;
        };

        auto create_pipeline(Renderer &r, VkFormat fb, VkFormat depth)
                -> std::expected<PipelineNodeHandle, RendererError> {
            VkPushConstantRange const pc_range{
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    .offset = 0,
                    .size = sizeof(PC),
            };

            return r.register_pipeline(PipelineRegisterInfo{
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
                    .push_constant_ranges = {pc_range},
                    .colour_formats = {fb},
                    .depth_format = depth,
                    .stencil_format = VK_FORMAT_UNDEFINED,
                    .samples = r.samples(),
                    .blending = false,
                    .debug_name = "debug_draw.pipeline",
            });
        }
    } // namespace

    DebugRenderer::DebugRenderer(Renderer &r) : renderer(r) {
        frame_buffers.resize(renderer.context().swapchain.frame_count());
    }

    auto DebugRenderer::begin_frame() -> void { pending_lines.clear(); }

    void DebugRenderer::drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) {
        const std::uint32_t rgba = pack_rgba(color);
        pending_lines.push_back(Vertex{from.x(), from.y(), from.z(), rgba});
        pending_lines.push_back(Vertex{to.x(), to.y(), to.z(), rgba});
    }

    void DebugRenderer::drawContactPoint(const btVector3 &point_on_b, const btVector3 &normal_on_b, btScalar distance,
                                         int /*life_time*/, const btVector3 &color) {
        constexpr btScalar length = 0.5;
        drawLine(point_on_b, point_on_b + normal_on_b * length, color);
    }

    void DebugRenderer::reportErrorWarning(const char *warning_string) { warn("(Bullet) {}", warning_string); }

    void DebugRenderer::draw3dText(const btVector3 &, const char *) {}

    auto DebugRenderer::ensure_capacity(std::uint32_t frame_index, std::uint32_t vertex_count) -> void {
        FrameBuffer &fb = frame_buffers[frame_index];
        if (fb.capacity >= vertex_count)
            return;

        const auto size = next_power_of_two(static_cast<std::size_t>(vertex_count) * sizeof(Vertex));
        info("(DebugDraw) Reallocating line buffer to {} bytes", size);

        auto created = Buffer::create(renderer.context(), BufferCreateInfo{
                                                                  .size = size,
                                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                  .memory = BufferMemory::upload,
                                                                  .debug_name = "debug_draw_vertex_buffer",
                                                          });

        if (!created) {
            error("(DebugDraw) Failed to allocate line buffer");
            return;
        }

        fb.vertex = std::make_unique<Buffer>(std::move(*created));
        fb.capacity = static_cast<std::uint32_t>(size / sizeof(Vertex));
    }

    auto DebugRenderer::render(VkCommandBuffer cmd, const glm::mat4 &vp, std::uint32_t frame_index) -> void {
        render(cmd, std::span<const float, 16>(glm::value_ptr(vp), 16), frame_index);
    }
    auto DebugRenderer::render(VkCommandBuffer cmd, const std::span<const float, 16> vp, std::uint32_t frame_index)
            -> void {
        if (pending_lines.empty())
            return;

        if (force_recompile || !pipeline.valid()) {
            auto created = create_pipeline(renderer, renderer.hdr_format(), renderer.depth_format());
            if (!created) {
                error("(DebugDraw) Failed to create pipeline");
                return;
            }
            pipeline = *created;
            force_recompile = false;
        }

        ensure_capacity(frame_index, static_cast<std::uint32_t>(pending_lines.size()));
        FrameBuffer &fb = frame_buffers[frame_index];
        if (!fb.vertex->write(0, std::span<const Vertex>(pending_lines))) {
            error("(DebugDraw) Failed to write line buffer");
            return;
        }

        const auto *p = renderer.resolve_pipeline(pipeline);
        if (!p)
            return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p->pipeline());

        vkCmdSetDepthTestEnable(cmd, VK_TRUE);
        vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
        vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
        vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        vkCmdSetLineWidth(cmd, 1.0F);
        renderer.resource_table().bind(cmd, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, p->layout());

        PC pc{
                .view_proj = {},
                .vertices = fb.vertex->device_address,
        };
        std::memcpy(pc.view_proj, vp.data(), vp.size_bytes());
        pc.vertices = fb.vertex->device_address;

        vkCmdPushConstants(cmd, p->layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                           &pc);
        vkCmdDraw(cmd, static_cast<std::uint32_t>(pending_lines.size()), 1, 0, 0);
    }

} // namespace debug_draw
