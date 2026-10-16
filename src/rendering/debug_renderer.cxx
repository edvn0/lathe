#include "rendering/debug_renderer.hxx"
#include "gpu/upload_ring.hxx"
#include "rendering/overlay_shader.hxx"

#include <btBulletDynamicsCommon.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

namespace debug_draw {

    namespace {

        [[nodiscard]]
        constexpr auto to_byte(float value) noexcept -> std::uint32_t {
            return static_cast<std::uint32_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
        }

        [[nodiscard]]
        constexpr auto pack_rgba(btVector3 const &colour, float alpha = 1.0F) noexcept -> std::uint32_t {
            return to_byte(colour.x()) | (to_byte(colour.y()) << 8) | (to_byte(colour.z()) << 16) |
                   (to_byte(alpha) << 24);
        }

        [[nodiscard]]
        constexpr auto pack_rgba(glm::vec3 const &colour, float alpha = 1.0F) noexcept -> std::uint32_t {
            return to_byte(colour.x) | (to_byte(colour.y) << 8) | (to_byte(colour.z) << 16) | (to_byte(alpha) << 24);
        }

        struct Vertex {
            float x;
            float y;
            float z;
            std::uint32_t rgba;
        };

        struct PC {
            glm::mat4 view_proj;
            VkDeviceAddress vertices;
        };

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
                                  int , btVector3 const &colour) override {
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
            int debug_mode_ = DBG_NoDebug;
        };

    }

    struct DebugRenderer::Impl {
        explicit Impl(Renderer &r)
            : renderer{r},
              vertices{UploadRing::Info{.name = "debug_draw.vertices"}},
              shader{r, OverlayShaderInfo{.source = "assets/shaders/debug_draw.slang",
                                          .debug_name = "debug_draw.pipeline"}},
              bullet_debug_draw{pending_lines} {}

        Renderer &renderer;

        std::vector<Vertex> pending_lines;

        std::vector<Vertex> extra_lines;

        UploadRing vertices;
        OverlayShader shader;

        BulletDebugDraw bullet_debug_draw;
        bool model_bounds_debug_enabled = false;
        bool meshlet_cones_debug_enabled = false;
    };

    DebugRenderer::DebugRenderer(Renderer &renderer) : impl_{std::make_unique<Impl>(renderer)} {}

    DebugRenderer::~DebugRenderer() = default;

    DebugRenderer::DebugRenderer(DebugRenderer &&) noexcept = default;

    auto DebugRenderer::operator=(DebugRenderer &&) noexcept -> DebugRenderer & = default;

    auto DebugRenderer::bullet_debug_draw() noexcept -> btIDebugDraw * { return &impl_->bullet_debug_draw; }

    auto DebugRenderer::begin_frame() -> void { impl_->pending_lines.clear(); }

    auto DebugRenderer::record(OverlayRecordContext const &context) -> void {
        if (impl_->pending_lines.empty() && impl_->extra_lines.empty()) {
            return;
        }

        auto &gpu = impl_->renderer.context();
        auto &ring = impl_->vertices;

        if (auto const begun = ring.begin_frame(gpu, context.frame_index); !begun) {
            error("[DebugDraw] Could not prepare the line buffer: {}", begun.error().message);
            return;
        }

        auto const lines = ring.write(gpu, context.frame_index, std::span<Vertex const>{impl_->pending_lines});
        auto const extra = ring.write(gpu, context.frame_index, std::span<Vertex const>{impl_->extra_lines});
        impl_->extra_lines.clear();

        if (!lines || !extra) {
            error("[DebugDraw] Could not write the line buffer: {}", (!lines ? lines : extra).error().message);
            return;
        }

        if (auto const flushed = ring.flush(context.frame_index); !flushed) {
            error("[DebugDraw] Could not flush the line buffer: {}", flushed.error().message);
            return;
        }

        auto const bound = impl_->shader.bind(context);
        if (!bound) {
            return;
        }

        bound->set_topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        bound->set_line_width(1.0F);

        auto const draw = [&](UploadSlice const &slice) {
            if (slice.empty()) {
                return;
            }
            bound->push(PC{.view_proj = context.view_projection, .vertices = slice.address});
            bound->draw(static_cast<std::uint32_t>(slice.size / sizeof(Vertex)));
        };
        draw(*lines);
        draw(*extra);
    }

    auto DebugRenderer::clear_lines() -> void { impl_->bullet_debug_draw.clearLines(); }

    auto DebugRenderer::add_line(glm::vec3 const &from, glm::vec3 const &to, glm::vec3 const &colour) -> void {
        auto const rgba = pack_rgba(colour);

        impl_->extra_lines.push_back(Vertex{.x = from.x, .y = from.y, .z = from.z, .rgba = rgba});
        impl_->extra_lines.push_back(Vertex{.x = to.x, .y = to.y, .z = to.z, .rgba = rgba});
    }

    auto DebugRenderer::add_aabb(glm::vec3 const &min, glm::vec3 const &max, glm::vec3 const &colour) -> void {
        std::array<glm::vec3, 8> const corners{{
                {min.x, min.y, min.z},
                {max.x, min.y, min.z},
                {max.x, max.y, min.z},
                {min.x, max.y, min.z},
                {min.x, min.y, max.z},
                {max.x, min.y, max.z},
                {max.x, max.y, max.z},
                {min.x, max.y, max.z},
        }};

        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 12> edges{{
                {0, 1},
                {1, 2},
                {2, 3},
                {3, 0},
                {4, 5},
                {5, 6},
                {6, 7},
                {7, 4},
                {0, 4},
                {1, 5},
                {2, 6},
                {3, 7},
        }};

        for (auto const &[a, b]: edges) {
            add_line(corners[a], corners[b], colour);
        }
    }

    auto DebugRenderer::set_physics_debug_enabled(bool enabled) noexcept -> void {
        impl_->bullet_debug_draw.setDebugMode(enabled ? btIDebugDraw::DBG_DrawWireframe : btIDebugDraw::DBG_NoDebug);
    }

    auto DebugRenderer::physics_debug_enabled() const noexcept -> bool {
        return impl_->bullet_debug_draw.getDebugMode() != btIDebugDraw::DBG_NoDebug;
    }

    auto DebugRenderer::set_meshlet_cones_debug_enabled(bool enabled) noexcept -> void {
        impl_->meshlet_cones_debug_enabled = enabled;
    }

    auto DebugRenderer::meshlet_cones_debug_enabled() const noexcept -> bool {
        return impl_->meshlet_cones_debug_enabled;
    }

    auto DebugRenderer::set_model_bounds_debug_enabled(bool enabled) noexcept -> void {
        impl_->model_bounds_debug_enabled = enabled;
    }

    auto DebugRenderer::model_bounds_debug_enabled() const noexcept -> bool {
        return impl_->model_bounds_debug_enabled;
    }

}
