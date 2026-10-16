#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <entt/fwd.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "core/renderer_error.hxx"
#include "gpu/upload_ring.hxx"
#include "rendering/game_gpu.hxx"
#include "rendering/game_graph.hxx"
#include "rendering/text_cache.hxx"
#include "scene/components.hxx"

struct Renderer;

// Draws text with fonts read through FreeType, entirely on the GPU. A glyph's outline is turned into a signed
// distance field by a compute shader, a second one shelf-packs the new glyphs into a persistent atlas, and the atlas
// is then drawn two ways: into the scene image by a compute compositor (screen-space text), or as world-space quads
// in the forward pass (text with a transform, depth-tested like any other transparent object).
//
// It is an immediate-mode renderer like the debug and ImGui ones: draw() and draw_world() queue what to show this
// frame, and the frame graph hook (declare) consumes the queue. Anything may call the draw functions from any
// thread. Because a distance field scales, one rasterisation at 64 px per em serves every size.
class TextRenderer {
public:
    // The font draws use when they name none: the engine's own, loaded by create().
    static constexpr text::FontId default_font = 0;

    // Registers the shaders and loads the built-in font. Safe to call again.
    auto create(Renderer &renderer, GameGpu &gpu) -> std::expected<void, RendererError>;

    // Reads a TrueType or OpenType font (relative to the data directory unless absolute).
    [[nodiscard]] auto load_font(std::filesystem::path const &path) -> std::expected<text::FontId, std::string>;

    // Queues text for this frame. `position` is the top-left, in pixels of the scene image.
    auto draw(text::Draw draw) -> void;
    // Queues text placed in the world: see text::WorldDraw for the space the transform acts on.
    auto draw_world(text::WorldDraw draw) -> void;

    // The size, in pixels, `utf8` takes drawn at `size`.
    [[nodiscard]] auto measure(text::FontId font, std::string_view utf8, float size) -> glm::vec2;

    // The frame graph hook. frame_start builds the frame's batch, rasterises and packs new glyphs and declares the
    // world-space draw; before_composite declares the screen-space compositor. Other slots do nothing. Every
    // Components::Text in `registry` (if there is one) is drawn along with what draw() and draw_world() queued, placed
    // by its entity's world transform.
    auto declare(GameGraph &graph, entt::registry const *registry = nullptr) -> void;

    // Points per outline an emitter draws its spawn positions from.
    static constexpr std::uint32_t emission_point_count = 1024;

    // Where a particle emitter can spawn from text.
    struct EmissionPoints {
        // float2 each, in the text's local space in world units (see ParticleShape::text).
        VkDeviceAddress address = 0;
        std::uint32_t count = 0;
    };

    // The outline of `text` as spawn positions, uploaded for this frame's simulation. Valid from the frame_start
    // declare of this frame on, which is where the particle system asks; nothing if the text has no outline.
    [[nodiscard]] auto emission_points(Components::Text const &text) -> std::optional<EmissionPoints>;

    // What the last frame's batch held, for tests and tools.
    struct Stats {
        std::uint32_t new_glyphs = 0;
        std::uint32_t screen_glyphs = 0;
        std::uint32_t world_glyphs = 0;
        bool atlas_reset = false;
    };
    [[nodiscard]] auto stats() const noexcept -> Stats { return stats_; }

private:
    struct FrameState {
        // Valid from frame_start to before_composite of one frame.
        bool screen_text = false;
        UploadSlice instances;
        std::uint32_t instance_count = 0;
    };

    auto declare_glyphs_and_world(GameGraph &graph, entt::registry const *registry) -> void;
    auto declare_screen(GameGraph &graph) -> void;

    Renderer *renderer_ = nullptr;
    GameGpu *gpu_ = nullptr;

    GameComputeShader raster_{};
    GameComputeShader place_{};
    GameComputeShader blit_{};
    GameComputeShader bin_{};
    GameComputeShader composite_{};
    GameGraphicsShader world_{};
    bool ready_ = false;

    std::mutex mutex_;
    text::Cache cache_;
    std::vector<text::Draw> pending_;
    std::vector<text::WorldDraw> pending_world_;

    UploadRing upload_{UploadRing::Info{.name = "text.upload"}};
    // The frame the upload ring is open for, so emission_points can write to it after the frame_start declare.
    bool upload_open_ = false;
    std::uint32_t upload_slot_ = 0;
    // Outline points by the text they came from, so an emitter does not trace the outline again every frame.
    std::unordered_map<std::string, std::vector<glm::vec2>> points_;
    FrameState frame_;
    Stats stats_;
};
