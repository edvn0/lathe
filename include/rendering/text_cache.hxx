#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "rendering/font.hxx"
#include "rendering/text_gpu.hxx"

namespace text {
    using FontId = std::uint32_t;

    // One piece of text to draw this frame.
    struct Draw {
        FontId font = 0;
        std::string text;
        // The top-left of the text, in pixels of the scene target (y down).
        glm::vec2 position{0.0F};
        // The em size, in pixels.
        float size = 16.0F;
        // Linear RGBA.
        glm::vec4 colour{1.0F};
    };

    // One piece of text placed in the world. Its local space reads left to right along +x with its up on +y, the first
    // line's baseline starting at the origin, for a viewer looking along +z; `transform` moves that into the world.
    struct WorldDraw {
        FontId font = 0;
        std::string text;
        glm::mat4 transform{1.0F};
        // The em size, in world units.
        float size = 1.0F;
        // Linear RGBA.
        glm::vec4 colour{1.0F};
        // Multiplies the colour's red, green and blue (see WorldGlyph::intensity).
        float intensity = 1.0F;
        // The origin is the middle of the text, instead of the start of its first baseline.
        bool centred = false;
    };

    // Everything the GPU needs for one frame of text.
    struct Batch {
        // Outlines of the glyphs to rasterise, as float4 (from.xy, to.xy) in em.
        std::vector<glm::vec4> segments;
        std::vector<GlyphJob> jobs;
        std::vector<GlyphInstance> instances;
        std::vector<WorldGlyph> world_instances;
        // One per world draw that has glyphs, in the order WorldGlyph::transform indexes them.
        std::vector<glm::mat4> world_transforms;
        // Texels the jobs' tiles take in all.
        std::uint32_t texel_count = 0;
        // The atlas was full: clear it before placing this batch's jobs, which are then every glyph the frame draws.
        bool reset_atlas = false;
        // Glyphs left for a later frame because this one hit a per-frame limit.
        std::uint32_t deferred_glyphs = 0;

        [[nodiscard]] auto empty() const noexcept -> bool {
            return jobs.empty() && instances.empty() && world_instances.empty();
        }
    };

    // Fonts, and which of their glyphs the atlas holds. Pure CPU: build() decides what the GPU has to rasterise and
    // where each drawn glyph's quad goes, and mirrors the GPU's atlas placement to know when it is full.
    class Cache {
    public:
        // Adds a font and returns the id draws name it by.
        auto add_font(Font font) -> FontId;
        [[nodiscard]] auto font(FontId id) const noexcept -> Font const *;
        [[nodiscard]] auto font_count() const noexcept -> std::size_t { return fonts_.size(); }

        // Fonts tried, in order, for a character the font a draw names does not have: an emoji font, say.
        auto set_fallback_fonts(std::vector<FontId> fonts) -> void { fallbacks_ = std::move(fonts); }

        // The size, in pixels, `utf8` takes drawn at `size`.
        [[nodiscard]] auto measure(FontId id, std::string_view utf8, float size) const -> glm::vec2;

        // Points spread evenly along the outline of `utf8`'s glyphs, `count` of them, in em from the text's origin (the
        // start of the first baseline, or the middle of the text if `centred`), y up. Empty if there is nothing to
        // outline. This is where an emitter on text spawns its particles.
        [[nodiscard]] auto outline_points(FontId id, std::string_view utf8, bool centred, std::uint32_t count) const
                -> std::vector<glm::vec2>;

        // Plans one frame. A glyph the atlas lacks gets a job and a slot; every drawn glyph gets a quad.
        [[nodiscard]] auto build(std::span<Draw const> draws, std::span<WorldDraw const> world_draws = {}) -> Batch;

        // Glyphs the atlas holds (or has been told to hold), blank ones such as spaces included.
        [[nodiscard]] auto glyph_count() const noexcept -> std::size_t { return glyphs_.size(); }
        // How many times the atlas has been cleared because it was full.
        [[nodiscard]] auto evictions() const noexcept -> std::uint32_t { return evictions_; }

    private:
        struct Entry {
            std::uint32_t slot = 0;
            std::uint32_t tile_width = 0;
            std::uint32_t tile_height = 0;
            // The em-space top-left of the tile, y up.
            glm::vec2 origin{0.0F};
            // Nothing to draw: a space, or a glyph too big for a tile.
            bool blank = false;
        };

        [[nodiscard]] auto attempt(std::span<Draw const> draws, std::span<WorldDraw const> world_draws,
                                   bool already_evicted, bool &needs_eviction) -> Batch;
        // The entry for a glyph, creating it (and its job) if the atlas lacks it. Null when it has to wait: this
        // frame's limits are reached, or the atlas must be cleared first (`needs_eviction`).
        [[nodiscard]] auto entry_for(Font const &font, FontId id, std::uint32_t glyph, Batch &batch,
                                     bool already_evicted, bool &needs_eviction) -> Entry const *;
        auto clear() -> void;

        // The fonts a draw's text is laid out with: the one it names, then the fallbacks.
        struct FontChain {
            std::vector<Font const *> fonts;
            std::vector<FontId> ids;
        };

        [[nodiscard]] auto chain_for(FontId id) const -> FontChain;

        std::vector<std::unique_ptr<Font>> fonts_;
        std::vector<FontId> fallbacks_;
        std::unordered_map<std::uint64_t, Entry> glyphs_;
        ShelfPlanner planner_;
        std::uint32_t next_slot_ = 0;
        std::uint32_t evictions_ = 0;
    };

    [[nodiscard]] auto pack_colour(glm::vec4 colour) noexcept -> std::uint32_t;
} // namespace text
