#pragma once

#include <cstdint>
#include <optional>

#include <glm/vec2.hpp>

// What the text renderer's CPU and GPU halves agree on. The structs are read by assets/shaders/text_*.slang as they
// are, so their layouts are asserted here.
namespace text {
    // The atlas is a square of 8-bit signed distances, one byte per texel, in a persistent buffer.
    inline constexpr std::uint32_t atlas_size = 2048;
    // Glyphs are rasterised at this many pixels per em and drawn at any size by scaling the distance field.
    inline constexpr float raster_em = 64.0F;
    // How far, in atlas texels, the distance field reaches outside a glyph's outline (and inside it).
    inline constexpr std::uint32_t spread = 8;
    // The largest tile one glyph may take, in texels per side; bigger glyphs are not drawn.
    inline constexpr std::uint32_t max_tile = 160;
    // Glyphs the atlas indexes, and how many may be rasterised in one frame (the rest wait for the next).
    inline constexpr std::uint32_t max_slots = 4096;
    inline constexpr std::uint32_t max_new_glyphs_per_frame = 128;
    inline constexpr std::uint32_t max_segments_per_frame = 1U << 18;
    // Glyph quads drawn in one frame, the screen tile the compositor bins them into, and how many quads one
    // screen tile keeps.
    inline constexpr std::uint32_t max_instances = 1U << 14;
    inline constexpr std::uint32_t screen_tile = 16;
    inline constexpr std::uint32_t tile_capacity = 64;

    // A glyph to rasterise into the atlas.
    struct GlyphJob {
        // This glyph's slice of the segment buffer (float4: from.xy, to.xy, in em).
        std::uint32_t segment_offset = 0;
        std::uint32_t segment_count = 0;
        // Its tile, in texels; the width is a multiple of four.
        std::uint32_t tile_width = 0;
        std::uint32_t tile_height = 0;
        // The em-space point the tile's top-left corner sits at (y up).
        float origin_x = 0.0F;
        float origin_y = 0.0F;
        // Which entry of the rectangle table it fills.
        std::uint32_t slot = 0;
        // Where its tile starts in the scratch buffer, in texels. Ascending over a frame's jobs.
        std::uint32_t texel_offset = 0;
    };

    // A glyph to draw: its quad in screen pixels and its colour.
    struct GlyphInstance {
        glm::vec2 min{0.0F};
        glm::vec2 max{0.0F};
        std::uint32_t slot = 0;
        // RGBA, 8 bits each, red in the low byte.
        std::uint32_t colour = 0;
        std::uint32_t padding[2]{};
    };

    // A glyph to draw in the world: its quad in the text's local space (em units, y up) and which transform places it.
    struct WorldGlyph {
        glm::vec2 min{0.0F};
        glm::vec2 max{0.0F};
        std::uint32_t slot = 0;
        std::uint32_t colour = 0;
        // Index into the frame's array of transforms.
        std::uint32_t transform = 0;
        // Multiplies the colour's red, green and blue: above 1 the text is brighter than white, which is what makes
        // it glow once bloom has had it.
        float intensity = 1.0F;
    };

    static_assert(sizeof(GlyphJob) == 32);
    static_assert(sizeof(GlyphInstance) == 32);
    static_assert(sizeof(WorldGlyph) == 32);

    struct AtlasRect {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    // Shelf packing: tiles go left to right along a shelf, and a tile that does not fit starts the next one. The
    // text_place shader runs the same steps on the GPU; this copy tells the CPU when the atlas is full without a
    // read-back, and the tests hold the two to the same rules.
    class ShelfPlanner {
    public:
        explicit ShelfPlanner(std::uint32_t size = atlas_size) noexcept : size_{size} {}

        // Where a width x height tile goes, or nothing (leaving the planner as it was) if it does not fit.
        [[nodiscard]] auto place(std::uint32_t width, std::uint32_t height) noexcept -> std::optional<AtlasRect> {
            auto x = cursor_x_;
            auto y = cursor_y_;
            auto shelf = shelf_height_;

            if (x + width > size_) {
                y += shelf;
                x = 0;
                shelf = 0;
            }
            if (y + height > size_ || width > size_) {
                return std::nullopt;
            }

            cursor_x_ = x + width;
            cursor_y_ = y;
            shelf_height_ = shelf > height ? shelf : height;
            return AtlasRect{.x = x, .y = y, .width = width, .height = height};
        }

        auto reset() noexcept -> void { *this = ShelfPlanner{size_}; }

        [[nodiscard]] auto size() const noexcept -> std::uint32_t { return size_; }
        [[nodiscard]] auto cursor_x() const noexcept -> std::uint32_t { return cursor_x_; }
        [[nodiscard]] auto cursor_y() const noexcept -> std::uint32_t { return cursor_y_; }
        [[nodiscard]] auto shelf_height() const noexcept -> std::uint32_t { return shelf_height_; }

    private:
        std::uint32_t size_;
        std::uint32_t cursor_x_ = 0;
        std::uint32_t cursor_y_ = 0;
        std::uint32_t shelf_height_ = 0;
    };
} // namespace text
