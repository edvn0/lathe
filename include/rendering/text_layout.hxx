#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <glm/vec2.hpp>

#include "rendering/font.hxx"

namespace text {
    struct PlacedGlyph {
        std::uint32_t glyph = Font::missing_glyph;
        // Where the glyph's pen sits, in em from the top-left of the text, y down. The y is the line's baseline.
        glm::vec2 pen{0.0F};
        // Which font supplied the glyph: 0 is the primary, n is fallbacks[n - 1].
        std::uint32_t font_slot = 0;
    };

    struct Layout {
        std::vector<PlacedGlyph> glyphs;
        // The text's box in em: the widest line, and every line's height.
        glm::vec2 size{0.0F};
    };

    // The code points of UTF-8 text. A malformed sequence becomes U+FFFD, one per bad byte.
    [[nodiscard]] auto decode_utf8(std::string_view utf8) -> std::vector<char32_t>;

    // Characters that take no room and draw nothing: joiners, zero-width spaces and variation selectors, which only
    // change how a neighbour is drawn and which this renderer has no shaping for.
    [[nodiscard]] auto is_invisible(char32_t code_point) noexcept -> bool;

    // Places `utf8` with the pen advancing by each glyph's advance plus the pair's kerning. '\n' starts a new line and
    // '\t' advances four spaces. Each character is drawn with the first of `font` and then `fallbacks` that has it, and
    // a character none of them has is drawn as `font`'s missing-glyph box. Lines are measured with `font`'s metrics.
    [[nodiscard]] auto layout(Font const &font, std::span<Font const *const> fallbacks, std::string_view utf8) -> Layout;

    [[nodiscard]] inline auto layout(Font const &font, std::string_view utf8) -> Layout {
        return layout(font, {}, utf8);
    }
} // namespace text
