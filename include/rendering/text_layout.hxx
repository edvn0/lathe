#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <glm/vec2.hpp>

#include "rendering/font.hxx"

namespace text {
    struct PlacedGlyph {
        std::uint32_t glyph = Font::missing_glyph;
        // Where the glyph's pen sits, in em from the top-left of the text, y down. The y is the line's baseline.
        glm::vec2 pen{0.0F};
    };

    struct Layout {
        std::vector<PlacedGlyph> glyphs;
        // The text's box in em: the widest line, and every line's height.
        glm::vec2 size{0.0F};
    };

    // The code points of UTF-8 text. A malformed sequence becomes U+FFFD, one per bad byte.
    [[nodiscard]] auto decode_utf8(std::string_view utf8) -> std::vector<char32_t>;

    // Places `utf8` with the pen advancing by each glyph's advance plus the pair's kerning. '\n' starts a new line and
    // '\t' advances four spaces. Glyphs the font lacks are drawn as its missing-glyph box.
    [[nodiscard]] auto layout(Font const &font, std::string_view utf8) -> Layout;
} // namespace text
