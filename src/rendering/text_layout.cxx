#include "rendering/text_layout.hxx"

#include <algorithm>

namespace text {
    auto decode_utf8(std::string_view utf8) -> std::vector<char32_t> {
        constexpr auto replacement = char32_t{0xFFFD};

        auto result = std::vector<char32_t>{};
        result.reserve(utf8.size());

        auto i = std::size_t{0};
        while (i < utf8.size()) {
            auto const lead = static_cast<unsigned char>(utf8[i]);
            auto length = std::size_t{1};
            auto code_point = char32_t{lead};

            if (lead >= 0xF0 && lead <= 0xF4) {
                length = 4;
                code_point = lead & 0x07U;
            } else if (lead >= 0xE0 && lead < 0xF0) {
                length = 3;
                code_point = lead & 0x0FU;
            } else if (lead >= 0xC2 && lead < 0xE0) {
                length = 2;
                code_point = lead & 0x1FU;
            } else if (lead >= 0x80) {
                result.push_back(replacement);
                ++i;
                continue;
            }

            auto valid = i + length <= utf8.size();
            for (auto k = std::size_t{1}; valid && k < length; ++k) {
                auto const next = static_cast<unsigned char>(utf8[i + k]);
                valid = (next & 0xC0U) == 0x80U;
                code_point = (code_point << 6U) | (next & 0x3FU);
            }

            // Overlong forms, surrogates and anything past U+10FFFF are not text.
            auto const minimum = length == 2 ? 0x80U : length == 3 ? 0x800U : length == 4 ? 0x10000U : 0U;
            valid = valid && code_point >= minimum && code_point <= 0x10FFFF &&
                    !(code_point >= 0xD800 && code_point <= 0xDFFF);

            if (valid) {
                result.push_back(code_point);
                i += length;
            } else {
                result.push_back(replacement);
                ++i;
            }
        }

        return result;
    }

    auto layout(Font const &font, std::string_view utf8) -> Layout {
        auto result = Layout{};
        auto const &metrics = font.metrics();

        auto pen_x = 0.0F;
        auto line = 0;
        auto previous = Font::missing_glyph;
        auto widest = 0.0F;

        auto const space = font.glyph_index(U' ');
        auto const space_advance = font.advance(space);

        for (auto const code_point: decode_utf8(utf8)) {
            if (code_point == U'\n') {
                widest = std::max(widest, pen_x);
                pen_x = 0.0F;
                ++line;
                previous = Font::missing_glyph;
                continue;
            }
            if (code_point == U'\r') {
                continue;
            }
            if (code_point == U'\t') {
                pen_x += 4.0F * space_advance;
                previous = Font::missing_glyph;
                continue;
            }

            auto const glyph = font.glyph_index(code_point);

            if (previous != Font::missing_glyph) {
                pen_x += font.kerning(previous, glyph);
            }

            result.glyphs.push_back(PlacedGlyph{
                    .glyph = glyph,
                    .pen = {pen_x, metrics.ascent + static_cast<float>(line) * metrics.line_height},
            });

            pen_x += font.advance(glyph);
            previous = glyph;
        }

        widest = std::max(widest, pen_x);
        result.size = {widest, static_cast<float>(line + 1) * metrics.line_height};

        return result;
    }
} // namespace text
