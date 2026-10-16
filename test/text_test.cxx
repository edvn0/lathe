#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <string>

#include "rendering/font.hxx"
#include "rendering/text_cache.hxx"
#include "rendering/text_gpu.hxx"
#include "rendering/text_layout.hxx"

namespace {
    auto font_path() -> std::filesystem::path {
        return std::filesystem::path{TEST_ASSETS_DIR} / "assets" / "fonts" / "GoogleSansCode-Regular.ttf";
    }

    auto load_font() -> Font {
        auto font = Font::load(font_path());
        REQUIRE(font.has_value());
        return std::move(*font);
    }

    // The winding number of an outline around a point: non-zero is inside. The same test the rasteriser shader runs.
    auto winding(GlyphOutline const &outline, glm::vec2 p) -> int {
        auto result = 0;
        for (auto const &s: outline.segments) {
            if ((s.from.y <= p.y) != (s.to.y <= p.y)) {
                auto const x = s.from.x + (p.y - s.from.y) / (s.to.y - s.from.y) * (s.to.x - s.from.x);
                if (x > p.x) {
                    result += s.to.y > s.from.y ? 1 : -1;
                }
            }
        }
        return result;
    }

    auto utf8(std::initializer_list<unsigned char> bytes) -> std::string {
        return std::string{bytes.begin(), bytes.end()};
    }
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("a font gives metrics, glyph indices and advances") {
        auto const font = load_font();

        CHECK(font.metrics().ascent > 0.0F);
        CHECK(font.metrics().descent < 0.0F);
        CHECK(font.metrics().line_height > font.metrics().ascent);

        auto const a = font.glyph_index(U'A');
        CHECK(a != Font::missing_glyph);
        CHECK(font.glyph_index(U'a') != a);
        CHECK(font.advance(a) > 0.1F);
        CHECK(font.advance(a) < 1.5F);
    }

    TEST_CASE("a glyph outline is closed, inside its bounds and flattens finer with a smaller tolerance") {
        auto const font = load_font();
        auto const o = font.glyph_index(U'O');

        auto const coarse = font.outline(o, 1.0F / 64.0F);
        auto const fine = font.outline(o, 1.0F / 4096.0F);
        REQUIRE(coarse.has_value());
        REQUIRE(fine.has_value());

        CHECK(!fine->empty());
        CHECK(fine->segments.size() > coarse->segments.size());
        CHECK(fine->max.x > fine->min.x);
        CHECK(fine->max.y > fine->min.y);
        CHECK(fine->min.y > -0.3F);
        CHECK(fine->max.y < 1.3F);

        // Every contour returns to where it began, so the segments chain: each starts where the one before ended,
        // except at a contour's first segment.
        auto breaks = 0;
        for (auto i = std::size_t{1}; i < fine->segments.size(); ++i) {
            if (glm::distance(fine->segments[i - 1].to, fine->segments[i].from) > 1.0e-5F) {
                ++breaks;
            }
        }
        // 'O' has an outer contour and an inner one.
        CHECK(breaks == 1);

        // Inside the ring is inside; the counter (hole) and the outside are not.
        auto const centre = (fine->min + fine->max) * 0.5F;
        CHECK(winding(*fine, centre) == 0);
        CHECK(winding(*fine, {fine->min.x - 0.1F, centre.y}) == 0);
        auto ring = false;
        for (auto x = fine->min.x; x < centre.x; x += 0.005F) {
            ring = ring || winding(*fine, {x, centre.y}) != 0;
        }
        CHECK(ring);
    }

    TEST_CASE("a space has an advance and no outline") {
        auto const font = load_font();
        auto const space = font.glyph_index(U' ');

        auto const outline = font.outline(space);
        REQUIRE(outline.has_value());
        CHECK(outline->empty());
        CHECK(font.advance(space) > 0.0F);
    }

    TEST_CASE("a font that is not a font is refused") {
        auto const bad = Font::load_memory(std::vector<std::byte>(64, std::byte{7}), "junk");
        CHECK(!bad.has_value());
        CHECK(!Font::load("no/such/font.ttf").has_value());
    }

    TEST_CASE("UTF-8 decodes to code points, and what is not UTF-8 becomes U+FFFD") {
        using text::decode_utf8;

        CHECK(decode_utf8("Hi") == std::vector<char32_t>{U'H', U'i'});
        CHECK(decode_utf8(utf8({0xC3, 0xA5})) == std::vector<char32_t>{U'å'});
        CHECK(decode_utf8(utf8({0xE2, 0x82, 0xAC})) == std::vector<char32_t>{U'€'});
        CHECK(decode_utf8(utf8({0xF0, 0x9F, 0x98, 0x80})) == std::vector<char32_t>{U'\U0001F600'});

        // A stray continuation byte, a truncated sequence, an overlong 'A', a surrogate and a lead past U+10FFFF.
        CHECK(decode_utf8(utf8({0x80})) == std::vector<char32_t>{U'�'});
        CHECK(decode_utf8(utf8({0xE2, 0x82})) == std::vector<char32_t>{U'�', U'�'});
        CHECK(decode_utf8(utf8({0xC1, 0x81})).front() == U'�');
        CHECK(decode_utf8(utf8({0xED, 0xA0, 0x80})).front() == U'�');
        CHECK(decode_utf8(utf8({0xF5, 0x80, 0x80, 0x80})).front() == U'�');
        CHECK(decode_utf8("").empty());
    }

    TEST_CASE("layout advances the pen, breaks lines and measures the widest") {
        auto const font = load_font();
        auto const a = font.glyph_index(U'A');
        auto const b = font.glyph_index(U'B');

        auto const one = text::layout(font, "AB");
        REQUIRE(one.glyphs.size() == 2);
        CHECK(one.glyphs[0].glyph == a);
        CHECK(one.glyphs[0].pen.x == 0.0F);
        CHECK(one.glyphs[0].pen.y == doctest::Approx(font.metrics().ascent));
        CHECK(one.glyphs[1].pen.x == doctest::Approx(font.advance(a) + font.kerning(a, b)));
        CHECK(one.size.x == doctest::Approx(one.glyphs[1].pen.x + font.advance(b)));
        CHECK(one.size.y == doctest::Approx(font.metrics().line_height));

        auto const two = text::layout(font, "A\nBBB");
        REQUIRE(two.glyphs.size() == 4);
        CHECK(two.glyphs[1].pen.x == 0.0F);
        CHECK(two.glyphs[1].pen.y == doctest::Approx(font.metrics().ascent + font.metrics().line_height));
        CHECK(two.size.y == doctest::Approx(2.0F * font.metrics().line_height));
        CHECK(two.size.x == doctest::Approx(text::layout(font, "BBB").size.x));

        auto const tabbed = text::layout(font, "\tA");
        REQUIRE(tabbed.glyphs.size() == 1);
        CHECK(tabbed.glyphs[0].pen.x == doctest::Approx(4.0F * font.advance(font.glyph_index(U' '))));

        CHECK(text::layout(font, "").glyphs.empty());
    }

    TEST_CASE("shelf planning fills a row, starts the next, and never overlaps or leaves the atlas") {
        auto planner = text::ShelfPlanner{64};

        auto const first = planner.place(40, 10);
        auto const second = planner.place(20, 12);
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        CHECK(first->x == 0);
        CHECK(second->x == 40);
        CHECK(second->y == 0);

        // Does not fit beside them: the next shelf, below the tallest tile of the first.
        auto const wrapped = planner.place(30, 8);
        REQUIRE(wrapped.has_value());
        CHECK(wrapped->x == 0);
        CHECK(wrapped->y == 12);

        // A refusal changes nothing.
        auto const before_x = planner.cursor_x();
        auto const before_y = planner.cursor_y();
        CHECK(!planner.place(65, 4).has_value());
        CHECK(!planner.place(4, 200).has_value());
        CHECK(planner.cursor_x() == before_x);
        CHECK(planner.cursor_y() == before_y);

        planner.reset();
        CHECK(planner.place(64, 64).has_value());
        CHECK(!planner.place(4, 4).has_value());
    }

    TEST_CASE("shelf planning keeps many tiles apart") {
        auto planner = text::ShelfPlanner{256};
        auto placed = std::vector<text::AtlasRect>{};
        auto seed = std::uint32_t{12345};
        auto const next = [&](std::uint32_t modulo) {
            seed = seed * 1664525U + 1013904223U;
            return (seed >> 8U) % modulo;
        };

        for (auto i = 0; i < 400; ++i) {
            if (auto const rect = planner.place(4 + 4 * next(10), 4 + next(40))) {
                CHECK(rect->x + rect->width <= 256);
                CHECK(rect->y + rect->height <= 256);
                for (auto const &other: placed) {
                    auto const apart = rect->x + rect->width <= other.x || other.x + other.width <= rect->x ||
                                       rect->y + rect->height <= other.y || other.y + other.height <= rect->y;
                    CHECK(apart);
                }
                placed.push_back(*rect);
            }
        }
        CHECK(placed.size() > 20);
    }

    TEST_CASE("colours pack as 8-bit RGBA with red lowest, clamped, and a non-finite channel is zero") {
        CHECK(text::pack_colour({1.0F, 0.5F, 0.0F, 1.0F}) == 0xFF0080FFU);
        CHECK(text::pack_colour({2.0F, -1.0F, 0.0F, 0.0F}) == 0x000000FFU);
        CHECK(text::pack_colour({std::nanf(""), 0.0F, 0.0F, 1.0F}) == 0xFF000000U);
    }

    TEST_CASE("the cache rasterises each glyph once, skips blanks and quads every drawn glyph") {
        text::Cache cache;
        auto const id = cache.add_font(load_font());

        auto const draws = std::vector{text::Draw{.font = id, .text = "Hello Hello", .position = {10, 20}, .size = 32}};

        auto const first = cache.build(draws);
        // H, e, l, o: four distinct glyphs; the space and the repeats need nothing new.
        CHECK(first.jobs.size() == 4);
        // Ten glyphs are drawn: all but the space.
        CHECK(first.instances.size() == 10);
        CHECK(!first.reset_atlas);
        CHECK(first.deferred_glyphs == 0);

        auto offset = std::uint32_t{0};
        auto slots = std::set<std::uint32_t>{};
        for (auto const &job: first.jobs) {
            CHECK(job.tile_width % 4 == 0);
            CHECK(job.tile_width <= text::max_tile);
            CHECK(job.tile_height <= text::max_tile);
            CHECK(job.texel_offset == offset);
            CHECK(job.segment_count > 0);
            CHECK(job.segment_offset + job.segment_count <= first.segments.size());
            offset += job.tile_width * job.tile_height;
            CHECK(slots.insert(job.slot).second);
        }
        CHECK(first.texel_count == offset);

        // Drawing it again costs nothing on the GPU, and the quads are the same.
        auto const again = cache.build(draws);
        CHECK(again.jobs.empty());
        CHECK(again.instances.size() == first.instances.size());
        CHECK(again.texel_count == 0);
        CHECK(cache.glyph_count() == 5);
    }

    TEST_CASE("a screen quad scales with the size and sits at the position") {
        text::Cache cache;
        auto const id = cache.add_font(load_font());

        auto const small =
                cache.build(std::vector{text::Draw{.font = id, .text = "I", .position = {100, 100}, .size = 20}});
        auto const large =
                cache.build(std::vector{text::Draw{.font = id, .text = "I", .position = {100, 100}, .size = 40}});
        REQUIRE(small.instances.size() == 1);
        REQUIRE(large.instances.size() == 1);

        auto const small_size = small.instances[0].max - small.instances[0].min;
        auto const large_size = large.instances[0].max - large.instances[0].min;
        CHECK(large_size.x == doctest::Approx(2.0F * small_size.x));
        CHECK(large_size.y == doctest::Approx(2.0F * small_size.y));
        // The quad is near its position, a little above and left because of the distance field's padding.
        CHECK(large.instances[0].min.x < 100.0F + 40.0F);
        CHECK(large.instances[0].max.y > 100.0F);
        CHECK(large.instances[0].colour == 0xFFFFFFFFU);
    }

    TEST_CASE("world text shares the atlas, folds the size into the transform and keeps glyphs above the baseline") {
        text::Cache cache;
        auto const id = cache.add_font(load_font());

        auto transform = glm::mat4{1.0F};
        transform[3] = glm::vec4{1.0F, 2.0F, 3.0F, 1.0F};

        auto const batch = cache.build(
                {}, std::vector{text::WorldDraw{.font = id, .text = "AA", .transform = transform, .size = 0.5F}});
        // One glyph rasterised, two quads, one transform.
        CHECK(batch.jobs.size() == 1);
        REQUIRE(batch.world_instances.size() == 2);
        REQUIRE(batch.world_transforms.size() == 1);
        CHECK(batch.instances.empty());

        CHECK(batch.world_transforms[0][0].x == doctest::Approx(0.5F));
        CHECK(batch.world_transforms[0][3].x == doctest::Approx(1.0F));
        CHECK(batch.world_transforms[0][3].z == doctest::Approx(3.0F));

        for (auto const &glyph: batch.world_instances) {
            CHECK(glyph.transform == 0);
            CHECK(glyph.max.y > 0.0F);
            CHECK(glyph.max.x > glyph.min.x);
            CHECK(glyph.max.y > glyph.min.y);
        }
        // The second 'A' is one advance to the right of the first.
        CHECK(batch.world_instances[1].min.x > batch.world_instances[0].min.x);

        // The same glyph drawn on screen needs no new rasterisation.
        auto const screen = cache.build(std::vector{text::Draw{.font = id, .text = "A", .size = 16}});
        CHECK(screen.jobs.empty());
        CHECK(screen.instances.size() == 1);
    }

    TEST_CASE("a frame stops rasterising at its limit and finishes on the next") {
        text::Cache cache;
        auto const id = cache.add_font(load_font());
        auto const *const font = cache.font(id);
        REQUIRE(font != nullptr);

        // Every distinct glyph with an outline the font has in the Latin blocks.
        auto text = std::string{};
        auto distinct = std::set<std::uint32_t>{};
        // Of those, the ones with something to draw: a no-break space has a glyph and no outline.
        auto drawable = std::size_t{0};
        for (auto code_point = char32_t{0x21}; code_point < 0x250; ++code_point) {
            auto const glyph = font->glyph_index(code_point);
            if (glyph != Font::missing_glyph && distinct.insert(glyph).second) {
                if (auto const outline = font->outline(glyph); outline && !outline->empty()) {
                    ++drawable;
                }
                // Encode as UTF-8.
                if (code_point < 0x80) {
                    text.push_back(static_cast<char>(code_point));
                } else {
                    text.push_back(static_cast<char>(0xC0 | (code_point >> 6U)));
                    text.push_back(static_cast<char>(0x80 | (code_point & 0x3FU)));
                }
            }
        }
        if (drawable <= text::max_new_glyphs_per_frame) {
            MESSAGE("the font has too few glyphs to reach the per-frame limit");
            return;
        }

        auto const draws = std::vector{text::Draw{.font = id, .text = text, .size = 16}};
        auto const first = cache.build(draws);
        CHECK(first.jobs.size() == text::max_new_glyphs_per_frame);
        CHECK(first.deferred_glyphs > 0);

        // The rest follows, each glyph rasterised once, in as many frames as the limit needs.
        auto rasterised = first.jobs.size();
        auto frames = 1U;
        for (; frames < 10; ++frames) {
            auto const batch = cache.build(draws);
            rasterised += batch.jobs.size();
            if (batch.deferred_glyphs == 0) {
                break;
            }
        }
        CHECK(rasterised == drawable);
        CHECK(frames + 1 == (drawable + text::max_new_glyphs_per_frame - 1) / text::max_new_glyphs_per_frame);
        CHECK(cache.build(draws).jobs.empty());
    }

    TEST_CASE("text the cache cannot draw is skipped") {
        text::Cache cache;
        auto const id = cache.add_font(load_font());

        auto const batch = cache.build(std::vector{
                text::Draw{.font = 99, .text = "unknown font"},
                text::Draw{.font = id, .text = ""},
                text::Draw{.font = id, .text = "zero size", .size = 0.0F},
        });
        CHECK(batch.empty());
        CHECK(cache.measure(99, "x", 10.0F) == glm::vec2{0.0F});
        CHECK(cache.measure(id, "ABC", 20.0F).x == doctest::Approx(2.0F * cache.measure(id, "ABC", 10.0F).x));
    }
}

namespace {
    auto emoji_font() -> Font {
        auto font = Font::load(std::filesystem::path{TEST_ASSETS_DIR} / "assets" / "fonts" / "NotoEmoji-Regular.ttf");
        REQUIRE(font.has_value());
        return std::move(*font);
    }
}

TEST_CASE("an emoji the font lacks comes from the fallback font as a real outline") {
    auto const primary = load_font();
    auto const emoji = emoji_font();
    // U+1F600 GRINNING FACE.
    auto const face = utf8({0xF0, 0x9F, 0x98, 0x80});

    REQUIRE(primary.glyph_index(U'\x1F600') == Font::missing_glyph);
    REQUIRE(emoji.glyph_index(U'\x1F600') != Font::missing_glyph);

    std::array<Font const *, 1> const fallbacks{&emoji};
    auto const placed = text::layout(primary, fallbacks, "a" + face);

    REQUIRE(placed.glyphs.size() == 2);
    CHECK(placed.glyphs[0].font_slot == 0);
    CHECK(placed.glyphs[1].font_slot == 1);
    CHECK(placed.glyphs[1].glyph == emoji.glyph_index(U'\x1F600'));
    // It is placed after the first glyph, by that glyph's advance.
    CHECK(placed.glyphs[1].pen.x == doctest::Approx(primary.advance(primary.glyph_index(U'a'))));

    auto const outline = emoji.outline(placed.glyphs[1].glyph);
    REQUIRE(outline.has_value());
    CHECK_FALSE(outline->empty());

    // Without the fallback it is the primary's missing-glyph box, as before.
    auto const alone = text::layout(primary, "a" + face);
    REQUIRE(alone.glyphs.size() == 2);
    CHECK(alone.glyphs[1].glyph == Font::missing_glyph);
    CHECK(alone.glyphs[1].font_slot == 0);
}

TEST_CASE("joiners and variation selectors take no room and draw nothing") {
    auto const primary = load_font();
    auto const emoji = emoji_font();
    std::array<Font const *, 1> const fallbacks{&emoji};

    // U+2764 HEAVY BLACK HEART, U+FE0F VARIATION SELECTOR-16, U+200D ZERO WIDTH JOINER, U+2764 again.
    auto const text = utf8({0xE2, 0x9D, 0xA4, 0xEF, 0xB8, 0x8F, 0xE2, 0x80, 0x8D, 0xE2, 0x9D, 0xA4});
    auto const placed = text::layout(primary, fallbacks, text);

    CHECK(placed.glyphs.size() == 2);
    CHECK(text::is_invisible(U'\xFE0F'));
    CHECK(text::is_invisible(U'\x200D'));
    CHECK_FALSE(text::is_invisible(U'a'));
}

TEST_CASE("the cache lays out, measures and plans an emoji with its fallback font") {
    text::Cache cache;
    auto const primary = cache.add_font(load_font());
    auto const emoji = cache.add_font(emoji_font());
    cache.set_fallback_fonts({emoji});

    auto const face = utf8({0xF0, 0x9F, 0x98, 0x80});

    auto const with = cache.measure(primary, face, 1.0F);
    CHECK(with.x > 0.0F);

    auto const batch = cache.build(std::array{text::Draw{.font = primary, .text = face, .size = 32.0F}});
    // The glyph is a real outline from the emoji font: it gets a job and a quad, not a blank.
    CHECK(batch.jobs.size() == 1);
    CHECK(batch.instances.size() == 1);

    text::Cache without;
    auto const bare = without.add_font(load_font());
    auto const missing = without.build(std::array{text::Draw{.font = bare, .text = face, .size = 32.0F}});
    // The primary's missing-glyph box is also an outline, but from a different font: the two do not share a tile.
    CHECK(missing.jobs.size() == 1);
    CHECK(missing.jobs[0].segment_count != batch.jobs[0].segment_count);
}
