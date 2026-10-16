#include "rendering/text_cache.hxx"

#include <algorithm>
#include <cmath>
#include <ranges>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "rendering/text_layout.hxx"

namespace text {
    namespace {
        [[nodiscard]] auto key_of(FontId font, std::uint32_t glyph) noexcept -> std::uint64_t {
            return (static_cast<std::uint64_t>(font) << 32U) | glyph;
        }

        [[nodiscard]] auto round_up_to_four(std::uint32_t value) noexcept -> std::uint32_t {
            return (value + 3U) & ~3U;
        }

        // The middle of laid-out text, in the local space of world text: x right, y up, the first baseline at y = 0.
        [[nodiscard]] auto text_centre(Font const &font, Layout const &placed) noexcept -> glm::vec2 {
            return {placed.size.x * 0.5F, font.metrics().ascent - placed.size.y * 0.5F};
        }
    } // namespace

    auto pack_colour(glm::vec4 colour) noexcept -> std::uint32_t {
        auto const byte = [](float value) {
            return static_cast<std::uint32_t>(
                    std::lround(std::clamp(std::isfinite(value) ? value : 0.0F, 0.0F, 1.0F) * 255.0F));
        };
        return byte(colour.r) | (byte(colour.g) << 8U) | (byte(colour.b) << 16U) | (byte(colour.a) << 24U);
    }

    auto Cache::add_font(Font font) -> FontId {
        fonts_.push_back(std::make_unique<Font>(std::move(font)));
        return static_cast<FontId>(fonts_.size() - 1);
    }

    auto Cache::font(FontId id) const noexcept -> Font const * {
        return id < fonts_.size() ? fonts_[id].get() : nullptr;
    }

    auto Cache::measure(FontId id, std::string_view utf8, float size) const -> glm::vec2 {
        auto const *const f = font(id);
        return f != nullptr ? layout(*f, utf8).size * size : glm::vec2{0.0F};
    }

    auto Cache::outline_points(FontId id, std::string_view utf8, bool centred, std::uint32_t count) const
            -> std::vector<glm::vec2> {
        auto const *const font = this->font(id);
        if (font == nullptr || utf8.empty() || count == 0) {
            return {};
        }

        struct Piece {
            glm::vec2 from;
            glm::vec2 to;
            // Length of the outline up to the end of this piece.
            float end = 0.0F;
        };

        auto const placed = layout(*font, utf8);
        auto const shift = centred ? text_centre(*font, placed) : glm::vec2{0.0F};

        auto pieces = std::vector<Piece>{};
        auto total = 0.0F;

        for (auto const &glyph: placed.glyphs) {
            // Points do not need the rasteriser's fine curves.
            auto const outline = font->outline(glyph.glyph, 1.0F / 512.0F);
            if (!outline) {
                continue;
            }

            auto const pen = glm::vec2{glyph.pen.x, font->metrics().ascent - glyph.pen.y} - shift;

            for (auto const &segment: outline->segments) {
                auto const from = pen + segment.from;
                auto const to = pen + segment.to;
                total += glm::distance(from, to);
                pieces.push_back({from, to, total});
            }
        }

        if (pieces.empty() || !(total > 0.0F)) {
            return {};
        }

        auto points = std::vector<glm::vec2>{};
        points.reserve(count);

        for (auto k = std::uint32_t{0}; k < count; ++k) {
            auto const target = (static_cast<float>(k) + 0.5F) * total / static_cast<float>(count);
            auto const piece = std::ranges::lower_bound(pieces, target, {}, &Piece::end);
            auto const &chosen = piece != pieces.end() ? *piece : pieces.back();
            auto const length = glm::distance(chosen.from, chosen.to);
            auto const along = length > 0.0F ? 1.0F - (chosen.end - target) / length : 0.0F;
            points.push_back(glm::mix(chosen.from, chosen.to, std::clamp(along, 0.0F, 1.0F)));
        }

        return points;
    }

    auto Cache::clear() -> void {
        glyphs_.clear();
        planner_.reset();
        next_slot_ = 0;
    }

    auto Cache::build(std::span<Draw const> draws, std::span<WorldDraw const> world_draws) -> Batch {
        auto needs_eviction = false;
        auto batch = attempt(draws, world_draws, false, needs_eviction);

        if (needs_eviction) {
            // The atlas (or its slot table) is full: start over, and have the GPU clear it before this frame's jobs.
            clear();
            ++evictions_;
            batch = attempt(draws, world_draws, true, needs_eviction);
            batch.reset_atlas = true;
        }

        return batch;
    }

    auto Cache::entry_for(Font const &font, FontId id, std::uint32_t glyph, Batch &batch, bool already_evicted,
                          bool &needs_eviction) -> Entry const * {
        auto const key = key_of(id, glyph);

        if (auto const found = glyphs_.find(key); found != glyphs_.end()) {
            return &found->second;
        }

        auto const outline = font.outline(glyph);
        auto entry = Entry{};

        if (!outline || outline->empty()) {
            entry.blank = true;
        } else {
            auto const extent = (outline->max - outline->min) * raster_em;
            auto const padding = static_cast<float>(spread);
            entry.tile_width = round_up_to_four(static_cast<std::uint32_t>(std::ceil(extent.x + 2.0F * padding)));
            entry.tile_height = static_cast<std::uint32_t>(std::ceil(extent.y + 2.0F * padding));
            entry.origin = {outline->min.x - padding / raster_em, outline->max.y + padding / raster_em};
            entry.blank = entry.tile_width > max_tile || entry.tile_height > max_tile;
        }

        if (!entry.blank) {
            if (batch.jobs.size() >= max_new_glyphs_per_frame ||
                batch.segments.size() + outline->segments.size() > max_segments_per_frame) {
                // Not cached, so the next frame tries again.
                ++batch.deferred_glyphs;
                return nullptr;
            }

            if (!planner_.place(entry.tile_width, entry.tile_height) || next_slot_ >= max_slots) {
                if (!already_evicted) {
                    needs_eviction = true;
                } else {
                    // Even a fresh atlas cannot take it.
                    ++batch.deferred_glyphs;
                }
                return nullptr;
            }

            entry.slot = next_slot_++;

            batch.jobs.push_back(GlyphJob{
                    .segment_offset = static_cast<std::uint32_t>(batch.segments.size()),
                    .segment_count = static_cast<std::uint32_t>(outline->segments.size()),
                    .tile_width = entry.tile_width,
                    .tile_height = entry.tile_height,
                    .origin_x = entry.origin.x,
                    .origin_y = entry.origin.y,
                    .slot = entry.slot,
                    .texel_offset = batch.texel_count,
            });
            batch.texel_count += entry.tile_width * entry.tile_height;

            for (auto const &segment: outline->segments) {
                batch.segments.emplace_back(segment.from, segment.to);
            }
        }

        return &glyphs_.emplace(key, entry).first->second;
    }

    auto Cache::attempt(std::span<Draw const> draws, std::span<WorldDraw const> world_draws, bool already_evicted,
                        bool &needs_eviction) -> Batch {
        auto batch = Batch{};

        for (auto const &draw: draws) {
            auto const *const font = this->font(draw.font);
            if (font == nullptr || draw.text.empty() || !(draw.size > 0.0F)) {
                continue;
            }

            for (auto const &glyph: layout(*font, draw.text).glyphs) {
                auto const *const entry =
                        entry_for(*font, draw.font, glyph.glyph, batch, already_evicted, needs_eviction);

                if (needs_eviction) {
                    return batch;
                }
                if (entry == nullptr || entry->blank || batch.instances.size() >= max_instances) {
                    continue;
                }

                auto const pen = draw.position + glyph.pen * draw.size;
                auto const min = glm::vec2{pen.x + entry->origin.x * draw.size, pen.y - entry->origin.y * draw.size};
                auto const extent =
                        glm::vec2{static_cast<float>(entry->tile_width), static_cast<float>(entry->tile_height)} *
                        (draw.size / raster_em);

                batch.instances.push_back(GlyphInstance{
                        .min = min,
                        .max = min + extent,
                        .slot = entry->slot,
                        .colour = pack_colour(draw.colour),
                });
            }
        }

        for (auto const &draw: world_draws) {
            auto const *const font = this->font(draw.font);
            if (font == nullptr || draw.text.empty() || !(draw.size > 0.0F)) {
                continue;
            }

            // The size is folded into the transform, so a glyph's quad stays in em units.
            auto const transform_index = static_cast<std::uint32_t>(batch.world_transforms.size());
            auto used = false;

            auto const placed = layout(*font, draw.text);
            auto const shift = draw.centred ? text_centre(*font, placed) : glm::vec2{0.0F};

            for (auto const &glyph: placed.glyphs) {
                auto const *const entry =
                        entry_for(*font, draw.font, glyph.glyph, batch, already_evicted, needs_eviction);

                if (needs_eviction) {
                    return batch;
                }
                if (entry == nullptr || entry->blank || batch.world_instances.size() >= max_instances) {
                    continue;
                }

                // Layout is y down from the top of the text; the world's local space is y up from the first baseline.
                auto const pen = glm::vec2{glyph.pen.x, font->metrics().ascent - glyph.pen.y} - shift;

                // The tile hangs down from its top edge, which is `origin.y` above the glyph's baseline.
                auto const top_left = glm::vec2{pen.x + entry->origin.x, pen.y + entry->origin.y};
                auto const extent =
                        glm::vec2{static_cast<float>(entry->tile_width), static_cast<float>(entry->tile_height)} /
                        raster_em;

                batch.world_instances.push_back(WorldGlyph{
                        .min = {top_left.x, top_left.y - extent.y},
                        .max = {top_left.x + extent.x, top_left.y},
                        .slot = entry->slot,
                        .colour = pack_colour(draw.colour),
                        .transform = transform_index,
                        .intensity = draw.intensity,
                });
                used = true;
            }

            if (used) {
                batch.world_transforms.push_back(draw.transform * glm::scale(glm::mat4{1.0F}, glm::vec3{draw.size}));
            }
        }

        return batch;
    }
} // namespace text
