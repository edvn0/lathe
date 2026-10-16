#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <glm/vec2.hpp>

// One straight piece of a glyph's outline. Curves are flattened when the outline is read, so everything downstream
// (the GPU rasteriser) only ever deals with line segments.
struct GlyphSegment {
    glm::vec2 from{0.0F};
    glm::vec2 to{0.0F};
};

// A glyph's outline in em units: 1.0 is the font's em size, x grows right and y grows up from the baseline.
struct GlyphOutline {
    std::vector<GlyphSegment> segments;
    // The box around the segments; zero-sized for a glyph with no outline, such as a space.
    glm::vec2 min{0.0F};
    glm::vec2 max{0.0F};

    [[nodiscard]] auto empty() const noexcept -> bool { return segments.empty(); }
};

struct FontMetrics {
    // Distance from the baseline to the top and bottom of a line, in em; descent is negative.
    float ascent = 0.0F;
    float descent = 0.0F;
    // Baseline to baseline.
    float line_height = 0.0F;
};

// A font file read with FreeType. Only outlines and metrics are used: nothing is rasterised on the CPU, and no FreeType
// type leaks out of this header.
class Font {
public:
    static constexpr std::uint32_t missing_glyph = 0;

    // Reads a TrueType or OpenType file.
    [[nodiscard]] static auto load(std::filesystem::path const &path) -> std::expected<Font, std::string>;
    // Takes the file's bytes; `name` identifies the font in messages.
    [[nodiscard]] static auto load_memory(std::vector<std::byte> data, std::string name)
            -> std::expected<Font, std::string>;

    ~Font();
    Font(Font &&) noexcept;
    auto operator=(Font &&) noexcept -> Font &;
    Font(Font const &) = delete;
    auto operator=(Font const &) -> Font & = delete;

    [[nodiscard]] auto name() const noexcept -> std::string const &;
    [[nodiscard]] auto metrics() const noexcept -> FontMetrics const &;

    // The glyph a code point is drawn with, or missing_glyph.
    [[nodiscard]] auto glyph_index(char32_t code_point) const -> std::uint32_t;
    // How far the pen moves after a glyph, and the extra distance between a pair, both in em.
    [[nodiscard]] auto advance(std::uint32_t glyph) const -> float;
    [[nodiscard]] auto kerning(std::uint32_t left, std::uint32_t right) const -> float;

    // The glyph's outline, flattened until no segment strays further than `tolerance` em from the true curve.
    [[nodiscard]] auto outline(std::uint32_t glyph, float tolerance = 1.0F / 2048.0F) const
            -> std::expected<GlyphOutline, std::string>;

private:
    struct Impl;
    explicit Font(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};
