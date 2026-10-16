#include "rendering/font.hxx"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <system_error>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H

namespace {
    // FreeType hands back a 26.6 fixed-point or font-unit coordinate; with FT_LOAD_NO_SCALE it is in font units.
    struct Flattener {
        GlyphOutline *outline = nullptr;
        float scale = 1.0F;
        float tolerance = 1.0F / 2048.0F;
        glm::vec2 pen{0.0F};
        glm::vec2 contour_start{0.0F};
        bool open = false;

        [[nodiscard]] auto point(FT_Vector const &v) const noexcept -> glm::vec2 {
            return {static_cast<float>(v.x) * scale, static_cast<float>(v.y) * scale};
        }

        auto line(glm::vec2 to) -> void {
            if (to != pen) {
                outline->segments.push_back({.from = pen, .to = to});
            }
            pen = to;
        }

        // Segments needed for a curve whose control polygon deviates by `deviation` em from a chord. A quadratic or
        // cubic Bezier is within roughly deviation / n^2 of its n-piece polyline, so n grows with the square root.
        [[nodiscard]] auto pieces(float deviation) const noexcept -> int {
            return std::clamp(static_cast<int>(std::ceil(std::sqrt(deviation / tolerance))), 1, 64);
        }

        auto conic(glm::vec2 control, glm::vec2 to) -> void {
            auto const from = pen;
            auto const deviation = glm::length(from - 2.0F * control + to) * 0.5F;
            auto const n = pieces(deviation);

            for (auto i = 1; i <= n; ++i) {
                auto const t = static_cast<float>(i) / static_cast<float>(n);
                auto const u = 1.0F - t;
                line(u * u * from + 2.0F * u * t * control + t * t * to);
            }
        }

        auto cubic(glm::vec2 control_a, glm::vec2 control_b, glm::vec2 to) -> void {
            auto const from = pen;
            auto const deviation = std::max(glm::length(from - 2.0F * control_a + control_b),
                                            glm::length(control_a - 2.0F * control_b + to)) *
                                   0.75F;
            auto const n = pieces(deviation);

            for (auto i = 1; i <= n; ++i) {
                auto const t = static_cast<float>(i) / static_cast<float>(n);
                auto const u = 1.0F - t;
                line(u * u * u * from + 3.0F * u * u * t * control_a + 3.0F * u * t * t * control_b + t * t * t * to);
            }
        }

        auto close() -> void {
            if (open) {
                line(contour_start);
                open = false;
            }
        }
    };

    auto move_to(FT_Vector const *to, void *user) -> int {
        auto &f = *static_cast<Flattener *>(user);
        f.close();
        f.pen = f.contour_start = f.point(*to);
        f.open = true;
        return 0;
    }

    auto line_to(FT_Vector const *to, void *user) -> int {
        auto &f = *static_cast<Flattener *>(user);
        f.line(f.point(*to));
        return 0;
    }

    auto conic_to(FT_Vector const *control, FT_Vector const *to, void *user) -> int {
        auto &f = *static_cast<Flattener *>(user);
        f.conic(f.point(*control), f.point(*to));
        return 0;
    }

    auto cubic_to(FT_Vector const *control_a, FT_Vector const *control_b, FT_Vector const *to, void *user) -> int {
        auto &f = *static_cast<Flattener *>(user);
        f.cubic(f.point(*control_a), f.point(*control_b), f.point(*to));
        return 0;
    }
} // namespace

struct Font::Impl {
    ~Impl() {
        if (face != nullptr) {
            FT_Done_Face(face);
        }
        if (library != nullptr) {
            FT_Done_FreeType(library);
        }
    }

    std::string name;
    std::vector<std::byte> data;
    FT_Library library = nullptr;
    FT_Face face = nullptr;
    FontMetrics metrics;
    float units_per_em = 1.0F;
};

Font::Font(std::unique_ptr<Impl> impl) noexcept : impl_{std::move(impl)} {}
Font::~Font() = default;
Font::Font(Font &&) noexcept = default;
auto Font::operator=(Font &&) noexcept -> Font & = default;

auto Font::load(std::filesystem::path const &path) -> std::expected<Font, std::string> {
    auto file = std::ifstream{path, std::ios::binary};
    if (!file) {
        return std::unexpected(std::format("cannot open font '{}'", path.string()));
    }

    auto size = std::error_code{};
    auto const length = std::filesystem::file_size(path, size);
    if (size) {
        return std::unexpected(std::format("cannot read the size of '{}'", path.string()));
    }

    auto bytes = std::vector<std::byte>(static_cast<std::size_t>(length));
    if (!file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        return std::unexpected(std::format("cannot read font '{}'", path.string()));
    }

    return load_memory(std::move(bytes), path.filename().string());
}

auto Font::load_memory(std::vector<std::byte> data, std::string name) -> std::expected<Font, std::string> {
    auto impl = std::make_unique<Impl>();
    impl->name = std::move(name);
    impl->data = std::move(data);

    if (FT_Init_FreeType(&impl->library) != 0) {
        return std::unexpected(std::string{"FreeType could not be initialised"});
    }

    if (auto const error = FT_New_Memory_Face(impl->library, reinterpret_cast<FT_Byte const *>(impl->data.data()),
                                              static_cast<FT_Long>(impl->data.size()), 0, &impl->face);
        error != 0) {
        return std::unexpected(std::format("'{}' is not a font FreeType can read (error {})", impl->name, error));
    }

    if (!FT_IS_SCALABLE(impl->face)) {
        return std::unexpected(std::format("'{}' has no outlines to draw", impl->name));
    }

    impl->units_per_em = static_cast<float>(std::max<FT_UShort>(impl->face->units_per_EM, 1));
    impl->metrics = {
            .ascent = static_cast<float>(impl->face->ascender) / impl->units_per_em,
            .descent = static_cast<float>(impl->face->descender) / impl->units_per_em,
            .line_height = static_cast<float>(impl->face->height) / impl->units_per_em,
    };

    return Font{std::move(impl)};
}

auto Font::name() const noexcept -> std::string const & { return impl_->name; }

auto Font::metrics() const noexcept -> FontMetrics const & { return impl_->metrics; }

auto Font::glyph_index(char32_t code_point) const -> std::uint32_t {
    return FT_Get_Char_Index(impl_->face, static_cast<FT_ULong>(code_point));
}

auto Font::advance(std::uint32_t glyph) const -> float {
    if (FT_Load_Glyph(impl_->face, glyph, FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP) != 0) {
        return 0.0F;
    }
    return static_cast<float>(impl_->face->glyph->metrics.horiAdvance) / impl_->units_per_em;
}

auto Font::kerning(std::uint32_t left, std::uint32_t right) const -> float {
    if (!FT_HAS_KERNING(impl_->face)) {
        return 0.0F;
    }

    auto delta = FT_Vector{};
    if (FT_Get_Kerning(impl_->face, left, right, FT_KERNING_UNSCALED, &delta) != 0) {
        return 0.0F;
    }
    return static_cast<float>(delta.x) / impl_->units_per_em;
}

auto Font::outline(std::uint32_t glyph, float tolerance) const -> std::expected<GlyphOutline, std::string> {
    if (FT_Load_Glyph(impl_->face, glyph, FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP) != 0) {
        return std::unexpected(std::format("glyph {} of '{}' cannot be loaded", glyph, impl_->name));
    }

    auto &slot = *impl_->face->glyph;
    auto result = GlyphOutline{};

    if (slot.format != FT_GLYPH_FORMAT_OUTLINE) {
        return result;
    }

    auto flattener = Flattener{.outline = &result, .scale = 1.0F / impl_->units_per_em, .tolerance = tolerance};
    auto const functions = FT_Outline_Funcs{
            .move_to = &move_to,
            .line_to = &line_to,
            .conic_to = &conic_to,
            .cubic_to = &cubic_to,
            .shift = 0,
            .delta = 0,
    };

    if (FT_Outline_Decompose(&slot.outline, &functions, &flattener) != 0) {
        return std::unexpected(std::format("glyph {} of '{}' has a broken outline", glyph, impl_->name));
    }
    flattener.close();

    if (!result.segments.empty()) {
        result.min = result.max = result.segments.front().from;
        for (auto const &segment: result.segments) {
            for (auto const point: {segment.from, segment.to}) {
                result.min = glm::min(result.min, point);
                result.max = glm::max(result.max, point);
            }
        }
    }

    return result;
}
