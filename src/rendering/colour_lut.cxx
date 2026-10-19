#include "rendering/colour_lut.hxx"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace {
    constexpr auto max_cube_size = std::uint32_t{64};

    constexpr auto luma_weights = glm::vec3{0.2126F, 0.7152F, 0.0722F};

    [[nodiscard]] auto luma(glm::vec3 colour) noexcept -> float {
        return glm::dot(colour, luma_weights);
    }

    // Pulls colour towards its own luma (amount < 1) or away from it (amount > 1).
    [[nodiscard]] auto saturation(glm::vec3 colour, float amount) noexcept -> glm::vec3 {
        return glm::mix(glm::vec3{luma(colour)}, colour, amount);
    }

    // A smoothstep S-curve blended in by `amount`: deepens shadows and brightens highlights around the mid-grey.
    [[nodiscard]] auto contrast(glm::vec3 colour, float amount) noexcept -> glm::vec3 {
        auto const clamped = glm::clamp(colour, 0.0F, 1.0F);
        return glm::mix(clamped, clamped * clamped * (3.0F - 2.0F * clamped), amount);
    }

    [[nodiscard]] auto cinematic(glm::vec3 colour) noexcept -> glm::vec3 {
        constexpr auto teal = glm::vec3{0.05F, 0.42F, 0.50F};
        constexpr auto orange = glm::vec3{1.00F, 0.62F, 0.30F};

        auto graded = contrast(colour, 0.35F);
        graded = saturation(graded, 1.12F);

        // Split-tone: tint by where the pixel sits in the tonal range.
        auto const brightness = luma(graded);
        auto const shadows = 1.0F - glm::smoothstep(0.0F, 0.55F, brightness);
        auto const highlights = glm::smoothstep(0.45F, 1.0F, brightness);

        graded = glm::mix(graded, graded * teal * 2.0F, 0.30F * shadows);
        graded = glm::mix(graded, graded * orange * 1.5F, 0.28F * highlights);

        return glm::clamp(graded, 0.0F, 1.0F);
    }

    [[nodiscard]] auto vintage(glm::vec3 colour) noexcept -> glm::vec3 {
        constexpr auto warm = glm::vec3{1.06F, 1.00F, 0.86F};
        constexpr auto black_point = glm::vec3{0.08F, 0.06F, 0.07F};
        constexpr auto white_point = glm::vec3{0.96F, 0.93F, 0.86F};

        auto graded = saturation(glm::clamp(colour, 0.0F, 1.0F), 0.72F);
        graded = contrast(graded, -0.15F);
        graded = glm::mix(black_point, white_point, graded);

        return glm::clamp(graded * warm, 0.0F, 1.0F);
    }

    [[nodiscard]] auto noir(glm::vec3 colour) noexcept -> glm::vec3 {
        constexpr auto shadow_tint = glm::vec3{0.90F, 0.97F, 1.08F};

        auto graded = saturation(glm::clamp(colour, 0.0F, 1.0F), 0.06F);
        graded = contrast(graded, 0.75F);

        auto const shadows = 1.0F - glm::smoothstep(0.0F, 0.5F, luma(graded));
        graded = glm::mix(graded, graded * shadow_tint, shadows);

        return glm::clamp(graded, 0.0F, 1.0F);
    }

    template<typename Grade>
    [[nodiscard]] auto make_graded_lut(std::uint32_t size, Grade grade) -> ColourLut {
        auto lut = make_identity_colour_lut(size);

        for (auto &entry: lut.entries) {
            entry = grade(entry);
        }

        return lut;
    }

    [[nodiscard]] auto to_byte(float value) noexcept -> std::byte {
        return static_cast<std::byte>(static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F)));
    }

    [[nodiscard]] auto trim(std::string_view text) noexcept -> std::string_view {
        constexpr auto whitespace = std::string_view{" \t\r"};

        auto const first = text.find_first_not_of(whitespace);
        if (first == std::string_view::npos) {
            return {};
        }

        return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
    }

    // Reads up to `count` floats separated by whitespace; false unless exactly that many parse and nothing is left.
    template<std::size_t N>
    [[nodiscard]] auto parse_floats(std::string_view text, std::array<float, N> &values) -> bool {
        auto const owned = std::string{text};
        auto const *cursor = owned.c_str();

        for (auto &value: values) {
            char *end = nullptr;
            value = std::strtof(cursor, &end);

            if (end == cursor) {
                return false;
            }

            cursor = end;
        }

        return trim(cursor).empty();
    }
}

auto ColourLut::sample(glm::vec3 colour) const noexcept -> glm::vec3 {
    if (size == 0) {
        return colour;
    }

    if (size == 1) {
        return entries.front();
    }

    auto const scaled = glm::clamp(colour, 0.0F, 1.0F) * static_cast<float>(size - 1);
    auto const base = glm::min(glm::uvec3{scaled}, glm::uvec3{size - 2});
    auto const t = scaled - glm::vec3{base};

    auto const lerp_red = [&](std::uint32_t g, std::uint32_t b) {
        return glm::mix(at(base.x, g, b), at(base.x + 1, g, b), t.x);
    };
    auto const lerp_green = [&](std::uint32_t b) {
        return glm::mix(lerp_red(base.y, b), lerp_red(base.y + 1, b), t.y);
    };

    return glm::mix(lerp_green(base.z), lerp_green(base.z + 1), t.z);
}

auto ColourLut::to_strip_rgba8() const -> std::vector<std::byte> {
    auto pixels = std::vector<std::byte>{};
    pixels.reserve(static_cast<std::size_t>(size) * size * size * 4);

    for (std::uint32_t g = 0; g < size; ++g) {
        for (std::uint32_t b = 0; b < size; ++b) {
            for (std::uint32_t r = 0; r < size; ++r) {
                auto const &colour = at(r, g, b);

                pixels.push_back(to_byte(colour.r));
                pixels.push_back(to_byte(colour.g));
                pixels.push_back(to_byte(colour.b));
                pixels.push_back(std::byte{255});
            }
        }
    }

    return pixels;
}

auto make_identity_colour_lut(std::uint32_t size) -> ColourLut {
    auto lut = ColourLut{};
    lut.size = size;
    lut.entries.reserve(static_cast<std::size_t>(size) * size * size);

    auto const scale = size > 1 ? 1.0F / static_cast<float>(size - 1) : 0.0F;

    for (std::uint32_t b = 0; b < size; ++b) {
        for (std::uint32_t g = 0; g < size; ++g) {
            for (std::uint32_t r = 0; r < size; ++r) {
                lut.entries.emplace_back(static_cast<float>(r) * scale, static_cast<float>(g) * scale,
                                         static_cast<float>(b) * scale);
            }
        }
    }

    return lut;
}

auto make_default_colour_lut(DefaultColourLut lut, std::uint32_t size) -> ColourLut {
    switch (lut) {
        case DefaultColourLut::cinematic:
            return make_graded_lut(size, cinematic);
        case DefaultColourLut::vintage:
            return make_graded_lut(size, vintage);
        case DefaultColourLut::noir:
            return make_graded_lut(size, noir);
    }

    return make_identity_colour_lut(size);
}

auto parse_cube_lut(std::string_view text) -> std::optional<ColourLut> {
    auto lut = ColourLut{};
    auto expected_entries = std::size_t{0};

    while (!text.empty()) {
        auto const newline = text.find('\n');
        auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);

        if (line.empty() || line.front() == '#') {
            continue;
        }

        if (line.starts_with("TITLE")) {
            continue;
        }

        if (line.starts_with("LUT_1D_SIZE")) {
            return std::nullopt;
        }

        if (line.starts_with("LUT_3D_SIZE")) {
            auto size = std::array<float, 1>{};
            if (!parse_floats(line.substr(std::string_view{"LUT_3D_SIZE"}.size()), size)) {
                return std::nullopt;
            }

            if (size[0] < 2.0F || size[0] > static_cast<float>(max_cube_size) || size[0] != std::floor(size[0])) {
                return std::nullopt;
            }

            lut.size = static_cast<std::uint32_t>(size[0]);
            expected_entries = static_cast<std::size_t>(lut.size) * lut.size * lut.size;
            lut.entries.reserve(expected_entries);
            continue;
        }

        if (line.starts_with("DOMAIN_MIN") || line.starts_with("DOMAIN_MAX")) {
            auto const is_min = line.starts_with("DOMAIN_MIN");
            auto bound = std::array<float, 3>{};
            if (!parse_floats(line.substr(std::string_view{"DOMAIN_MIN"}.size()), bound)) {
                return std::nullopt;
            }

            auto const expected = is_min ? 0.0F : 1.0F;
            if (bound[0] != expected || bound[1] != expected || bound[2] != expected) {
                return std::nullopt;
            }
            continue;
        }

        if (lut.size == 0) {
            return std::nullopt;
        }

        auto rgb = std::array<float, 3>{};
        if (!parse_floats(line, rgb) || lut.entries.size() == expected_entries) {
            return std::nullopt;
        }

        lut.entries.emplace_back(rgb[0], rgb[1], rgb[2]);
    }

    if (lut.size == 0 || lut.entries.size() != expected_entries) {
        return std::nullopt;
    }

    return lut;
}
