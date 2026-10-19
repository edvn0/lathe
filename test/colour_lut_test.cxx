#include <doctest/doctest.h>

#include <cmath>
#include <string>

#include "rendering/colour_lut.hxx"

namespace {
    [[nodiscard]] auto near(glm::vec3 a, glm::vec3 b, float tolerance = 1e-4F) -> bool {
        return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec3{tolerance}));
    }
}

TEST_CASE("an identity LUT returns its input") {
    auto const lut = make_identity_colour_lut(16);

    for (auto const colour: {glm::vec3{0.0F}, glm::vec3{1.0F}, glm::vec3{0.25F, 0.5F, 0.75F},
                             glm::vec3{0.13F, 0.91F, 0.42F}}) {
        CHECK(near(lut.sample(colour), colour));
    }
}

TEST_CASE("lookups clamp to the table") {
    auto const lut = make_identity_colour_lut(8);

    CHECK(near(lut.sample(glm::vec3{-1.0F, 2.0F, 0.5F}), glm::vec3{0.0F, 1.0F, 0.5F}));
}

TEST_CASE("the strip puts blue slices along x, green down y and red inside a slice") {
    constexpr std::uint32_t size = 4;
    auto lut = make_identity_colour_lut(size);
    lut.entries[1 + size * (2 + size * 3)] = glm::vec3{1.0F, 0.0F, 0.0F};

    auto const strip = lut.to_strip_rgba8();
    REQUIRE(strip.size() == static_cast<std::size_t>(size) * size * size * 4);

    auto const texel = (static_cast<std::size_t>(2) * size * size + (3 * size + 1)) * 4;
    CHECK(std::to_integer<int>(strip[texel]) == 255);
    CHECK(std::to_integer<int>(strip[texel + 1]) == 0);
    CHECK(std::to_integer<int>(strip[texel + 2]) == 0);
    CHECK(std::to_integer<int>(strip[texel + 3]) == 255);
}

TEST_CASE("the default LUTs are distinct, in range and keep black and white sensible") {
    auto const identity = make_identity_colour_lut(default_colour_lut_size);

    for (auto const kind: {DefaultColourLut::cinematic, DefaultColourLut::vintage, DefaultColourLut::noir}) {
        CAPTURE(default_colour_lut_name(kind));

        auto const lut = make_default_colour_lut(kind);
        REQUIRE(lut.size == default_colour_lut_size);
        REQUIRE(lut.entries.size() == identity.entries.size());

        auto differs = false;
        for (std::size_t i = 0; i < lut.entries.size(); ++i) {
            CHECK(glm::all(glm::greaterThanEqual(lut.entries[i], glm::vec3{0.0F})));
            CHECK(glm::all(glm::lessThanEqual(lut.entries[i], glm::vec3{1.0F})));
            differs = differs || !near(lut.entries[i], identity.entries[i], 0.02F);
        }
        CHECK(differs);
    }
}

TEST_CASE("noir is nearly grey and vintage lifts black") {
    auto const noir = make_default_colour_lut(DefaultColourLut::noir);
    auto const saturated = noir.sample(glm::vec3{0.9F, 0.2F, 0.1F});
    CHECK(std::abs(saturated.r - saturated.g) < 0.12F);

    auto const vintage = make_default_colour_lut(DefaultColourLut::vintage);
    CHECK(vintage.sample(glm::vec3{0.0F}).r > 0.04F);
}

TEST_CASE("cinematic pushes shadows towards teal and highlights towards orange") {
    auto const lut = make_default_colour_lut(DefaultColourLut::cinematic);

    auto const shadow = lut.sample(glm::vec3{0.2F});
    CHECK(shadow.b > shadow.r);

    auto const highlight = lut.sample(glm::vec3{0.85F});
    CHECK(highlight.r > highlight.b);
}

TEST_CASE("a .cube file parses into the same table the identity builds") {
    std::string text = "# comment\nTITLE \"test\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\nLUT_3D_SIZE 2\n";
    for (int b = 0; b < 2; ++b) {
        for (int g = 0; g < 2; ++g) {
            for (int r = 0; r < 2; ++r) {
                text += std::to_string(r) + " " + std::to_string(g) + " " + std::to_string(b) + "\n";
            }
        }
    }

    auto const lut = parse_cube_lut(text);
    REQUIRE(lut.has_value());
    CHECK(lut->size == 2);
    CHECK(near(lut->sample(glm::vec3{0.3F, 0.6F, 0.9F}), glm::vec3{0.3F, 0.6F, 0.9F}));
}

TEST_CASE("malformed or unsupported .cube files are rejected") {
    CHECK_FALSE(parse_cube_lut("").has_value());
    CHECK_FALSE(parse_cube_lut("LUT_1D_SIZE 4\n0 0 0\n").has_value());
    CHECK_FALSE(parse_cube_lut("LUT_3D_SIZE 2\n0 0 0\n").has_value());
    CHECK_FALSE(parse_cube_lut("0 0 0\nLUT_3D_SIZE 2\n").has_value());
    CHECK_FALSE(parse_cube_lut("LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2\n").has_value());
    CHECK_FALSE(parse_cube_lut("LUT_3D_SIZE 2\n0 0 x\n").has_value());
    CHECK_FALSE(parse_cube_lut("LUT_3D_SIZE 99\n").has_value());
}
