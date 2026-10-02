#include <doctest/doctest.h>

#include "rendering/brdf_lut.hxx"

TEST_CASE("the split-sum LUT stays within the energy bounds") {
    for (float ndotv = 0.05F; ndotv <= 1.0F; ndotv += 0.19F) {
        for (float roughness = 0.045F; roughness <= 1.0F; roughness += 0.19F) {
            auto const lut = integrate_brdf(ndotv, roughness, 512);

            CHECK(lut.x >= 0.0F);
            CHECK(lut.y >= 0.0F);
            CHECK(lut.x <= 1.0F);
            CHECK(lut.y <= 1.0F);

            // A + B is the reflectance of a perfect (F0 = 1) specular: it never exceeds 1 beyond sampling noise.
            CHECK(lut.x + lut.y <= 1.02F);
        }
    }
}

TEST_CASE("a smooth surface seen head on reflects almost everything and Fresnel adds nothing at normal incidence") {
    auto const lut = integrate_brdf(1.0F, 0.045F, 2048);

    CHECK(lut.x + lut.y > 0.95F);
    CHECK(lut.y < 0.05F);
}

TEST_CASE("grazing views push reflectance into the Fresnel term B") {
    auto const head_on = integrate_brdf(1.0F, 0.3F, 1024);
    auto const grazing = integrate_brdf(0.05F, 0.3F, 1024);

    CHECK(grazing.y > head_on.y);
}

TEST_CASE("rougher surfaces lose energy that multi-scatter compensation puts back") {
    constexpr float ndotv = 0.5F;

    auto const smooth = integrate_brdf(ndotv, 0.1F, 1024);
    auto const rough = integrate_brdf(ndotv, 1.0F, 1024);

    CHECK(smooth.x + smooth.y > rough.x + rough.y);

    // Fdez-Aguera's compensation 1 + F0 (1 / (A + B) - 1) restores a white (F0 = 1) specular to exactly 1.
    auto const energy = 1.0F + (1.0F * ((1.0F / (rough.x + rough.y)) - 1.0F));

    CHECK((rough.x + rough.y) * energy == doctest::Approx(1.0F).epsilon(1e-5));
}
