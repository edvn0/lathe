#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>

#include <glm/glm.hpp>

#include "rendering/cube_map.hxx"

namespace {
    constexpr float epsilon = 1e-5F;

    auto random_unit_vectors(std::size_t count) -> std::vector<glm::vec3> {
        std::mt19937 engine{1234};
        std::normal_distribution<float> normal{0.0F, 1.0F};

        std::vector<glm::vec3> result;
        result.reserve(count);

        while (result.size() < count) {
            glm::vec3 const candidate{normal(engine), normal(engine), normal(engine)};

            if (glm::length(candidate) > 1e-3F) {
                result.push_back(glm::normalize(candidate));
            }
        }

        return result;
    }
} // namespace

TEST_CASE("cube map face centres match Vulkan's face order") {
    CHECK(direction_to_cube_texel(glm::vec3{1, 0, 0}).face == 0);
    CHECK(direction_to_cube_texel(glm::vec3{-1, 0, 0}).face == 1);
    CHECK(direction_to_cube_texel(glm::vec3{0, 1, 0}).face == 2);
    CHECK(direction_to_cube_texel(glm::vec3{0, -1, 0}).face == 3);
    CHECK(direction_to_cube_texel(glm::vec3{0, 0, 1}).face == 4);
    CHECK(direction_to_cube_texel(glm::vec3{0, 0, -1}).face == 5);

    for (std::uint32_t face = 0; face < cube_face_count; ++face) {
        auto const texel = direction_to_cube_texel(cube_texel_direction(face, glm::vec2{0.5F}));

        CHECK(texel.face == face);
        CHECK(texel.uv.x == doctest::Approx(0.5F).epsilon(epsilon));
        CHECK(texel.uv.y == doctest::Approx(0.5F).epsilon(epsilon));
    }
}

TEST_CASE("cube map corners follow the Vulkan (sc, tc) table") {
    // +X looking down the axis: s runs toward -Z, t runs toward -Y (towards the face's lower rows).
    auto const plus_x = cube_texel_direction(0, glm::vec2{0.0F, 0.0F});
    CHECK(plus_x.z > 0.0F);
    CHECK(plus_x.y > 0.0F);

    // +Y: s runs toward +X, t toward +Z.
    auto const plus_y = cube_texel_direction(2, glm::vec2{1.0F, 1.0F});
    CHECK(plus_y.x > 0.0F);
    CHECK(plus_y.z > 0.0F);

    // -Z: s runs toward -X.
    auto const minus_z = cube_texel_direction(5, glm::vec2{1.0F, 0.0F});
    CHECK(minus_z.x < 0.0F);
    CHECK(minus_z.y > 0.0F);
}

TEST_CASE("cube map direction and texel round trip") {
    for (auto const direction: random_unit_vectors(4096)) {
        auto const texel = direction_to_cube_texel(direction);

        CHECK(texel.uv.x >= -epsilon);
        CHECK(texel.uv.x <= 1.0F + epsilon);
        CHECK(texel.uv.y >= -epsilon);
        CHECK(texel.uv.y <= 1.0F + epsilon);

        auto const back = cube_texel_direction(texel.face, texel.uv);

        CHECK(glm::length(back - direction) < 1e-4F);
    }
}

TEST_CASE("cube map texel solid angles sum to 4 pi") {
    for (std::uint32_t const size: {1U, 2U, 7U, 32U, 128U}) {
        double total = 0.0;

        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                auto const solid_angle = cube_texel_solid_angle(x, y, size);

                CHECK(solid_angle > 0.0);

                total += solid_angle;
            }
        }

        CHECK(total * cube_face_count == doctest::Approx(4.0 * std::numbers::pi).epsilon(1e-9));
    }
}

TEST_CASE("equirect centre column faces +Z and the top row is the zenith") {
    auto const centre = equirect_uv_direction(glm::vec2{0.5F, 0.5F});
    CHECK(centre.z == doctest::Approx(1.0F).epsilon(epsilon));
    CHECK(std::abs(centre.x) < epsilon);
    CHECK(std::abs(centre.y) < epsilon);

    auto const zenith = equirect_uv_direction(glm::vec2{0.3F, 0.0F});
    CHECK(zenith.y == doctest::Approx(1.0F).epsilon(epsilon));

    auto const quarter = equirect_uv_direction(glm::vec2{0.75F, 0.5F});
    CHECK(quarter.x == doctest::Approx(1.0F).epsilon(epsilon));

    auto const uv = direction_to_equirect_uv(glm::vec3{0, 0, 1});
    CHECK(uv.x == doctest::Approx(0.5F).epsilon(epsilon));
    CHECK(uv.y == doctest::Approx(0.5F).epsilon(epsilon));
}

TEST_CASE("equirect direction and uv round trip") {
    for (auto const direction: random_unit_vectors(2048)) {
        auto const back = equirect_uv_direction(direction_to_equirect_uv(direction));

        CHECK(glm::length(back - direction) < 1e-4F);
    }
}

TEST_CASE("rotate_y yaws about +Y") {
    auto const rotated = rotate_y(glm::vec3{1, 0, 0}, glm::vec2{0.0F, 1.0F});
    CHECK(rotated.x == doctest::Approx(0.0F).epsilon(epsilon));
    CHECK(rotated.z == doctest::Approx(-1.0F).epsilon(epsilon));

    auto const unchanged = rotate_y(glm::vec3{0.3F, 0.8F, -0.2F}, glm::vec2{1.0F, 0.0F});
    CHECK(unchanged.y == doctest::Approx(0.8F));
}
