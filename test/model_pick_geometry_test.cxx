#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <glm/geometric.hpp>

#include "assets/model_pick_geometry.hxx"

namespace {

    // A flat n x n grid of unit quads in the XZ plane at y = 0, spanning [0, n] on both axes.
    auto grid(int n) -> std::pair<std::vector<glm::vec3>, std::vector<std::uint32_t>> {
        std::vector<glm::vec3> positions;
        std::vector<std::uint32_t> indices;

        for (auto z = 0; z <= n; ++z) {
            for (auto x = 0; x <= n; ++x) {
                positions.emplace_back(static_cast<float>(x), 0.0F, static_cast<float>(z));
            }
        }

        auto const stride = static_cast<std::uint32_t>(n + 1);

        for (auto z = 0; z < n; ++z) {
            for (auto x = 0; x < n; ++x) {
                auto const a = (static_cast<std::uint32_t>(z) * stride) + static_cast<std::uint32_t>(x);
                indices.insert(indices.end(), {a, a + stride, a + 1, a + 1, a + stride, a + stride + 1});
            }
        }

        return {std::move(positions), std::move(indices)};
    }

}

TEST_CASE("a ray from above hits the plane with an upward normal") {
    auto [positions, indices] = grid(8);
    auto const geometry = ModelPickGeometry::build(std::move(positions), std::move(indices));

    auto const hit = geometry->raycast({3.25F, 5.0F, 4.75F}, {0.0F, -1.0F, 0.0F}, 100.0F);

    REQUIRE(hit.has_value());
    CHECK(hit->distance == doctest::Approx(5.0F));
    CHECK(hit->position.x == doctest::Approx(3.25F));
    CHECK(hit->position.y == doctest::Approx(0.0F));
    CHECK(hit->position.z == doctest::Approx(4.75F));
    CHECK(hit->normal.y == doctest::Approx(1.0F));
}

TEST_CASE("a ray from below hits the same plane and the normal faces the ray") {
    auto [positions, indices] = grid(8);
    auto const geometry = ModelPickGeometry::build(std::move(positions), std::move(indices));

    auto const hit = geometry->raycast({1.5F, -2.0F, 1.5F}, {0.0F, 1.0F, 0.0F}, 100.0F);

    REQUIRE(hit.has_value());
    CHECK(hit->normal.y == doctest::Approx(-1.0F));
}

TEST_CASE("rays that miss, point away or stop short find nothing") {
    auto [positions, indices] = grid(8);
    auto const geometry = ModelPickGeometry::build(std::move(positions), std::move(indices));

    CHECK_FALSE(geometry->raycast({20.0F, 5.0F, 4.0F}, {0.0F, -1.0F, 0.0F}, 100.0F).has_value());
    CHECK_FALSE(geometry->raycast({3.0F, 5.0F, 4.0F}, {0.0F, 1.0F, 0.0F}, 100.0F).has_value());
    CHECK_FALSE(geometry->raycast({3.0F, 5.0F, 4.0F}, {0.0F, -1.0F, 0.0F}, 4.0F).has_value());
    CHECK_FALSE(geometry->raycast({3.0F, 0.5F, -4.0F}, {0.0F, 0.0F, 1.0F}, 100.0F).has_value());
}

TEST_CASE("an empty geometry never hits") {
    auto const geometry = ModelPickGeometry::build({}, {});
    CHECK(geometry->triangle_count() == 0);
    CHECK_FALSE(geometry->raycast({0.0F, 1.0F, 0.0F}, {0.0F, -1.0F, 0.0F}, 10.0F).has_value());
}

TEST_CASE("triangles that index past the vertices are ignored") {
    auto const geometry = ModelPickGeometry::build({{0, 0, 0}, {1, 0, 0}, {0, 0, 1}}, {0, 1, 2, 0, 1, 9, 2});
    CHECK(geometry->triangle_count() == 1);
}

TEST_CASE("the hierarchy agrees with testing every triangle") {
    // A lumpy surface so the nearest hit is not just the first triangle in the list.
    auto [positions, indices] = grid(40);

    for (auto &point: positions) {
        point.y = 0.4F * std::sin(point.x * 0.7F) * std::cos(point.z * 0.5F);
    }

    auto const brute = [&](glm::vec3 const &origin, glm::vec3 const &direction) {
        auto best = std::numeric_limits<float>::max();

        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            auto const &a = positions[indices[i]];
            auto const edge_ab = positions[indices[i + 1]] - a;
            auto const edge_ac = positions[indices[i + 2]] - a;
            auto const p = glm::cross(direction, edge_ac);
            auto const determinant = glm::dot(edge_ab, p);

            if (std::abs(determinant) < 1e-9F) {
                continue;
            }

            auto const to_origin = origin - a;
            auto const u = glm::dot(to_origin, p) / determinant;
            auto const q = glm::cross(to_origin, edge_ab);
            auto const v = glm::dot(direction, q) / determinant;
            auto const t = glm::dot(edge_ac, q) / determinant;

            if (u >= 0.0F && v >= 0.0F && u + v <= 1.0F && t >= 0.0F) {
                best = std::min(best, t);
            }
        }

        return best;
    };

    auto const geometry = ModelPickGeometry::build(positions, indices);
    std::mt19937 random{1234};
    std::uniform_real_distribution<float> position{-2.0F, 42.0F};
    std::uniform_real_distribution<float> lean{-0.6F, 0.6F};

    for (auto i = 0; i < 300; ++i) {
        auto const origin = glm::vec3{position(random), 6.0F, position(random)};
        auto const direction = glm::normalize(glm::vec3{lean(random), -1.0F, lean(random)});

        auto const expected = brute(origin, direction);
        auto const hit = geometry->raycast(origin, direction, std::numeric_limits<float>::max());

        if (expected == std::numeric_limits<float>::max()) {
            CHECK_FALSE(hit.has_value());
        } else {
            REQUIRE(hit.has_value());
            CHECK(hit->distance == doctest::Approx(expected).epsilon(1e-4));
        }
    }
}
