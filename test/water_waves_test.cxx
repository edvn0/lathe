#include <doctest/doctest.h>

#include <cmath>

#include <entt/entt.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "rendering/water_system.hxx"
#include "rendering/water_waves.hxx"
#include "serialisation/scene_codec.hxx"
#include "scene/components.hxx"

// The wave field is mirrored from assets/shaders/water.slang; the GPU side cannot run here, so these check the CPU
// copy's own promises: the same input always gives the same surface, the offset stays within the wave height and
// averages out, and a footprint larger than every wave flattens it.

TEST_CASE("the water wave field is deterministic and bounded") {
    for (auto const position: {glm::vec2{0.0F}, glm::vec2{3.7F, -12.5F}, glm::vec2{250.0F, 41.0F}}) {
        for (auto const time: {0.0F, 1.5F, 60.0F}) {
            auto const first = water_waves::field(position, time, 0.13F, 1.0F, 0.05F);
            auto const second = water_waves::field(position, time, 0.13F, 1.0F, 0.05F);
            CHECK(first.offset == second.offset);
            CHECK(first.gradient == second.gradient);
            CHECK(std::isfinite(first.offset));
            CHECK(first.offset <= 1.0F);
            CHECK(first.offset >= -water_waves::wave_mean / water_waves::wave_peak);
            CHECK(first.height >= 0.0F);
            CHECK(first.height <= 1.0F);
        }
    }
}

TEST_CASE("the water surface averages out to its rest height") {
    auto sum = 0.0;
    auto count = 0;
    for (int x = 0; x < 64; ++x) {
        for (int z = 0; z < 64; ++z) {
            sum += water_waves::field({static_cast<float>(x) * 1.31F, static_cast<float>(z) * 0.97F}, 3.0F, 0.13F,
                                      1.0F, 0.05F)
                           .offset;
            ++count;
        }
    }
    CHECK(std::abs(sum / count) < 0.15);
}

TEST_CASE("waves shorter than the footprint are faded out to a flat surface") {
    auto const flat = water_waves::field({5.0F, 7.0F}, 2.0F, 0.13F, 1.0F, 1.0e4F);
    CHECK(flat.offset == doctest::Approx(0.0F));
    CHECK(flat.gradient.x == doctest::Approx(0.0F));
    CHECK(flat.gradient.y == doctest::Approx(0.0F));
}

TEST_CASE("WaterSystem queries find the body over a point") {
    auto registry = entt::registry{};

    auto const ocean = registry.create();
    registry.emplace<Components::Transform>(ocean, Components::Transform{.position = {0.0F, 0.0F, 0.0F}});
    registry.emplace<Components::Water>(ocean, Components::Water::ocean({100.0F, 100.0F}));

    auto const pool = registry.create();
    registry.emplace<Components::Transform>(pool, Components::Transform{.position = {10.0F, 2.0F, 10.0F}});
    registry.emplace<Components::Water>(pool, Components::Water::pool({4.0F, 4.0F}));

    auto const time = 1.25F;

    // Outside every body: nothing.
    CHECK_FALSE(WaterSystem::water_height(registry, {500.0F, 0.0F}, time).has_value());
    CHECK_FALSE(WaterSystem::is_underwater(registry, {500.0F, -10.0F, 0.0F}, time));

    // On the ocean: its height plus the displacement, within its wave height.
    auto const sea = WaterSystem::water_height(registry, {-20.0F, 30.0F}, time);
    REQUIRE(sea.has_value());
    auto const expected = water_waves::field({-20.0F, 30.0F}, time, Components::Water::ocean().wave_scale,
                                             Components::Water::ocean().wave_speed,
                                             water_waves::vertex_footprint_minimum)
                                  .offset *
                          Components::Water::ocean().wave_height;
    CHECK(*sea == doctest::Approx(expected));
    CHECK(std::abs(*sea) <= Components::Water::ocean().wave_height + 1e-4F);
    CHECK(WaterSystem::is_underwater(registry, {-20.0F, *sea - 0.1F, 30.0F}, time));
    CHECK_FALSE(WaterSystem::is_underwater(registry, {-20.0F, *sea + 0.1F, 30.0F}, time));

    // The pool wins over the ocean where both cover the point.
    auto const in_pool = WaterSystem::water_height(registry, {11.0F, 9.0F}, time);
    REQUIRE(in_pool.has_value());
    CHECK(*in_pool == doctest::Approx(2.0F).epsilon(0.05));

    auto const normal = WaterSystem::water_normal(registry, {-20.0F, 30.0F}, time);
    REQUIRE(normal.has_value());
    CHECK(glm::length(*normal) == doctest::Approx(1.0F));
    CHECK(normal->y > 0.0F);

    // A hidden body is not water.
    registry.get<Components::Water>(pool).visible = false;
    auto const without_pool = WaterSystem::water_height(registry, {11.0F, 9.0F}, time);
    REQUIRE(without_pool.has_value());
    CHECK(*without_pool == doctest::Approx(*WaterSystem::water_height(registry, {11.0F, 9.0F}, time)));
    CHECK(std::abs(*without_pool) < 1.0F);
}

TEST_CASE("lake grids and the ocean's snapping") {
    CHECK(WaterSystem::lake_cells({10.0F, 10.0F}) == glm::uvec2{80U, 80U});
    CHECK(WaterSystem::lake_cells({0.0F, 1.0e6F}) ==
          glm::uvec2{WaterSystem::lake_min_cells, WaterSystem::lake_max_cells});

    auto const centre = WaterSystem::ocean_centre({3.0F, -3.0F});
    CHECK(centre == WaterSystem::ocean_centre({3.5F, -2.0F}));
    CHECK(WaterSystem::ocean_centre({5.0F, 0.0F}).x == doctest::Approx(WaterSystem::ocean_snap));
    CHECK(std::fmod(std::abs(centre.x), WaterSystem::ocean_snap) == doctest::Approx(0.0F));
}

TEST_CASE("Water components survive a scene round trip") {
    auto scene = SceneDescription{};
    scene.entities.push_back(SceneEntity{.name = "sea", .transform = Components::Transform{}});
    scene.entities.push_back(SceneEntity{.name = "pool", .transform = Components::Transform{}});

    auto pool = Components::Water::pool({6.0F, 3.5F});
    pool.visible = false;
    pool.colour = {0.1F, 0.2F, 0.3F};
    scene.waters.push_back(SceneWaterComponent{.entity = 0, .water = Components::Water::ocean({500.0F, 250.0F})});
    scene.waters.push_back(SceneWaterComponent{.entity = 1, .water = pool});

    auto const encoded = encode_scene(scene);
    REQUIRE(encoded.has_value());
    auto const decoded = decode_scene(*encoded);
    REQUIRE(decoded.has_value());
    REQUIRE(decoded->waters.size() == 2);
    CHECK(decoded->waters[0].entity == 0);
    CHECK(decoded->waters[0].water == Components::Water::ocean({500.0F, 250.0F}));
    CHECK(decoded->waters[1].water == pool);

    // A negative extent is refused when read back.
    scene.waters[1].water.extent.x = -1.0F;
    auto const bad = encode_scene(scene);
    REQUIRE(bad.has_value());
    CHECK_FALSE(decode_scene(*bad).has_value());
}
