#include <doctest/doctest.h>

#include <cmath>
#include <numbers>
#include <unordered_set>

#include <glm/geometric.hpp>

#include "rendering/planet_system.hxx"
#include "serialisation/scene_codec.hxx"
#include "terrain/planet.hxx"

namespace {
    constexpr double earth = 6'371'000.0;
}

TEST_CASE("planet directions and latitude/longitude agree") {
    for (auto const latitude: {-80.0, -30.0, 0.0, 12.5, 60.0}) {
        for (auto const longitude: {-170.0, -45.0, 0.0, 90.0, 179.0}) {
            auto const position = planet::LatLon{.latitude = latitude * std::numbers::pi / 180.0,
                                                 .longitude = longitude * std::numbers::pi / 180.0};
            auto const direction = planet::direction_of(position);
            CHECK(glm::length(direction) == doctest::Approx(1.0));
            auto const back = planet::lat_lon_of(direction);
            CHECK(back.latitude == doctest::Approx(position.latitude));
            CHECK(back.longitude == doctest::Approx(position.longitude));
        }
    }
    // North is +y, longitude 0 is +z.
    CHECK(planet::direction_of({.latitude = std::numbers::pi / 2.0}).y == doctest::Approx(1.0));
    CHECK(planet::direction_of({}).z == doctest::Approx(1.0));
}

TEST_CASE("the heightmap is sampled bilinearly and wraps in longitude") {
    auto map = planet::Heightmap{.width = 4, .height = 2, .metres = {0, 100, 200, 300, 0, 100, 200, 300}};
    REQUIRE(map.valid());
    // Column centres sit at longitudes -135, -45, 45, 135 degrees.
    auto const at = [&](double longitude) {
        return map.sample(planet::LatLon{.latitude = 0.0, .longitude = longitude * std::numbers::pi / 180.0});
    };
    CHECK(at(-135.0) == doctest::Approx(0.0));
    CHECK(at(-90.0) == doctest::Approx(50.0));
    CHECK(at(135.0) == doctest::Approx(300.0));
    // Across the date line, between the last column and the first.
    CHECK(at(180.0) == doctest::Approx(150.0));
    CHECK(map.lowest() == 0.0);
    CHECK(map.highest() == 300.0);
}

TEST_CASE("chunk selection splits towards the camera and never overlaps") {
    auto const settings = planet::LodSettings{.radius = earth, .max_level = 10, .highest = 9000.0, .lowest = -11000.0};

    // From far away the six faces are enough.
    auto const far = planet::select_chunks(glm::dvec3{0.0, 0.0, earth * 20.0}, settings);
    CHECK(far.size() <= 6U);

    // Near the surface the chunks under the camera are the finest, and none hides behind the horizon.
    auto const camera = glm::dvec3{0.0, 0.0, earth + 3000.0};
    auto const near = planet::select_chunks(camera, settings);
    REQUIRE(!near.empty());
    CHECK(near.size() < 2000U);

    auto deepest = std::uint8_t{0};
    auto keys = std::unordered_set<planet::ChunkKey, planet::ChunkKeyHash>{near.begin(), near.end()};
    for (auto const &key: near) {
        deepest = std::max(deepest, key.level);
        for (auto ancestor = key; ancestor.level > 0;) {
            ancestor = ancestor.parent();
            CHECK_FALSE(keys.contains(ancestor));
        }
        // Nothing on the far side of the planet.
        CHECK(glm::dot(planet::chunk_centre_direction(key), glm::normalize(camera)) > -0.2);
    }
    CHECK(deepest == settings.max_level);
}

TEST_CASE("a chunk's scaled frame puts its vertices back on the sphere") {
    auto const key = planet::ChunkKey{.face = 4, .level = 6, .x = 20, .y = 33};
    auto const settings = planet::ChunkBuildSettings{.radius = earth, .exaggeration = 0.0, .cells = 8};
    auto const chunk = planet::build_chunk(key, nullptr, settings);

    REQUIRE(chunk.vertices.size() >= 81U);
    CHECK(chunk.indices.size() % 3U == 0U);
    for (std::size_t i = 0; i < 81; ++i) {
        auto const &vertex = chunk.vertices[i];
        CHECK(std::abs(vertex.position.x) <= 1.0001F);
        CHECK(std::abs(vertex.position.y) <= 1.0001F);
        CHECK(std::abs(vertex.position.z) <= 1.0001F);
        auto const position = chunk.origin + chunk.rotation * (chunk.scale * glm::dvec3{vertex.position});
        // Within a few metres of the sphere (the vertex is single precision in a ~150 km frame).
        CHECK(std::abs(glm::length(position) - earth) < 5.0);
    }

    // The camera-relative matrix keeps small numbers: a chunk near the origin lands near zero.
    auto const centre = glm::dvec3{chunk.origin};
    auto const matrix = PlanetSystem::chunk_matrix(glm::dmat4{1.0}, chunk, centre);
    CHECK(glm::length(glm::vec3{matrix[3]}) < 1e-3F);
}

TEST_CASE("Planet components survive a scene round trip") {
    auto scene = SceneDescription{};
    scene.entities.push_back(SceneEntity{.name = "earth", .transform = Components::Transform{}});
    auto planet = Components::Planet{.elevation_path = "a/elev.png",
                                     .bathymetry_path = "a/bath.png",
                                     .colour = Components::PlanetColour::image,
                                     .colour_path = "a/colour.jpg",
                                     .exaggeration = 5.0F,
                                     .max_level = 9,
                                     .atmosphere = false};
    scene.planets.push_back(ScenePlanetComponent{.entity = 0, .planet = planet});

    auto const encoded = encode_scene(scene);
    REQUIRE(encoded.has_value());
    auto const decoded = decode_scene(*encoded);
    REQUIRE(decoded.has_value());
    REQUIRE(decoded->planets.size() == 1);
    CHECK(decoded->planets[0].planet == planet);
}
