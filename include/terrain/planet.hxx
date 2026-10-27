#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <glm/mat3x3.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "assets/load_model.hxx"

// The CPU side of a planet at real scale (docs/globe.md): a cube-sphere quadtree, the chunk meshes, and an
// equirectangular elevation map. Everything here is in double precision in the planet's own frame (centre at the
// origin, +y the north pole, longitude 0 on +z, east towards -x so the globe reads the right way round from outside in
// the engine's left-handed space). rendering/planet_system.hxx turns chunks into camera-relative draws.
namespace planet {

    // ---- directions ---------------------------------------------------------------------------------------------

    // Latitude and longitude in radians.
    struct LatLon {
        double latitude = 0.0;
        double longitude = 0.0;
    };

    [[nodiscard]] auto direction_of(LatLon position) noexcept -> glm::dvec3;
    [[nodiscard]] auto lat_lon_of(glm::dvec3 const &direction) noexcept -> LatLon;
    // Equirectangular texture coordinates (u east from -180 degrees, v south from the north pole), both in [0, 1].
    [[nodiscard]] auto equirect_uv(LatLon position) noexcept -> glm::dvec2;

    // ---- the cube sphere ------------------------------------------------------------------------------------------

    inline constexpr std::uint32_t face_count = 6;

    // A chunk of the quadtree: face, depth, and its cell at that depth.
    struct ChunkKey {
        std::uint8_t face = 0;
        std::uint8_t level = 0;
        std::uint32_t x = 0;
        std::uint32_t y = 0;

        friend auto operator==(ChunkKey const &, ChunkKey const &) -> bool = default;

        [[nodiscard]] auto parent() const noexcept -> ChunkKey {
            return ChunkKey{.face = face, .level = static_cast<std::uint8_t>(level - 1U), .x = x / 2U, .y = y / 2U};
        }
        [[nodiscard]] auto child(std::uint32_t index) const noexcept -> ChunkKey {
            return ChunkKey{.face = face,
                            .level = static_cast<std::uint8_t>(level + 1U),
                            .x = x * 2U + (index & 1U),
                            .y = y * 2U + (index >> 1U)};
        }
    };

    struct ChunkKeyHash {
        [[nodiscard]] auto operator()(ChunkKey const &key) const noexcept -> std::size_t {
            auto const packed = (static_cast<std::uint64_t>(key.face) << 61U) ^
                                (static_cast<std::uint64_t>(key.level) << 56U) ^
                                (static_cast<std::uint64_t>(key.x) << 28U) ^ static_cast<std::uint64_t>(key.y);
            return std::hash<std::uint64_t>{}(packed * 0x9E3779B97F4A7C15ULL);
        }
    };

    // The unit direction of face coordinates (s, t) in [-1, 1]; the tangent warp spreads cells evenly over the sphere.
    [[nodiscard]] auto cube_direction(std::uint32_t face, double s, double t) noexcept -> glm::dvec3;
    // The face coordinate range of a chunk: s0, t0, s1, t1.
    [[nodiscard]] auto chunk_extent(ChunkKey const &key) noexcept -> std::array<double, 4>;
    [[nodiscard]] auto chunk_centre_direction(ChunkKey const &key) noexcept -> glm::dvec3;
    // About the length (m) of a chunk's edge on a sphere of `radius`.
    [[nodiscard]] auto chunk_edge_length(std::uint8_t level, double radius) noexcept -> double;

    // ---- the elevation map ----------------------------------------------------------------------------------------

    // Metres above (positive) or below (negative) sea level on an equirectangular grid: row 0 is the north pole, column
    // 0 is longitude -180 degrees.
    struct Heightmap {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::int16_t> metres;

        [[nodiscard]] auto valid() const noexcept -> bool {
            return width > 1U && height > 1U && metres.size() == static_cast<std::size_t>(width) * height;
        }
        // Bilinear, wrapping in longitude.
        [[nodiscard]] auto sample(LatLon position) const noexcept -> double;
        [[nodiscard]] auto sample(glm::dvec3 const &direction) const noexcept -> double {
            return sample(lat_lon_of(direction));
        }
        // The range is found once (a full scan of the map) and remembered; not thread safe the first time.
        [[nodiscard]] auto lowest() const noexcept -> double;
        [[nodiscard]] auto highest() const noexcept -> double;

        auto find_range() const noexcept -> void;
        // Filled by find_range (members, not private, so the struct stays an aggregate).
        mutable bool range_known_ = false;
        mutable double lowest_ = 0.0;
        mutable double highest_ = 0.0;
    };

    // Where the two 8-bit grey images of NASA's Blue Marble topography/bathymetry set come from, and how they are read
    // (docs/globe.md): land 0..255 is 0..6400 m; ocean 255 is sea level and 0 is 8000 m deep (the bathymetry scale is
    // an assumption: NASA's page states only the land one).
    struct HeightmapSources {
        std::filesystem::path elevation;
        std::filesystem::path bathymetry;
        // Where the combined grid is cached; read instead of decoding the images when it is there.
        std::filesystem::path cache;
        // The images are averaged down by this factor on each axis.
        std::uint32_t downsample = 2;
    };

    inline constexpr double land_scale_metres = 6400.0;
    inline constexpr double ocean_scale_metres = 8000.0;

    [[nodiscard]] auto load_heightmap(HeightmapSources const &sources) -> std::expected<Heightmap, std::string>;

    // An RGBA8 equirectangular colour map made from elevation and latitude: a depth gradient under the sea, then
    // beach, grass, forest, rock and snow, with ice towards the poles.
    struct ColourImage {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<std::uint8_t> rgba;
    };
    [[nodiscard]] auto elevation_colours(Heightmap const &heightmap, std::uint32_t width) -> ColourImage;
    // An RGBA8 image file (the Blue Marble colour), decoded.
    [[nodiscard]] auto load_colour_image(std::filesystem::path const &path) -> std::expected<ColourImage, std::string>;

    // ---- choosing chunks ------------------------------------------------------------------------------------------

    struct LodSettings {
        double radius = 6'371'000.0;
        std::uint8_t max_level = 11;
        // A chunk splits when the camera is closer than this many of its edge lengths.
        double split_factor = 1.6;
        // How far (m) the surface may stand above and below the radius (elevation times exaggeration).
        double highest = 9000.0;
        double lowest = -11000.0;
    };

    // The chunks to draw for a camera at `camera` (planet frame): split by distance, and with what is below the
    // horizon (behind the planet) left out.
    [[nodiscard]] auto select_chunks(glm::dvec3 const &camera, LodSettings const &settings) -> std::vector<ChunkKey>;

    // Whether a sphere is hidden behind the planet (a ball of `occluder_radius`) for a camera at `camera`.
    [[nodiscard]] auto below_horizon(glm::dvec3 const &camera, double occluder_radius, glm::dvec3 const &centre,
                                     double radius) noexcept -> bool;

    // ---- chunk meshes ---------------------------------------------------------------------------------------------

    struct ChunkBuildSettings {
        double radius = 6'371'000.0;
        // Multiplies the elevation; 0 makes a smooth sphere (the ocean).
        double exaggeration = 1.0;
        // Cells along each side.
        std::uint32_t cells = 32;
        // Skirt depth as a fraction of the edge length, hiding cracks against coarser neighbours.
        double skirt = 0.015;
    };

    // A chunk's mesh in a frame of its own. The positions are scaled into [-1, 1] on each axis (the half-precision vertex
    // format would lose a large chunk's relief otherwise) and the frame undoes that:
    //   planet position = origin + rotation * (scale * vertex)
    // Normals and tangents are stored divided by the scale, so the engine's model-matrix transform of them comes out
    // right despite the non-uniform scale.
    struct ChunkMesh {
        ChunkKey key;
        glm::dvec3 origin{0.0};
        glm::dmat3 rotation{1.0};
        glm::dvec3 scale{1.0};
        // The surface's height range (m, elevation after exaggeration) inside the chunk.
        double lowest = 0.0;
        double highest = 0.0;
        std::vector<ModelVertex> vertices;
        std::vector<std::uint32_t> indices;
    };

    // `heightmap` may be null (or `exaggeration` 0) for a smooth sphere.
    [[nodiscard]] auto build_chunk(ChunkKey const &key, Heightmap const *heightmap, ChunkBuildSettings const &settings)
            -> ChunkMesh;

}
