#include "terrain/planet.hxx"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <optional>
#include <limits>
#include <numbers>

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>

#include <stb_image.h>

namespace planet {

    namespace {

        constexpr double pi = std::numbers::pi;

        struct Face {
            glm::dvec3 normal;
            glm::dvec3 u;
            glm::dvec3 v;
        };

        constexpr std::array<Face, face_count> faces{{
                {{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},
                {{-1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, {0.0, 1.0, 0.0}},
                {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}},
                {{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, -1.0}},
                {{0.0, 0.0, 1.0}, {-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
                {{0.0, 0.0, -1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
        }};

        auto wrap_unit(double value) -> double { return value - std::floor(value); }

        auto file_size_or_zero(std::filesystem::path const &path) -> std::uint64_t {
            std::error_code error;
            auto const size = std::filesystem::file_size(path, error);
            return error ? 0U : static_cast<std::uint64_t>(size);
        }

        constexpr std::array<char, 4> cache_magic{'L', 'P', 'H', '1'};

        struct CacheHeader {
            std::array<char, 4> magic = cache_magic;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t downsample = 0;
            std::uint64_t elevation_bytes = 0;
            std::uint64_t bathymetry_bytes = 0;
        };

        auto read_cache(HeightmapSources const &sources) -> std::optional<Heightmap> {
            auto file = std::ifstream{sources.cache, std::ios::binary};
            if (!file) {
                return std::nullopt;
            }
            auto header = CacheHeader{};
            file.read(reinterpret_cast<char *>(&header), sizeof(header));
            if (!file || header.magic != cache_magic || header.downsample != sources.downsample ||
                header.elevation_bytes != file_size_or_zero(sources.elevation) ||
                header.bathymetry_bytes != file_size_or_zero(sources.bathymetry) || header.width == 0 ||
                header.height == 0 || header.width > 65536 || header.height > 65536) {
                return std::nullopt;
            }
            auto map = Heightmap{.width = header.width, .height = header.height};
            map.metres.resize(static_cast<std::size_t>(header.width) * header.height);
            file.read(reinterpret_cast<char *>(map.metres.data()),
                      static_cast<std::streamsize>(map.metres.size() * sizeof(std::int16_t)));
            if (!file) {
                return std::nullopt;
            }
            return map;
        }

        auto write_cache(HeightmapSources const &sources, Heightmap const &map) -> void {
            std::error_code error;
            std::filesystem::create_directories(sources.cache.parent_path(), error);
            auto const temporary = std::filesystem::path{sources.cache.string() + ".tmp"};
            {
                auto file = std::ofstream{temporary, std::ios::binary | std::ios::trunc};
                if (!file) {
                    return;
                }
                auto const header = CacheHeader{
                        .width = map.width,
                        .height = map.height,
                        .downsample = sources.downsample,
                        .elevation_bytes = file_size_or_zero(sources.elevation),
                        .bathymetry_bytes = file_size_or_zero(sources.bathymetry),
                };
                file.write(reinterpret_cast<char const *>(&header), sizeof(header));
                file.write(reinterpret_cast<char const *>(map.metres.data()),
                           static_cast<std::streamsize>(map.metres.size() * sizeof(std::int16_t)));
            }
            std::filesystem::rename(temporary, sources.cache, error);
        }

        // Decodes an 8-bit grey image and sums it into blocks of `downsample` pixels.
        struct Block {
            std::vector<double> sum;
            std::vector<std::uint32_t> count;
        };

        template<typename Accumulate>
        auto decode_blocks(std::filesystem::path const &path, std::uint32_t downsample, std::uint32_t &width,
                           std::uint32_t &height, Accumulate &&accumulate) -> std::expected<void, std::string> {
            int w = 0;
            int h = 0;
            int channels = 0;
            auto *const pixels = stbi_load(path.string().c_str(), &w, &h, &channels, 1);
            if (pixels == nullptr) {
                return std::unexpected(std::format("could not decode '{}': {}", path.string(),
                                                   stbi_failure_reason() != nullptr ? stbi_failure_reason() : "?"));
            }
            auto const out_width = static_cast<std::uint32_t>(w) / downsample;
            auto const out_height = static_cast<std::uint32_t>(h) / downsample;
            if (width != 0 && (out_width != width || out_height != height)) {
                stbi_image_free(pixels);
                return std::unexpected(std::format("'{}' is not the size of the other heightmap image", path.string()));
            }
            width = out_width;
            height = out_height;

            for (std::uint32_t y = 0; y < out_height * downsample; ++y) {
                auto const *const row = pixels + static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
                auto const out_row = static_cast<std::size_t>(y / downsample) * out_width;
                for (std::uint32_t x = 0; x < out_width * downsample; ++x) {
                    accumulate(out_row + x / downsample, row[x]);
                }
            }
            stbi_image_free(pixels);
            return {};
        }

        auto lerp3(glm::dvec3 a, glm::dvec3 b, double t) -> glm::dvec3 { return a + (b - a) * t; }

    }

    auto direction_of(LatLon position) noexcept -> glm::dvec3 {
        auto const c = std::cos(position.latitude);
        return glm::dvec3{-c * std::sin(position.longitude), std::sin(position.latitude),
                          c * std::cos(position.longitude)};
    }

    auto lat_lon_of(glm::dvec3 const &direction) noexcept -> LatLon {
        auto const d = glm::normalize(direction);
        return LatLon{.latitude = std::asin(std::clamp(d.y, -1.0, 1.0)), .longitude = std::atan2(-d.x, d.z)};
    }

    auto equirect_uv(LatLon position) noexcept -> glm::dvec2 {
        return glm::dvec2{wrap_unit((position.longitude + pi) / (2.0 * pi)),
                          std::clamp((pi * 0.5 - position.latitude) / pi, 0.0, 1.0)};
    }

    auto cube_direction(std::uint32_t face, double s, double t) noexcept -> glm::dvec3 {
        auto const &f = faces[face % face_count];
        auto const u = std::tan(s * pi * 0.25);
        auto const v = std::tan(t * pi * 0.25);
        return glm::normalize(f.normal + f.u * u + f.v * v);
    }

    auto chunk_extent(ChunkKey const &key) noexcept -> std::array<double, 4> {
        auto const size = 2.0 / static_cast<double>(1U << key.level);
        auto const s0 = -1.0 + size * static_cast<double>(key.x);
        auto const t0 = -1.0 + size * static_cast<double>(key.y);
        return {s0, t0, s0 + size, t0 + size};
    }

    auto chunk_centre_direction(ChunkKey const &key) noexcept -> glm::dvec3 {
        auto const extent = chunk_extent(key);
        return cube_direction(key.face, (extent[0] + extent[2]) * 0.5, (extent[1] + extent[3]) * 0.5);
    }

    auto chunk_edge_length(std::uint8_t level, double radius) noexcept -> double {
        return pi * 0.5 * radius / static_cast<double>(1U << level);
    }

    // ---- heightmap ------------------------------------------------------------------------------------------------

    auto Heightmap::sample(LatLon position) const noexcept -> double {
        if (!valid()) {
            return 0.0;
        }
        auto const uv = equirect_uv(position);
        auto const x = uv.x * static_cast<double>(width) - 0.5;
        auto const y = std::clamp(uv.y * static_cast<double>(height) - 0.5, 0.0, static_cast<double>(height - 1U));
        auto const x0 = std::floor(x);
        auto const y0 = std::floor(y);
        auto const fx = x - x0;
        auto const fy = y - y0;

        auto const column = [&](double value) {
            auto const wrapped = static_cast<std::int64_t>(value) % static_cast<std::int64_t>(width);
            return static_cast<std::size_t>(wrapped < 0 ? wrapped + width : wrapped);
        };
        auto const row = [&](double value) {
            return static_cast<std::size_t>(std::clamp(value, 0.0, static_cast<double>(height - 1U)));
        };
        auto const at = [&](std::size_t r, std::size_t c) {
            return static_cast<double>(metres[r * width + c]);
        };

        auto const c0 = column(x0);
        auto const c1 = column(x0 + 1.0);
        auto const r0 = row(y0);
        auto const r1 = row(y0 + 1.0);
        auto const top = at(r0, c0) * (1.0 - fx) + at(r0, c1) * fx;
        auto const bottom = at(r1, c0) * (1.0 - fx) + at(r1, c1) * fx;
        return top * (1.0 - fy) + bottom * fy;
    }

    auto Heightmap::find_range() const noexcept -> void {
        if (range_known_) {
            return;
        }
        if (!metres.empty()) {
            auto const [low, high] = std::ranges::minmax_element(metres);
            lowest_ = static_cast<double>(*low);
            highest_ = static_cast<double>(*high);
        }
        range_known_ = true;
    }

    auto Heightmap::lowest() const noexcept -> double {
        find_range();
        return lowest_;
    }

    auto Heightmap::highest() const noexcept -> double {
        find_range();
        return highest_;
    }

    auto load_heightmap(HeightmapSources const &sources) -> std::expected<Heightmap, std::string> {
        if (!sources.cache.empty()) {
            if (auto cached = read_cache(sources)) {
                return std::move(*cached);
            }
        }

        auto const downsample = std::max(sources.downsample, 1U);
        auto width = std::uint32_t{0};
        auto height = std::uint32_t{0};

        // Land: the mean elevation of every pixel (0 over the sea). Sea: the summed depth of the sea pixels and their
        // count. A block is sea when most of its pixels are.
        auto land_sum = std::vector<double>{};
        auto sea_sum = std::vector<double>{};
        auto sea_count = std::vector<std::uint32_t>{};

        auto land = decode_blocks(sources.elevation, downsample, width, height, [&](std::size_t index, std::uint8_t value) {
            if (land_sum.empty()) {
                land_sum.assign(static_cast<std::size_t>(width) * height, 0.0);
            }
            land_sum[index] += static_cast<double>(value) * (land_scale_metres / 255.0);
        });
        if (!land) {
            return std::unexpected(land.error());
        }

        auto sea = decode_blocks(sources.bathymetry, downsample, width, height, [&](std::size_t index, std::uint8_t value) {
            if (sea_sum.empty()) {
                sea_sum.assign(static_cast<std::size_t>(width) * height, 0.0);
                sea_count.assign(static_cast<std::size_t>(width) * height, 0U);
            }
            if (value < 255U) {
                sea_sum[index] -= static_cast<double>(255U - value) * (ocean_scale_metres / 255.0);
                ++sea_count[index];
            }
        });
        if (!sea) {
            return std::unexpected(sea.error());
        }

        auto map = Heightmap{.width = width, .height = height};
        map.metres.resize(static_cast<std::size_t>(width) * height);
        auto const block = static_cast<double>(downsample * downsample);
        for (std::size_t i = 0; i < map.metres.size(); ++i) {
            auto const sea_pixels = static_cast<double>(sea_count[i]);
            double metres = 0.0;
            if (sea_pixels * 2.0 > block) {
                // The sea is always a little below sea level, so it never ties with the ocean surface.
                metres = std::min(sea_sum[i] / sea_pixels, -10.0);
            } else {
                // Low land is a little above it.
                metres = std::max(land_sum[i] / std::max(block - sea_pixels, 1.0), 2.0);
            }
            map.metres[i] = static_cast<std::int16_t>(std::clamp(std::lround(metres), -32000L, 32000L));
        }

        if (!sources.cache.empty()) {
            write_cache(sources, map);
        }
        return map;
    }

    auto elevation_colours(Heightmap const &heightmap, std::uint32_t width) -> ColourImage {
        auto image = ColourImage{.width = width, .height = std::max(width / 2U, 1U)};
        image.rgba.resize(static_cast<std::size_t>(image.width) * image.height * 4U);

        struct Stop {
            double metres;
            glm::dvec3 colour;
        };
        constexpr std::array<Stop, 9> stops{{
                {-8000.0, {0.02, 0.05, 0.16}},
                {-3000.0, {0.05, 0.12, 0.30}},
                {-200.0, {0.12, 0.35, 0.50}},
                {-5.0, {0.55, 0.62, 0.55}},
                {10.0, {0.76, 0.70, 0.50}},
                {200.0, {0.25, 0.42, 0.18}},
                {1200.0, {0.33, 0.36, 0.22}},
                {2800.0, {0.45, 0.40, 0.36}},
                {4500.0, {0.95, 0.95, 0.97}},
        }};

        for (std::uint32_t y = 0; y < image.height; ++y) {
            auto const latitude = pi * 0.5 - (static_cast<double>(y) + 0.5) / image.height * pi;
            for (std::uint32_t x = 0; x < image.width; ++x) {
                auto const longitude = (static_cast<double>(x) + 0.5) / image.width * 2.0 * pi - pi;
                auto const metres = heightmap.sample(LatLon{.latitude = latitude, .longitude = longitude});

                auto colour = stops.back().colour;
                for (std::size_t i = 1; i < stops.size(); ++i) {
                    if (metres < stops[i].metres) {
                        auto const t = std::clamp((metres - stops[i - 1].metres) / (stops[i].metres - stops[i - 1].metres),
                                                  0.0, 1.0);
                        colour = lerp3(stops[i - 1].colour, stops[i].colour, t);
                        break;
                    }
                }
                // Snow and ice towards the poles, on land and over the polar seas.
                auto const polar = std::clamp((std::abs(latitude) - 1.1) / 0.15, 0.0, 1.0);
                colour = lerp3(colour, glm::dvec3{0.92, 0.94, 0.97}, polar);

                auto *const out = image.rgba.data() + (static_cast<std::size_t>(y) * image.width + x) * 4U;
                for (int c = 0; c < 3; ++c) {
                    out[c] = static_cast<std::uint8_t>(std::clamp(std::pow(colour[c], 1.0 / 2.2) * 255.0, 0.0, 255.0));
                }
                out[3] = 255U;
            }
        }
        return image;
    }

    auto load_colour_image(std::filesystem::path const &path) -> std::expected<ColourImage, std::string> {
        int w = 0;
        int h = 0;
        int channels = 0;
        auto *const pixels = stbi_load(path.string().c_str(), &w, &h, &channels, 4);
        if (pixels == nullptr) {
            return std::unexpected(std::format("could not decode '{}'", path.string()));
        }
        auto image = ColourImage{.width = static_cast<std::uint32_t>(w), .height = static_cast<std::uint32_t>(h)};
        image.rgba.assign(pixels, pixels + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4U);
        stbi_image_free(pixels);
        return image;
    }

    // ---- LOD --------------------------------------------------------------------------------------------------------

    auto below_horizon(glm::dvec3 const &camera, double occluder_radius, glm::dvec3 const &centre,
                       double radius) noexcept -> bool {
        auto const camera_distance = glm::length(camera);
        if (camera_distance <= occluder_radius) {
            return false;
        }
        // The camera sees up to its horizon on the occluder and, beyond it, whatever rises above the occluder.
        auto const horizon = std::sqrt(camera_distance * camera_distance - occluder_radius * occluder_radius);
        auto const top = glm::length(centre) + radius;
        auto const beyond = top > occluder_radius ? std::sqrt(top * top - occluder_radius * occluder_radius) : 0.0;
        return glm::length(centre - camera) - radius > horizon + beyond;
    }

    auto select_chunks(glm::dvec3 const &camera, LodSettings const &settings) -> std::vector<ChunkKey> {
        auto chunks = std::vector<ChunkKey>{};
        auto const occluder = settings.radius + std::min(settings.lowest, 0.0);
        auto const relief = std::max(settings.highest, 0.0) - std::min(settings.lowest, 0.0);

        auto const visit = [&](auto &&self, ChunkKey const &key) -> void {
            auto const edge = chunk_edge_length(key.level, settings.radius);
            auto const centre = chunk_centre_direction(key) * settings.radius;
            auto const bound = edge * 0.75 + relief + edge * edge / (4.0 * settings.radius);

            if (key.level > 0 && below_horizon(camera, occluder, centre, bound)) {
                return;
            }

            auto const distance = std::max(glm::length(camera - centre) - bound, 0.0);
            if (key.level < settings.max_level && distance < settings.split_factor * edge) {
                for (std::uint32_t child = 0; child < 4; ++child) {
                    self(self, key.child(child));
                }
                return;
            }
            chunks.push_back(key);
        };

        for (std::uint8_t face = 0; face < face_count; ++face) {
            visit(visit, ChunkKey{.face = face});
        }
        return chunks;
    }

    // ---- chunk meshes ---------------------------------------------------------------------------------------------

    auto build_chunk(ChunkKey const &key, Heightmap const *heightmap, ChunkBuildSettings const &settings) -> ChunkMesh {
        auto const cells = std::max(settings.cells, 2U);
        auto const stride = cells + 3U; // one ring of neighbours around the chunk, for the normals
        auto const extent = chunk_extent(key);
        auto const ds = (extent[2] - extent[0]) / cells;
        auto const dt = (extent[3] - extent[1]) / cells;
        auto const relief = heightmap != nullptr && heightmap->valid() && settings.exaggeration != 0.0;
        // A flat triangle sags below the sphere between its corners by up to cell^2 / (8 R). The ocean's chunks are often
        // finer than the terrain's (or the terrain is still streaming in), and coarse terrain would then dip under the
        // sea; lifting the terrain by half the sag keeps a coarse coast above the water.
        auto const cell = chunk_edge_length(key.level, settings.radius) / cells;
        auto const lift = relief ? 0.5 * cell * cell / (8.0 * settings.radius) : 0.0;

        auto mesh = ChunkMesh{.key = key};
        mesh.lowest = std::numeric_limits<double>::max();
        mesh.highest = std::numeric_limits<double>::lowest();

        // Planet positions of the grid and its border, and the elevations.
        auto directions = std::vector<glm::dvec3>(static_cast<std::size_t>(stride) * stride);
        auto positions = std::vector<glm::dvec3>(directions.size());
        auto elevations = std::vector<double>(directions.size());
        for (std::uint32_t j = 0; j < stride; ++j) {
            for (std::uint32_t i = 0; i < stride; ++i) {
                auto const s = extent[0] + (static_cast<double>(i) - 1.0) * ds;
                auto const t = extent[1] + (static_cast<double>(j) - 1.0) * dt;
                auto const index = static_cast<std::size_t>(j) * stride + i;
                auto const direction = cube_direction(key.face, s, t);
                auto const elevation = relief ? heightmap->sample(direction) * settings.exaggeration + lift : 0.0;
                directions[index] = direction;
                elevations[index] = elevation;
                positions[index] = direction * (settings.radius + elevation);
                if (i >= 1 && j >= 1 && i <= cells + 1 && j <= cells + 1) {
                    mesh.lowest = std::min(mesh.lowest, elevation);
                    mesh.highest = std::max(mesh.highest, elevation);
                }
            }
        }
        auto const at = [&](std::uint32_t i, std::uint32_t j) { return static_cast<std::size_t>(j) * stride + i; };

        // The chunk's frame: up through its centre, right along the face's s axis.
        auto const up = chunk_centre_direction(key);
        auto const along = positions[at(cells + 1, cells / 2 + 1)] - positions[at(1, cells / 2 + 1)];
        auto const right = glm::normalize(along - up * glm::dot(along, up));
        auto const forward = glm::cross(right, up);
        mesh.rotation = glm::dmat3{right, up, forward};
        auto const to_local = glm::transpose(mesh.rotation);
        auto const anchor = up * settings.radius;

        // Skirt vertices hang below each edge vertex.
        auto const skirt_depth = chunk_edge_length(key.level, settings.radius) * settings.skirt;

        auto low = glm::dvec3{std::numeric_limits<double>::max()};
        auto high = glm::dvec3{std::numeric_limits<double>::lowest()};
        auto const include = [&](glm::dvec3 const &position) {
            auto const local = to_local * (position - anchor);
            low = glm::min(low, local);
            high = glm::max(high, local);
        };
        for (std::uint32_t j = 1; j <= cells + 1; ++j) {
            for (std::uint32_t i = 1; i <= cells + 1; ++i) {
                include(positions[at(i, j)]);
                include(positions[at(i, j)] - directions[at(i, j)] * skirt_depth);
            }
        }
        auto const middle = (low + high) * 0.5;
        mesh.scale = glm::max((high - low) * 0.5, glm::dvec3{1e-3});
        mesh.origin = anchor + mesh.rotation * middle;

        auto const centre_uv = equirect_uv(lat_lon_of(up));
        auto const vertex_at = [&](std::uint32_t i, std::uint32_t j, glm::dvec3 const &position) {
            auto const outward = directions[at(i, j)];
            auto normal = glm::normalize(glm::cross(positions[at(i + 1, j)] - positions[at(i - 1, j)],
                                                    positions[at(i, j + 1)] - positions[at(i, j - 1)]));
            if (glm::dot(normal, outward) < 0.0) {
                normal = -normal;
            }
            auto const tangent = glm::normalize(positions[at(i + 1, j)] - positions[at(i - 1, j)]);

            auto uv = equirect_uv(lat_lon_of(outward));
            // Keep a chunk's texture coordinates continuous across the date line (the sampler repeats).
            if (uv.x - centre_uv.x > 0.5) {
                uv.x -= 1.0;
            } else if (uv.x - centre_uv.x < -0.5) {
                uv.x += 1.0;
            }

            auto const local = (to_local * (position - mesh.origin)) / mesh.scale;
            return ModelVertex{
                    .position = glm::vec3{local},
                    .normal = glm::vec3{glm::normalize((to_local * normal) / mesh.scale)},
                    .tangent = glm::vec4{glm::vec3{glm::normalize((to_local * tangent) / mesh.scale)}, 1.0F},
                    .texcoord = glm::vec2{uv},
            };
        };

        auto const side = cells + 1;
        mesh.vertices.reserve(static_cast<std::size_t>(side) * side + 4U * side);
        for (std::uint32_t j = 1; j <= cells + 1; ++j) {
            for (std::uint32_t i = 1; i <= cells + 1; ++i) {
                mesh.vertices.push_back(vertex_at(i, j, positions[at(i, j)]));
            }
        }
        auto const grid_index = [&](std::uint32_t i, std::uint32_t j) { return (j - 1) * side + (i - 1); };

        // Front faces turn their winding towards the outside (cross(b - a, c - a) along the outward direction).
        auto const triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c, glm::dvec3 const &outward,
                                  glm::dvec3 const &pa, glm::dvec3 const &pb, glm::dvec3 const &pc) {
            if (glm::dot(glm::cross(pb - pa, pc - pa), outward) >= 0.0) {
                mesh.indices.insert(mesh.indices.end(), {a, b, c});
            } else {
                mesh.indices.insert(mesh.indices.end(), {a, c, b});
            }
        };

        for (std::uint32_t j = 1; j <= cells; ++j) {
            for (std::uint32_t i = 1; i <= cells; ++i) {
                auto const outward = directions[at(i, j)];
                auto const p00 = positions[at(i, j)];
                auto const p10 = positions[at(i + 1, j)];
                auto const p01 = positions[at(i, j + 1)];
                auto const p11 = positions[at(i + 1, j + 1)];
                triangle(grid_index(i, j), grid_index(i + 1, j), grid_index(i + 1, j + 1), outward, p00, p10, p11);
                triangle(grid_index(i, j), grid_index(i + 1, j + 1), grid_index(i, j + 1), outward, p00, p11, p01);
            }
        }

        // Skirts: each edge vertex again, lowered, and both windings of the strip so it shows from either side.
        auto const add_skirt = [&](auto &&edge_vertex) {
            auto first_skirt = static_cast<std::uint32_t>(mesh.vertices.size());
            for (std::uint32_t k = 0; k <= cells; ++k) {
                auto const [i, j] = edge_vertex(k);
                mesh.vertices.push_back(vertex_at(i, j, positions[at(i, j)] - directions[at(i, j)] * skirt_depth));
            }
            for (std::uint32_t k = 0; k < cells; ++k) {
                auto const [i0, j0] = edge_vertex(k);
                auto const [i1, j1] = edge_vertex(k + 1);
                auto const top0 = grid_index(i0, j0);
                auto const top1 = grid_index(i1, j1);
                auto const bottom0 = first_skirt + k;
                auto const bottom1 = first_skirt + k + 1;
                mesh.indices.insert(mesh.indices.end(), {top0, top1, bottom1, top0, bottom1, bottom0});
                mesh.indices.insert(mesh.indices.end(), {top0, bottom1, top1, top0, bottom0, bottom1});
            }
        };
        using Pair = std::pair<std::uint32_t, std::uint32_t>;
        add_skirt([&](std::uint32_t k) { return Pair{1 + k, 1}; });
        add_skirt([&](std::uint32_t k) { return Pair{1 + k, cells + 1}; });
        add_skirt([&](std::uint32_t k) { return Pair{1, 1 + k}; });
        add_skirt([&](std::uint32_t k) { return Pair{cells + 1, 1 + k}; });

        if (mesh.lowest > mesh.highest) {
            mesh.lowest = mesh.highest = 0.0;
        }
        return mesh;
    }

}
