#include "rendering/planet_system.hxx"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <ranges>
#include <unordered_set>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <tracy/Tracy.hpp>

#include "assets/primitive_meshes.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/thread_pool.hxx"
#include "rendering/game_gpu.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/water_system.hxx"
#include "rendering/water_waves.hxx"

struct PlanetSystem::Built {
    std::shared_ptr<planet::ChunkMesh> frame;
    ModelCpuData model;
};

namespace {

    // Colours of the quadtree levels for Planet::debug_lod.
    constexpr std::array<glm::vec3, 16> level_colours{{
            {0.9F, 0.2F, 0.2F}, {0.9F, 0.6F, 0.2F}, {0.9F, 0.9F, 0.2F}, {0.5F, 0.9F, 0.2F},
            {0.2F, 0.9F, 0.4F}, {0.2F, 0.9F, 0.9F}, {0.2F, 0.5F, 0.9F}, {0.3F, 0.2F, 0.9F},
            {0.7F, 0.2F, 0.9F}, {0.9F, 0.2F, 0.7F}, {0.9F, 0.9F, 0.9F}, {0.6F, 0.6F, 0.6F},
            {0.6F, 0.3F, 0.1F}, {0.1F, 0.6F, 0.3F}, {0.1F, 0.3F, 0.6F}, {0.3F, 0.3F, 0.3F},
    }};

    // The planet's frame in the world: the entity's world transform with its scale taken out.
    auto planet_matrix(glm::mat4 const &world) -> glm::dmat4 {
        auto matrix = glm::dmat4{world};
        for (int axis = 0; axis < 3; ++axis) {
            auto const column = glm::dvec3{matrix[axis]};
            auto const length = glm::length(column);
            matrix[axis] = glm::dvec4{length > 1e-12 ? column / length : glm::dvec3{0.0}, 0.0};
        }
        return matrix;
    }

    auto sea_radius(Components::Planet const &planet) -> double {
        return planet.radius + static_cast<double>(planet.sea_level);
    }

    auto files_present(Components::Planet const &planet) -> bool {
        std::error_code error;
        return !planet.elevation_path.empty() && !planet.bathymetry_path.empty() &&
               std::filesystem::exists(planet.elevation_path, error) &&
               std::filesystem::exists(planet.bathymetry_path, error);
    }

    // Whether two planets differ in something that changes the chunks' geometry.
    auto same_geometry(Components::Planet const &a, Components::Planet const &b) -> bool {
        return a.radius == b.radius && a.elevation_path == b.elevation_path && a.bathymetry_path == b.bathymetry_path &&
               a.exaggeration == b.exaggeration && a.sea_level == b.sea_level;
    }

    auto same_colour(Components::Planet const &a, Components::Planet const &b) -> bool {
        return a.colour == b.colour && a.colour_path == b.colour_path && a.elevation_path == b.elevation_path &&
               a.bathymetry_path == b.bathymetry_path;
    }

}

PlanetSystem::~PlanetSystem() = default;

auto PlanetSystem::create(GameGpu &gpu) -> std::expected<void, RendererError> {
    if (atmosphere_shader_) {
        return {};
    }
    auto shader = gpu.register_compute({
            .source = "assets/shaders/atmosphere.slang",
            .debug_name = "planet_atmosphere",
    });
    if (!shader) {
        return std::unexpected(shader.error());
    }
    gpu_ = &gpu;
    atmosphere_shader_ = *shader;
    return {};
}

auto PlanetSystem::chunk_matrix(glm::dmat4 const &planet, planet::ChunkMesh const &chunk, glm::dvec3 const &origin)
        -> glm::mat4 {
    auto const world_origin = glm::dvec3{planet * glm::dvec4{chunk.origin, 1.0}};
    auto const rotation = glm::dmat3{planet} * chunk.rotation;

    auto matrix = glm::dmat4{rotation};
    matrix[0] *= chunk.scale.x;
    matrix[1] *= chunk.scale.y;
    matrix[2] *= chunk.scale.z;
    matrix[3] = glm::dvec4{world_origin - origin, 1.0};
    return glm::mat4{matrix};
}

auto PlanetSystem::start_loading(Body &body) -> void {
    auto const &planet = body.planet;
    if (!files_present(planet)) {
        return;
    }

    auto const elevation = std::filesystem::path{planet.elevation_path};
    auto const sources = planet::HeightmapSources{
            .elevation = elevation,
            .bathymetry = planet.bathymetry_path,
            .cache = elevation.parent_path() / std::format("{}.lph", elevation.stem().string()),
            .downsample = 2,
    };
    body.heightmap_loading = thread_pool().submit_task([sources] {
        ZoneScopedNC("Planet heightmap load", tracy::Color::SeaGreen);
        return planet::load_heightmap(sources);
    });
    info("[Planet] Loading the elevation map from '{}' and '{}'", planet.elevation_path, planet.bathymetry_path);
}

auto PlanetSystem::poll_loading(Renderer &renderer, Body &body) -> void {
    if (body.heightmap_loading.valid() &&
        body.heightmap_loading.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
        auto loaded = body.heightmap_loading.get();
        if (!loaded) {
            error("[Planet] Could not load the elevation map: {}", loaded.error());
            body.load_failed = true;
        } else {
            info("[Planet] Elevation map ready: {} x {}, {:.0f} m to {:.0f} m", loaded->width, loaded->height,
                 loaded->lowest(), loaded->highest());
            body.heightmap = std::make_shared<planet::Heightmap const>(std::move(*loaded));
            // The smooth placeholder chunks give way to real ones.
            for (auto *chunks: {&body.terrain}) {
                for (auto &chunk: *chunks | std::views::values) {
                    if (chunk.model.valid()) {
                        retired_.push_back(Retired{.model = chunk.model});
                    }
                }
                chunks->clear();
            }

            // The colour: the image, or one made from the elevation.
            auto const planet = body.planet;
            auto const heightmap = body.heightmap;
            body.colour_loading = thread_pool().submit_task(
                    [planet, heightmap]() -> std::expected<planet::ColourImage, std::string> {
                        ZoneScopedNC("Planet colour", tracy::Color::SeaGreen);
                        std::error_code error;
                        if (planet.colour == Components::PlanetColour::image && !planet.colour_path.empty() &&
                            std::filesystem::exists(planet.colour_path, error)) {
                            return planet::load_colour_image(planet.colour_path);
                        }
                        return planet::elevation_colours(*heightmap, 4096);
                    });
        }
    }

    if (body.colour_loading.valid() &&
        body.colour_loading.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
        auto colour = body.colour_loading.get();
        if (!colour) {
            error("[Planet] Could not load the planet's colour: {}", colour.error());
            return;
        }
        auto const largest = std::max(colour->width, colour->height);
        auto const mips = static_cast<std::uint32_t>(std::floor(std::log2(static_cast<double>(largest)))) + 1U;
        auto image = renderer.image_storage().create_image(
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = colour->width, .height = colour->height, .depth = 1},
                        .format = VK_FORMAT_R8G8B8A8_SRGB,
                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .mip_levels = mips,
                        .array_layers = 1,
                        .debug_name = "planet.colour",
                },
                std::as_bytes(std::span{colour->rgba}), ImageMipSource::generate);
        if (!image) {
            error("[Planet] Could not create the colour image: {}", describe(image.error()));
            return;
        }
        if (body.colour_image.valid()) {
            retired_.push_back(Retired{.image = body.colour_image});
        }
        body.colour_image = *image;
        if (body.terrain_material.valid()) {
            // Remade with the texture below.
            retired_.push_back(Retired{.material = body.terrain_material});
            body.terrain_material = {};
        }
        info("[Planet] Colour ready ({} x {})", colour->width, colour->height);
    }
}

auto PlanetSystem::ensure_materials(Renderer &renderer, Body &body) -> void {
    auto &images = renderer.image_storage();
    auto const base = MaterialCreateInfo{
            .base_colour_factor = glm::vec4{1.0F},
            .metallic_factor = 0.0F,
            .roughness_factor = 0.9F,
            .base_colour_texture = images.white(),
            .normal_texture = images.flat_normal(),
            .metallic_roughness_texture = images.white(),
            .occlusion_texture = images.occlusion(),
            .emissive_texture = images.emissive(),
            .sampler = renderer.sampler_storage().linear_repeat(),
            .max_shadow_cascade = GpuMaterial::no_shadow_cascade,
    };

    if (!body.terrain_material.valid()) {
        auto info = base;
        info.planet_surface = true;
        if (body.colour_image.valid()) {
            info.base_colour_texture = body.colour_image;
        } else {
            info.base_colour_factor = glm::vec4{0.35F, 0.4F, 0.45F, 1.0F};
        }
        if (auto created = renderer.create_material(info)) {
            body.terrain_material = *created;
            body.terrain_info = info;
        }
    }

    if (body.planet.debug_lod && !body.debug_materials[0].valid()) {
        for (std::size_t level = 0; level < body.debug_materials.size(); ++level) {
            auto info = base;
            info.base_colour_factor = glm::vec4{level_colours[level], 1.0F};
            if (auto created = renderer.create_material(info)) {
                body.debug_materials[level] = *created;
            }
        }
    }

    if (body.planet.ocean && !body.ocean_material.valid()) {
        auto water = body.planet.ocean_water;
        water.body = Components::WaterBody::lake; // no water zone
        auto info = WaterSystem::material_info(renderer, water, glm::mat4{1.0F});
        info.water_settings.sphere_centre = glm::vec4{0.0F, 0.0F, 0.0F, 1.0F};
        if (auto created = renderer.create_material(info)) {
            body.ocean_material = *created;
        }
    }
}

auto PlanetSystem::update_ocean_material(Renderer &renderer, Body &body, glm::dmat4 const &planet,
                                         glm::dvec3 const &camera, PlanetView const &view) -> void {
    if (!body.ocean_material.valid()) {
        return;
    }

    auto water = body.planet.ocean_water;
    water.body = Components::WaterBody::lake;
    auto info = WaterSystem::material_info(renderer, water, glm::mat4{1.0F});
    auto const radius = sea_radius(body.planet);

    // The wave anchor: the point under the camera, snapped to a 0.01 degree grid so it holds still between snaps.
    constexpr double snap = 0.01 * std::numbers::pi / 180.0;
    auto position = planet::lat_lon_of(camera);
    position.latitude = std::round(position.latitude / snap) * snap;
    position.longitude = std::round(position.longitude / snap) * snap;
    auto const anchor = planet::direction_of(position) * radius;

    auto const sin_lat = std::sin(position.latitude);
    auto const cos_lat = std::cos(position.latitude);
    auto const sin_lon = std::sin(position.longitude);
    auto const cos_lon = std::cos(position.longitude);
    auto const east = glm::dvec3{-cos_lon, 0.0, -sin_lon};
    auto const north = glm::dvec3{sin_lat * sin_lon, cos_lat, -sin_lat * cos_lon};

    // The anchor's coordinates in the wave plane: metres east along its parallel and north along its meridian.
    auto const coordinates = glm::dvec2{position.longitude * radius * cos_lat, position.latitude * radius};
    auto const phases = water_waves::anchor_phases(coordinates, static_cast<double>(water.wave_scale));

    auto const rotation = glm::dmat3{planet};
    auto const centre = glm::dvec3{planet[3]} - view.origin;
    info.water_settings.sphere_centre = glm::vec4{glm::vec3{centre}, 1.0F};
    info.water_settings.anchor = glm::vec4{glm::vec3{glm::dvec3{planet * glm::dvec4{anchor, 1.0}} - view.origin}, 0.0F};
    info.water_settings.east = glm::vec4{glm::vec3{rotation * east}, 0.0F};
    info.water_settings.north = glm::vec4{glm::vec3{rotation * north}, 0.0F};
    for (std::size_t i = 0; i < phases.size(); ++i) {
        info.water_settings.phases[i / 4][static_cast<glm::length_t>(i % 4)] = phases[i];
    }

    if (auto const updated = renderer.update_material(body.ocean_material, info); !updated) {
        error("[Planet] Could not update the ocean: {}", describe(updated.error()));
    }
}

auto PlanetSystem::draw_tree(Renderer &renderer, Body &body, ChunkMap &chunks,
                             std::vector<planet::ChunkKey> const &wanted, bool ocean, glm::dmat4 const &planet,
                             PlanetView const &view, std::uint32_t &builds, std::uint32_t &uploads) -> std::size_t {
    auto const settings = planet::ChunkBuildSettings{
            .radius = ocean ? sea_radius(body.planet) : body.planet.radius,
            .exaggeration = ocean ? 0.0 : static_cast<double>(body.planet.exaggeration),
            .cells = ocean ? ocean_cells : terrain_cells,
            .skirt = ocean ? 0.004 : 0.015,
    };
    auto const heightmap = ocean ? std::shared_ptr<planet::Heightmap const>{} : body.heightmap;

    // The six faces and their first split are always kept, so a chunk still being built always has a ready
    // ancestor to stand in for it and the planet never shows a hole.
    auto pinned = std::vector<planet::ChunkKey>{};
    for (std::uint8_t face = 0; face < planet::face_count; ++face) {
        auto const root = planet::ChunkKey{.face = face};
        pinned.push_back(root);
        for (std::uint32_t child = 0; child < 4; ++child) {
            pinned.push_back(root.child(child));
        }
    }

    // Start what is missing, coarse pinned chunks first, then the wanted ones (coarse before fine anyway).
    // Builds in flight are capped too: queued work for chunks the camera has already left would delay the ones it
    // needs now.
    auto in_flight = std::size_t{0};
    for (auto const &chunk: chunks | std::views::values) {
        in_flight += chunk.building.valid() ? 1U : 0U;
    }
    auto const request = [&](planet::ChunkKey const &key) {
        auto &chunk = chunks[key];
        chunk.last_used = frame_;
        if (chunk.model.valid() || chunk.building.valid() || builds >= builds_per_frame ||
            in_flight >= max_in_flight) {
            return;
        }
        ++builds;
        ++in_flight;
        chunk.building = thread_pool().submit_task([key, heightmap, settings] {
            ZoneScopedNC("Planet chunk build", tracy::Color::SeaGreen);
            auto mesh = planet::build_chunk(key, heightmap.get(), settings);
            auto built = std::make_shared<Built>();
            built->model = to_model_cpu_data(
                    PrimitiveMeshData{.vertices = std::move(mesh.vertices), .indices = std::move(mesh.indices)});
            built->frame = std::make_shared<planet::ChunkMesh>(std::move(mesh));
            return built;
        });
    };
    std::ranges::for_each(pinned, request);
    std::ranges::for_each(wanted, request);

    // Upload what has finished, within a time budget so a burst of finished chunks does not stall the frame.
    auto const upload_start = std::chrono::steady_clock::now();
    for (auto &chunk: chunks | std::views::values) {
        if (uploads >= uploads_per_frame || std::chrono::steady_clock::now() - upload_start > upload_budget) {
            break;
        }
        if (!chunk.building.valid() || chunk.building.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
            continue;
        }
        auto built = chunk.building.get();
        if (auto model = renderer.create_model_from_cpu_data(built->model)) {
            chunk.model = *model;
            chunk.mesh = built->frame;
            ++uploads;
        } else {
            error("[Planet] Could not upload a chunk: {}", describe(model.error()));
        }
    }

    // What to draw: each wanted chunk, or while it is not ready its four children (if they are) or its nearest ready
    // ancestor, never both a chunk and one of its ancestors.
    auto const ready = [&](planet::ChunkKey const &key) {
        auto const found = chunks.find(key);
        return found != chunks.end() && found->second.model.valid();
    };
    auto draws = std::vector<planet::ChunkKey>{};
    auto fallbacks = std::unordered_set<planet::ChunkKey, planet::ChunkKeyHash>{};
    for (auto const &key: wanted) {
        if (ready(key)) {
            draws.push_back(key);
            continue;
        }
        auto children_ready = key.level < 15;
        for (std::uint32_t child = 0; child < 4 && children_ready; ++child) {
            children_ready = ready(key.child(child));
        }
        if (children_ready) {
            for (std::uint32_t child = 0; child < 4; ++child) {
                draws.push_back(key.child(child));
            }
            continue;
        }
        for (auto ancestor = key; ancestor.level > 0;) {
            ancestor = ancestor.parent();
            if (ready(ancestor)) {
                fallbacks.insert(ancestor);
                break;
            }
        }
    }
    auto const has_fallback_ancestor = [&](planet::ChunkKey key) {
        while (key.level > 0) {
            key = key.parent();
            if (fallbacks.contains(key)) {
                return true;
            }
        }
        return false;
    };
    std::erase_if(draws, has_fallback_ancestor);
    for (auto const &fallback: fallbacks) {
        if (!has_fallback_ancestor(fallback)) {
            draws.push_back(fallback);
        }
    }

    auto drawn = std::size_t{0};
    for (auto const &key: draws) {
        auto &chunk = chunks[key];
        chunk.last_used = frame_;
        auto material = ocean ? body.ocean_material : body.terrain_material;
        if (!ocean && body.planet.debug_lod && body.debug_materials[std::min<std::size_t>(key.level, 15)].valid()) {
            material = body.debug_materials[std::min<std::size_t>(key.level, 15)];
        }
        if (!material.valid() || !chunk.mesh) {
            continue;
        }
        if (auto const submitted = renderer.submit_model(chunk.model, chunk_matrix(planet, *chunk.mesh, view.origin),
                                                         material);
            !submitted) {
            error("[Planet] Could not submit a chunk: {}", describe(submitted.error()));
            continue;
        }
        ++drawn;
    }

    // Forget chunks unused for a while once over budget, oldest first.
    if (chunks.size() > chunk_budget / 2) {
        auto stale = std::vector<std::pair<std::uint64_t, planet::ChunkKey>>{};
        for (auto const &[key, chunk]: chunks) {
            if (chunk.last_used + keep_frames < frame_ && !chunk.building.valid()) {
                stale.emplace_back(chunk.last_used, key);
            }
        }
        std::ranges::sort(stale, {}, &std::pair<std::uint64_t, planet::ChunkKey>::first);
        auto const excess = chunks.size() - chunk_budget / 2;
        for (std::size_t i = 0; i < std::min(excess, stale.size()); ++i) {
            auto const found = chunks.find(stale[i].second);
            if (found->second.model.valid()) {
                retired_.push_back(Retired{.model = found->second.model});
            }
            chunks.erase(found);
        }
    }

    return drawn;
}

auto PlanetSystem::submit(Renderer &renderer, entt::registry const &registry, PlanetView const &view) -> void {
    ZoneScopedNC("Planets", tracy::Color::SeaGreen);
    ++frame_;
    age_retired(renderer, false);
    stats_ = Stats{};

    for (auto &body: bodies_ | std::views::values) {
        body.seen = false;
    }

    auto builds = std::uint32_t{0};
    auto uploads = std::uint32_t{0};

    auto const planets = registry.view<Components::Transform const, Components::Planet const>();
    for (auto [entity, transform, component]: planets.each()) {
        auto &body = bodies_[entity];
        body.seen = true;

        if (!same_geometry(body.planet, component) || !same_colour(body.planet, component) ||
            body.planet.ocean_water != component.ocean_water || body.planet.ocean != component.ocean ||
            body.planet.debug_lod != component.debug_lod) {
            auto const reload = !same_geometry(body.planet, component) || !same_colour(body.planet, component);
            if (reload) {
                drop_body(body);
            } else if (body.ocean_material.valid() && body.planet.ocean_water != component.ocean_water) {
                retired_.push_back(Retired{.material = body.ocean_material});
                body.ocean_material = {};
            }
            body.planet = component;
        }
        body.planet = component;

        if (!body.heightmap && !body.heightmap_loading.valid() && !body.load_failed) {
            start_loading(body);
        }
        poll_loading(renderer, body);
        ensure_materials(renderer, body);
        stats_.heightmap_ready = stats_.heightmap_ready || body.heightmap != nullptr;

        if (!component.visible) {
            continue;
        }

        auto const world = planet_matrix(systems::get_world_transform(registry, entity, transform));
        auto const camera = glm::dvec3{glm::inverse(world) * glm::dvec4{view.origin + view.camera, 1.0}};

        // The terrain's colour coordinates come from the direction to the centre, which moves with the origin.
        if (body.terrain_material.valid()) {
            auto &info = body.terrain_info.water_settings;
            info.sphere_centre = glm::vec4{glm::vec3{glm::dvec3{world[3]} - view.origin}, 1.0F};
            info.east = glm::vec4{glm::vec3{glm::dvec3{world[0]}}, 0.0F};
            info.anchor = glm::vec4{glm::vec3{glm::dvec3{world[1]}}, 0.0F};
            info.north = glm::vec4{glm::vec3{glm::dvec3{world[2]}}, 0.0F};
            if (auto const updated = renderer.update_material(body.terrain_material, body.terrain_info); !updated) {
                error("[Planet] Could not update the terrain material: {}", describe(updated.error()));
            }
        }

        // Terrain: a smooth sphere until the heightmap is in.
        auto const exaggeration = static_cast<double>(component.exaggeration);
        auto const relief_high = body.heightmap ? body.heightmap->highest() * exaggeration : 0.0;
        auto const relief_low = body.heightmap ? body.heightmap->lowest() * exaggeration : 0.0;
        auto const terrain = planet::select_chunks(camera, planet::LodSettings{
                                                                   .radius = component.radius,
                                                                   .max_level = component.max_level,
                                                                   .split_factor = 1.6,
                                                                   .highest = relief_high,
                                                                   .lowest = relief_low,
                                                           });
        stats_.terrain_drawn += draw_tree(renderer, body, body.terrain, terrain, false, world, view, builds, uploads);

        if (component.ocean && body.ocean_material.valid()) {
            update_ocean_material(renderer, body, world, camera, view);
            auto const ocean = planet::select_chunks(camera, planet::LodSettings{
                                                                     .radius = sea_radius(component),
                                                                     .max_level = ocean_max_level,
                                                                     .split_factor = 1.5,
                                                                     .highest = static_cast<double>(component.ocean_water.wave_height),
                                                                     // The sea hides what is behind its own horizon.
                                                                     .lowest = -static_cast<double>(component.ocean_water.wave_height),
                                                             });
            stats_.ocean_drawn += draw_tree(renderer, body, body.ocean, ocean, true, world, view, builds, uploads);
        }

        stats_.cached += body.terrain.size() + body.ocean.size();
        for (auto const *chunks: {&body.terrain, &body.ocean}) {
            for (auto const &chunk: *chunks | std::views::values) {
                stats_.building += chunk.building.valid() ? 1U : 0U;
            }
        }
    }

    if (frame_ % 600U == 0U && !bodies_.empty()) {
        debug("[Planet] {} terrain and {} ocean chunks drawn, {} cached, {} building", stats_.terrain_drawn,
              stats_.ocean_drawn, stats_.cached, stats_.building);
    }

    for (auto it = bodies_.begin(); it != bodies_.end();) {
        if (!it->second.seen) {
            drop_body(it->second);
            it = bodies_.erase(it);
        } else {
            ++it;
        }
    }
}

auto PlanetSystem::drop_body(Body &body) -> void {
    for (auto *chunks: {&body.terrain, &body.ocean}) {
        for (auto &chunk: *chunks | std::views::values) {
            if (chunk.model.valid()) {
                retired_.push_back(Retired{.model = chunk.model});
            }
        }
        chunks->clear();
    }
    auto retire_material = [&](MaterialHandle &material) {
        if (material.valid()) {
            retired_.push_back(Retired{.material = material});
        }
        material = {};
    };
    retire_material(body.terrain_material);
    retire_material(body.ocean_material);
    for (auto &material: body.debug_materials) {
        retire_material(material);
    }
    if (body.colour_image.valid()) {
        retired_.push_back(Retired{.image = body.colour_image});
        body.colour_image = {};
    }
    // Loads still running finish into futures nobody reads; the thread pool's futures do not block when dropped.
    body.heightmap_loading = {};
    body.colour_loading = {};
    body.heightmap.reset();
    body.load_failed = false;
}

auto PlanetSystem::age_retired(Renderer &renderer, bool everything) -> void {
    std::erase_if(retired_, [&](Retired &retired) {
        if (!everything && retired.frames_left-- > 0) {
            return false;
        }
        if (retired.model.valid()) {
            renderer.release_model(retired.model);
        }
        renderer.release_material(retired.material);
        if (retired.image.valid()) {
            static_cast<void>(renderer.image_storage().destroy_image(retired.image));
        }
        return true;
    });
}

auto PlanetSystem::clear(Renderer &renderer) -> void {
    for (auto &body: bodies_ | std::views::values) {
        drop_body(body);
    }
    bodies_.clear();
    age_retired(renderer, true);
}

auto PlanetSystem::terrain_elevation(entt::registry const &registry, glm::dvec3 const &world) const
        -> std::optional<double> {
    auto best = std::optional<double>{};
    auto best_distance = std::numeric_limits<double>::max();
    auto const planets = registry.view<Components::Transform const, Components::Planet const>();
    for (auto [entity, transform, component]: planets.each()) {
        auto const found = bodies_.find(entity);
        if (found == bodies_.end() || !found->second.heightmap) {
            continue;
        }
        auto const frame = planet_matrix(systems::get_world_transform(registry, entity, transform));
        auto const local = glm::dvec3{glm::inverse(frame) * glm::dvec4{world, 1.0}};
        auto const distance = glm::length(local) - component.radius;
        if (distance < best_distance) {
            best_distance = distance;
            best = found->second.heightmap->sample(local) * static_cast<double>(component.exaggeration);
        }
    }
    return best;
}

auto PlanetSystem::is_underwater(entt::registry const &registry, glm::dvec3 const &world) const -> bool {
    auto const planets = registry.view<Components::Transform const, Components::Planet const>();
    for (auto [entity, transform, component]: planets.each()) {
        if (!component.ocean) {
            continue;
        }
        auto const frame = planet_matrix(systems::get_world_transform(registry, entity, transform));
        auto const local = glm::dvec3{glm::inverse(frame) * glm::dvec4{world, 1.0}};
        auto const height = glm::length(local);
        if (height >= sea_radius(component)) {
            continue;
        }
        auto const found = bodies_.find(entity);
        auto const floor = found != bodies_.end() && found->second.heightmap
                                   ? component.radius + found->second.heightmap->sample(local) *
                                                                static_cast<double>(component.exaggeration)
                                   : -std::numeric_limits<double>::max();
        if (height > floor) {
            return true;
        }
    }
    return false;
}

namespace {

    // Mirrors AtmospherePush in assets/shaders/atmosphere.slang.
    struct AtmosphereParams {
        ImageRead colour;
        ImageRead depth;
        ImageWrite result;
        std::uint32_t flags = 0;
        glm::mat4 inverse_view_projection{1.0F};
        glm::vec3 planet_centre{0.0F};
        float planet_radius = 1.0F;
        glm::vec3 sun_direction{0.0F, 1.0F, 0.0F};
        float atmosphere_radius = 1.0F;
        float sun_intensity = 1.0F;
        float intensity = 1.0F;
        float time = 0.0F;
        float padding = 0.0F;
        using Layout = PushLayout<&AtmosphereParams::colour, &AtmosphereParams::depth, &AtmosphereParams::result,
                                  &AtmosphereParams::flags, &AtmosphereParams::inverse_view_projection,
                                  &AtmosphereParams::planet_centre, &AtmosphereParams::planet_radius,
                                  &AtmosphereParams::sun_direction, &AtmosphereParams::atmosphere_radius,
                                  &AtmosphereParams::sun_intensity, &AtmosphereParams::intensity,
                                  &AtmosphereParams::time, &AtmosphereParams::padding>;
    };

}

auto PlanetSystem::declare(GameGraph &graph, entt::registry const &registry, glm::mat4 const &view,
                           glm::mat4 const &projection, glm::dvec3 const &origin, glm::vec3 const &sun_direction,
                           float sun_intensity) -> void {
    if (graph.slot() != GameSlot::after_lighting || !atmosphere_shader_) {
        return;
    }

    auto const planets = registry.view<Components::Transform const, Components::Planet const>();
    for (auto [entity, transform, component]: planets.each()) {
        if (!component.visible || !component.atmosphere) {
            continue;
        }
        auto colour = graph.scene_colour_source();
        auto const depth = graph.scene_depth();
        if (!colour || !depth) {
            return;
        }

        auto const world = planet_matrix(systems::get_world_transform(registry, entity, transform));
        auto params = AtmosphereParams{
                .colour = *colour,
                .depth = ImageRead{*depth},
                .result = ImageWrite{GameImageDesc{.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "planet_atmosphere"}},
                .flags = 0,
                .inverse_view_projection = glm::inverse(projection * view),
                .planet_centre = glm::vec3{glm::dvec3{world[3]} - origin},
                .planet_radius = static_cast<float>(sea_radius(component)),
                .sun_direction = glm::normalize(sun_direction),
                .atmosphere_radius = static_cast<float>(sea_radius(component)) + component.atmosphere_height,
                .sun_intensity = sun_intensity,
                .intensity = component.atmosphere_intensity,
        };
        graph.add_compute("planet_atmosphere", params, *atmosphere_shader_, Threads::of_extent(graph.scene_extent()),
                          {.queue = frame_graph::QueueAffinity::compute_preferred,
                           .label = "Atmosphere",
                           .color = 0x4682B4});
        graph.replace_scene_colour(params.result.image);
        return; // one atmosphere per frame
    }
}
