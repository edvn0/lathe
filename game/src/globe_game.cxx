#include "globe_game.hxx"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <numbers>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"
#include "terrain/planet.hxx"

namespace {

    constexpr double earth_radius = 6'371'000.0;

    // NASA Earth Observatory, Blue Marble: Next Generation (topography and bathymetry from GEBCO; public domain NASA
    // imagery, credit "NASA Earth Observatory"). See docs/globe.md and THIRD_PARTY_NOTICES.md.
    struct Download {
        std::string_view uri;
        std::string_view file;
        std::string_view sha256;
    };
    constexpr Download elevation_source{
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png",
            "gebco_08_rev_elev_21600x10800.png", "4a70590d40dd7b9c69b9ab359d1dc4475ce97c1d2d13625223b248364112c699"};
    constexpr Download bathymetry_source{
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73963/gebco_08_rev_bath_21600x10800.png",
            "gebco_08_rev_bath_21600x10800.png", "b3a67076fccfbee73ebe437fd66a8ed4969fbc5ce949033b50675d43f379e403"};
    constexpr Download colour_source{
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg",
            "world.topo.bathy.200407.3x5400x2700.jpg", "4f4240673a3a1b173d61b92ca4b07bac5fd17059ea5f725ba6da5a9c5386b7ba"};

    constexpr std::array<float, 3> exaggerations{1.0F, 5.0F, 20.0F};

    auto download_options() -> HttpClientOptions {
        HttpClientOptions options;
        options.worker_count = 3;
        options.timeout = std::chrono::seconds{600};
        options.max_response_bytes = std::size_t{128} * 1024U * 1024U;
        return options;
    }

    auto on_surface(double latitude_degrees, double longitude_degrees, double altitude) -> glm::dvec3 {
        return planet::direction_of(planet::LatLon{.latitude = glm::radians(latitude_degrees),
                                                   .longitude = glm::radians(longitude_degrees)}) *
               (earth_radius + altitude);
    }

    // Where the sun stands over each shot (latitude, longitude of the subsolar point, degrees).
    constexpr std::array<std::array<double, 2>, 4> sun_positions{{
            {12.0, 45.0},  // the whole Earth: day over Africa and Europe, the terminator over Asia
            {20.0, 40.0},  // Europe in the morning-to-midday light
            {32.0, 75.0},  // Gibraltar in the morning, the sun low ahead for a glint on the strait
            {10.0, 40.0},  // the Himalaya in long afternoon light from the south-west
    }};

    auto set_sun(Renderer &renderer, std::uint32_t shot) {
        auto const &sun = sun_positions[shot % sun_positions.size()];
        auto const direction = glm::vec3{on_surface(sun[0], sun[1], 0.0) / earth_radius};
        renderer.set_directional_light(Renderer::DirectionalLight{
                .direction = glm::normalize(direction),
                .colour = {1.0F, 0.97F, 0.92F},
                .intensity = 3.0F,
        });
    }

    auto planet_entity(Scene &scene) -> entt::entity {
        auto const view = scene.get_registry().view<Components::Planet>();
        return view.empty() ? entt::entity{entt::null} : view.front();
    }

}

GlobeGame::GlobeGame() : http_(download_options()) {}

auto GlobeGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &) -> void {
    renderer_ = &renderer;
    scene.get_registry().clear();

    scene.environment.source = EnvironmentSource::flat_ambient;
    scene.environment.draw_skybox = false;
    scene.environment.ambient_intensity = 0.015F;
    scene.environment.sun_drives_directional_light = false;
    set_sun(renderer, shot_);

    auto const directory = cache_path("downloads/globe").absolute();
    std::error_code directory_error;
    std::filesystem::create_directories(directory, directory_error);

    auto const fetch = [&](Download const &source, std::filesystem::path &path) {
        path = directory / source.file;
        return http_.get_file_async({.uri = std::string{source.uri},
                                     .destination = path,
                                     .sha256 = std::string{source.sha256}});
    };
    if (!heightmap_ready_) {
        elevation_download_ = fetch(elevation_source, elevation_path_);
        bathymetry_download_ = fetch(bathymetry_source, bathymetry_path_);
        info("[Globe] Fetching the NASA Blue Marble elevation and bathymetry (about 53 MB, cached after the first run)");
    }
    if (!colour_ready_) {
        colour_download_ = fetch(colour_source, colour_path_);
    }

    auto const earth = Entity{&scene, "earth"};
    earth.emplace<Components::Transform>(Components::Transform{});
    auto planet = Components::Planet{
            .radius = earth_radius,
            .colour = Components::PlanetColour::elevation,
            .exaggeration = exaggerations[exaggeration_index_],
            .debug_lod = debug_lod_,
    };
    if (heightmap_ready_) {
        planet.elevation_path = elevation_path_.string();
        planet.bathymetry_path = bathymetry_path_.string();
    }
    if (colour_ready_ && image_colour_) {
        planet.colour = Components::PlanetColour::image;
        planet.colour_path = colour_path_.string();
    }
    earth.emplace<Components::Planet>(std::move(planet));

    info("[Globe] Keys: 1-4 shots, P pause, X exaggeration, L LOD colours, C colour source, F free flight");
}

auto GlobeGame::on_update(Scene &scene, float delta_time) -> void {
    auto const finished = [](auto &download, std::string_view what) {
        if (!download.valid() || download.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
            return false;
        }
        if (auto const result = download.get(); !result) {
            warn("[Globe] Could not fetch the {}: {}", what, describe(result.error()));
            return false;
        }
        return true;
    };

    // The heightmap needs both images.
    elevation_done_ = elevation_done_ || finished(elevation_download_, "elevation map");
    bathymetry_done_ = bathymetry_done_ || finished(bathymetry_download_, "bathymetry map");
    heightmap_ready_ = heightmap_ready_ || (elevation_done_ && bathymetry_done_);
    colour_ready_ = colour_ready_ || finished(colour_download_, "Blue Marble colour");

    if (auto const entity = planet_entity(scene); entity != entt::null) {
        auto &registry = scene.get_registry();
        auto const &current = registry.get<Components::Planet>(entity);
        auto wanted = current;
        if (heightmap_ready_) {
            wanted.elevation_path = elevation_path_.string();
            wanted.bathymetry_path = bathymetry_path_.string();
        }
        wanted.colour = colour_ready_ && image_colour_ ? Components::PlanetColour::image
                                                       : Components::PlanetColour::elevation;
        wanted.colour_path = colour_ready_ && image_colour_ ? colour_path_.string() : std::string{};
        wanted.exaggeration = exaggerations[exaggeration_index_];
        wanted.debug_lod = debug_lod_;
        if (!(wanted == current)) {
            registry.replace<Components::Planet>(entity, std::move(wanted));
        }
    }

    if (orbiting_ && !free_flight_) {
        orbit_time_ += static_cast<double>(delta_time);
    }

    if (free_flight_) {
        auto const up = glm::normalize(free_position_);
        auto const altitude = std::max(glm::length(free_position_) - earth_radius, 10.0);
        auto const speed = std::max(altitude * 0.6, 50.0);
        auto const turn = 0.8 * static_cast<double>(delta_time);

        if (held_.contains(GLFW_KEY_LEFT)) {
            free_yaw_ -= turn;
        }
        if (held_.contains(GLFW_KEY_RIGHT)) {
            free_yaw_ += turn;
        }
        if (held_.contains(GLFW_KEY_UP)) {
            free_pitch_ = std::min(free_pitch_ + turn, 1.5);
        }
        if (held_.contains(GLFW_KEY_DOWN)) {
            free_pitch_ = std::max(free_pitch_ - turn, -1.5);
        }

        auto const east = glm::normalize(glm::cross(glm::dvec3{0.0, 1.0, 0.0}, up) + glm::dvec3{1e-9, 0.0, 0.0});
        auto const north = glm::normalize(glm::cross(up, east));
        auto const heading = std::cos(free_yaw_) * north + std::sin(free_yaw_) * east;
        auto const forward = std::cos(free_pitch_) * heading + std::sin(free_pitch_) * up;
        auto const side = glm::normalize(glm::cross(up, heading));

        auto move = glm::dvec3{0.0};
        move += held_.contains(GLFW_KEY_W) ? forward : glm::dvec3{0.0};
        move -= held_.contains(GLFW_KEY_S) ? forward : glm::dvec3{0.0};
        move += held_.contains(GLFW_KEY_D) ? side : glm::dvec3{0.0};
        move -= held_.contains(GLFW_KEY_A) ? side : glm::dvec3{0.0};
        move += held_.contains(GLFW_KEY_E) ? up : glm::dvec3{0.0};
        move -= held_.contains(GLFW_KEY_Q) ? up : glm::dvec3{0.0};
        free_position_ += move * speed * static_cast<double>(delta_time);

        // Never below the ground (sea level plus a little).
        if (glm::length(free_position_) < earth_radius + 10.0) {
            free_position_ = glm::normalize(free_position_) * (earth_radius + 10.0);
        }
    }
}

auto GlobeGame::start_free_flight() -> void {
    auto const eye = scripted_eye();
    free_position_ = eye.position;
    auto const up = glm::normalize(eye.position);
    auto const east = glm::normalize(glm::cross(glm::dvec3{0.0, 1.0, 0.0}, up) + glm::dvec3{1e-9, 0.0, 0.0});
    auto const north = glm::normalize(glm::cross(up, east));
    auto const look = glm::normalize(eye.target - eye.position);
    free_pitch_ = std::asin(std::clamp(glm::dot(look, up), -1.0, 1.0));
    free_yaw_ = std::atan2(glm::dot(look, east), glm::dot(look, north));
}

auto GlobeGame::subscribe_input(InputContext const &ctx) -> void {
    listen(ctx, &GlobeGame::on_key_pressed);
    listen(ctx, &GlobeGame::on_key_released);
}

auto GlobeGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    held_.insert(event.key);
    switch (event.key) {
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
            shot_ = static_cast<std::uint32_t>(event.key - GLFW_KEY_1);
            free_flight_ = false;
            if (renderer_ != nullptr) {
                set_sun(*renderer_, shot_);
            }
            break;
        case GLFW_KEY_P:
            orbiting_ = !orbiting_;
            break;
        case GLFW_KEY_X:
            exaggeration_index_ = (exaggeration_index_ + 1U) % static_cast<std::uint32_t>(exaggerations.size());
            info("[Globe] Relief exaggeration {}x", exaggerations[exaggeration_index_]);
            break;
        case GLFW_KEY_L:
            debug_lod_ = !debug_lod_;
            break;
        case GLFW_KEY_C:
            image_colour_ = !image_colour_;
            break;
        case GLFW_KEY_F:
            free_flight_ = !free_flight_;
            if (free_flight_) {
                start_free_flight();
            }
            break;
        default:
            break;
    }
}

auto GlobeGame::on_key_released(Scene &, KeyReleasedEvent const &event) -> void { held_.erase(event.key); }

auto GlobeGame::scripted_eye() const -> Eye {
    auto const t = orbit_time_;
    switch (shot_) {
        case 1:
            return Eye{.position = on_surface(31.0, 10.0 + t * 0.15, 700'000.0), .target = on_surface(46.0, 12.0, 0.0)};
        case 2:
            return Eye{.position = on_surface(35.88, -5.95 + t * 0.0004, 2'000.0),
                       .target = on_surface(35.95, -5.45, 0.0)};
        case 3:
            return Eye{.position = on_surface(24.6, 85.6 + t * 0.01, 100'000.0), .target = on_surface(28.3, 86.6, 0.0)};
        default:
            return Eye{.position = on_surface(15.0, 25.0 + t * 2.0, 20'000'000.0), .target = glm::dvec3{0.0}};
    }
}

auto GlobeGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    auto eye = scripted_eye();
    if (free_flight_) {
        auto const up = glm::normalize(free_position_);
        auto const east = glm::normalize(glm::cross(glm::dvec3{0.0, 1.0, 0.0}, up) + glm::dvec3{1e-9, 0.0, 0.0});
        auto const north = glm::normalize(glm::cross(up, east));
        auto const heading = std::cos(free_yaw_) * north + std::sin(free_yaw_) * east;
        eye = Eye{.position = free_position_,
                  .target = free_position_ + std::cos(free_pitch_) * heading + std::sin(free_pitch_) * up};
    }

    // Render space is centred on the eye (the floating origin), so the view only turns.
    auto const forward = glm::vec3{glm::normalize(eye.target - eye.position)};
    auto const up = glm::vec3{glm::normalize(eye.position)};
    // Up is away from the planet, or north when looking straight down (then "up" says nothing about the roll).
    auto const north = std::abs(forward.y) > 0.999F ? glm::vec3{0.0F, 0.0F, 1.0F} : glm::vec3{0.0F, 1.0F, 0.0F};
    auto const view = glm::lookAtLH(glm::vec3{0.0F}, forward, std::abs(glm::dot(forward, up)) > 0.99F ? north : up);

    // Near and far follow the altitude: the depth buffer only has to cover what can be seen, out past the horizon.
    auto const distance = glm::length(eye.position);
    auto const altitude = std::max(distance - earth_radius, 1.0);
    auto const deepest = earth_radius - 11'000.0;
    auto const horizon = std::sqrt(std::max(distance * distance - deepest * deepest, 0.0));
    auto const beyond = std::sqrt((earth_radius + 120'000.0) * (earth_radius + 120'000.0) - deepest * deepest);
    // Nothing is nearer than the ground below, so the near plane can sit at a good fraction of the altitude, which keeps
    // the depth precision the water needs (its depth-based colour) even from 20,000 km.
    auto const near_clip = static_cast<float>(std::clamp(altitude * 0.3, 0.5, 5'000'000.0));
    auto const far_clip = static_cast<float>(horizon + beyond);
    auto const fov = glm::radians(shot_ == 0 && !free_flight_ ? 35.0F : 55.0F);

    return CameraParams{
            .view = view,
            .projection = glm::perspectiveLH_ZO(fov, aspect_ratio, near_clip, far_clip),
            .near_clip = near_clip,
            .far_clip = far_clip,
            .vertical_fov_radians = fov,
            .origin = eye.position,
    };
}
