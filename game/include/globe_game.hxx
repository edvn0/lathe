#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <future>
#include <set>

#include <glm/vec3.hpp>

#include "game/game.hxx"
#include "net/http.hxx"

// The Earth at real scale (docs/globe.md): a Components::Planet with NASA's Blue Marble elevation, bathymetry and
// colour (downloaded on first run into the cache, never committed), an ocean and an atmosphere, lit by the sun against
// a starry sky. The camera works in double precision and hands the engine a floating origin (CameraParams::origin).
// Keys: 1 the whole Earth from 20,000 km, 2 Europe from low orbit, 3 the Strait of Gibraltar from 3 km, 4 the
// Himalaya from 100 km, P pause the orbit, X cycle the relief exaggeration (1x, 5x, 20x), L colour terrain chunks by LOD level,
// C switch colour (Blue Marble / from elevation), F free flight (W/S forward and back, A/D sideways, Q/E down and up,
// arrows turn; speed scales with altitude).
class GlobeGame final : public IGame {
public:
    GlobeGame();

    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto subscribe_input(InputContext const &ctx) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Eye {
        glm::dvec3 position{0.0};
        glm::dvec3 target{0.0};
    };

    [[nodiscard]] auto scripted_eye() const -> Eye;
    auto start_free_flight() -> void;

    HttpClient http_;
    Renderer *renderer_ = nullptr;

    std::future<std::expected<void, HttpError>> elevation_download_;
    std::future<std::expected<void, HttpError>> bathymetry_download_;
    std::future<std::expected<void, HttpError>> colour_download_;
    std::filesystem::path elevation_path_;
    std::filesystem::path bathymetry_path_;
    std::filesystem::path colour_path_;
    bool elevation_done_ = false;
    bool bathymetry_done_ = false;
    bool heightmap_ready_ = false;
    bool colour_ready_ = false;

    std::uint32_t shot_ = 0;
    bool orbiting_ = true;
    double orbit_time_ = 0.0;
    std::uint32_t exaggeration_index_ = 0;
    bool debug_lod_ = false;
    bool image_colour_ = true;

    bool free_flight_ = false;
    glm::dvec3 free_position_{0.0};
    double free_yaw_ = 0.0;
    double free_pitch_ = 0.0;
    std::set<int> held_;
};
