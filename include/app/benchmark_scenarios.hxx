#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "app/benchmark.hxx"
#include "scene/camera_path.hxx"

class Scene;
struct Renderer;
struct EngineModels;

// The scenes --benchmark-suite runs. Each stresses one axis of the renderer and has a load parameter it is swept
// over, so the suite reports how cost grows with load rather than one number for one scene
// (docs/perf-benchmark.md, "Scenarios").

enum class BenchmarkSceneSource : std::uint8_t {
    // The game's own scene (IGame::on_populate) and camera path, with its streaming terrain.
    game,
    // Built here from engine primitives; the terrain and the game's per-frame hooks are switched off.
    synthetic,
};

enum class BenchmarkLoadTarget : std::uint8_t {
    // The load is passed to populate() and camera_path().
    scene,
    // The load is the render resolution in percent of --benchmark-render-size; the scene is the game's.
    render_scale,
};

struct BenchmarkScenarioContext {
    Scene &scene;
    Renderer &renderer;
    EngineModels const &engine_models;
    std::uint32_t load = 0;
};

struct BenchmarkScenario {
    BenchmarkScenarioInfo info;
    std::string description;
    BenchmarkSceneSource source = BenchmarkSceneSource::synthetic;
    BenchmarkLoadTarget load_target = BenchmarkLoadTarget::scene;

    // Synthetic scenes: fills the (already cleared) scene. Deterministic given the random seed and load.
    std::function<void(BenchmarkScenarioContext const &)> populate;

    // Synthetic scenes: the camera loop for a load.
    std::function<std::vector<CameraKeyframe>(std::uint32_t load)> camera_path;
};

// `game_has_benchmark_path`: whether the game defines IGame::benchmark_camera_path(); without one the game scenarios
// are left out.
[[nodiscard]] auto builtin_benchmark_scenarios(bool game_has_benchmark_path) -> std::vector<BenchmarkScenario>;

[[nodiscard]] auto
scenario_infos(std::vector<BenchmarkScenario> const &scenarios) -> std::vector<BenchmarkScenarioInfo>;
