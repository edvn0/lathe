#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "app/benchmark.hxx"
#include "assets/material.hxx"
#include "scene/camera_path.hxx"

class Scene;
struct Renderer;
struct EngineModels;

enum class BenchmarkSceneSource : std::uint8_t {
    game,
    synthetic,
};

enum class BenchmarkLoadTarget : std::uint8_t {
    scene,
    render_scale,
};

struct BenchmarkScenarioContext {
    Scene &scene;
    Renderer &renderer;
    EngineModels const &engine_models;
    std::uint32_t load = 0;

    std::vector<MaterialHandle> &owned_materials;
};

struct BenchmarkScenario {
    BenchmarkScenarioInfo info;
    std::string description;
    BenchmarkSceneSource source = BenchmarkSceneSource::synthetic;
    BenchmarkLoadTarget load_target = BenchmarkLoadTarget::scene;

    std::function<void(BenchmarkScenarioContext const &)> populate;

    std::function<std::vector<CameraKeyframe>(std::uint32_t load)> camera_path;
};

[[nodiscard]] auto builtin_benchmark_scenarios(bool game_has_benchmark_path) -> std::vector<BenchmarkScenario>;

[[nodiscard]] auto
scenario_infos(std::vector<BenchmarkScenario> const &scenarios) -> std::vector<BenchmarkScenarioInfo>;
