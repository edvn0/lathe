#pragma once

#include <cstdint>
#include <vector>

#include "assets/material_storage.hxx"

class Scene;
class Renderer;
struct EngineModels;

struct ShadowCityOptions {
    std::uint32_t spot_lights = 24;
    std::uint32_t point_lights = 10;
    bool gpu_culling = true;
    bool caching = true;
    // Receives the materials the scene creates, for callers that release them afterwards.
    std::vector<MaterialHandle> *owned_materials = nullptr;
};

// Builds the dense shadow stress scene: the twenty gallery meshes, fifty copies each, on a grid under shadow casting
// spot and point lights, bloom off. Models missing from the gallery download cache are skipped. Returns the number of
// casters placed.
auto populate_shadow_city(Scene &scene, Renderer &renderer, EngineModels const &engine_models,
                          ShadowCityOptions const &options) -> std::uint32_t;
