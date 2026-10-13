#pragma once

#include <cstddef>

#include <glm/vec3.hpp>

#include "app/game.hxx"

// Walls of sample spheres spanning PBR material parameters, like the glTF material test models: roughness x metallic,
// tinted dielectrics x roughness, and emissive colours x strength (feeds bloom).
// Keys: 1-4 shot, B bloom on/off.
class MaterialWallGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    Renderer *renderer_ = nullptr;

    std::size_t shot_ = 0;
    float time_ = 0.0F;
    glm::vec3 eye_{0.0F, 6.0F, 22.0F};
    glm::vec3 look_at_{0.0F, 6.0F, 0.0F};
    float fov_degrees_ = 50.0F;
};
