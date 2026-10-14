#pragma once

#include <cstddef>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "game/game.hxx"

// Walls of sample spheres spanning PBR material parameters, like the glTF material test models: roughness x metallic,
// tinted dielectrics x roughness, and emissive colours x strength (feeds bloom).
// Keys: 1-6 shot (5 = decals, 6 = shadow-casting lights), B bloom, C local shadow GPU culling,
// V local shadow caching, O stop/start the orbiting cube.
class MaterialWallGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;
    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    Renderer *renderer_ = nullptr;

    // A cube circling inside one spot light's range: that light's shadow redraws every frame, the others stay cached.
    entt::entity orbiter_ = entt::null;
    glm::vec3 orbit_centre_{-2.0F, 0.5F, 9.0F};
    std::uint32_t frames_ = 0;
    bool orbiting_ = true;

    std::size_t shot_ = 0;
    float time_ = 0.0F;
    float orbit_time_ = 0.0F;
    glm::vec3 eye_{0.0F, 6.0F, 22.0F};
    glm::vec3 look_at_{0.0F, 6.0F, 0.0F};
    float fov_degrees_ = 50.0F;
};
