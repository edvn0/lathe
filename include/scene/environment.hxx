#pragma once

#include <cstdint>
#include <string>

#include <glm/vec3.hpp>

enum class EnvironmentSource : std::uint8_t {
    flat_ambient = 0,
    procedural_sky = 1,
    hdr_image = 2,
};

struct SceneSun {
    float azimuth_degrees = 30.0F;
    float elevation_degrees = 55.0F;

    float turbidity = 2.5F;
    glm::vec3 ground_albedo{0.3F};

    float angular_radius_degrees = 0.27F;

    glm::vec3 colour{1.0F, 0.97F, 0.92F};
    float intensity = 3.0F;

    bool derive_colour_from_sky = true;

    auto operator==(SceneSun const &) const -> bool = default;
};

struct SceneFog {
    bool enabled = false;
    glm::vec3 colour{0.5F};
    float extinction = 0.003F;
    float inscattering = 1.0F;

    bool from_environment = false;

    auto operator==(SceneFog const &) const -> bool = default;
};

struct SceneEnvironment {
    EnvironmentSource source = EnvironmentSource::flat_ambient;

    float ambient_intensity = 0.15F;

    std::string hdr_source;

    float rotation_degrees = 0.0F;

    float exposure_ev = 0.0F;

    float diffuse_intensity = 1.0F;
    float specular_intensity = 1.0F;

    float specular_occlusion = 1.0F;

    float sky_intensity = 1.0F;

    bool draw_skybox = true;
    bool fog_sky = false;
    bool sun_drives_directional_light = true;

    bool multi_scatter = true;

    std::uint32_t hdr_cube_size = 512;

    SceneSun sun;
    SceneFog fog;

    auto operator==(SceneEnvironment const &) const -> bool = default;
};

[[nodiscard]]
inline auto new_scene_environment() -> SceneEnvironment {
    SceneEnvironment environment;
    environment.source = EnvironmentSource::procedural_sky;

    return environment;
}
