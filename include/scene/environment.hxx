#pragma once

#include <cstdint>
#include <string>

#include <glm/vec3.hpp>

// The scene's environment: what lights it from all around and fills the background. Plain data, saved with the scene
// (SCEN section 14, see include/serialisation/scene_codec.hxx) and pushed to the renderer every frame. See
// docs/ibl-and-skybox.md.

enum class EnvironmentSource : std::uint8_t {
    // Today's look: a flat ambient term and the clear colour.
    flat_ambient = 0,
    // Preetham sky driven by the sun below.
    procedural_sky = 1,
    // An environment image: .hdr or .exr equirect, or a .ktx2 float equirect or cubemap.
    hdr_image = 2,
};

// The sun. In procedural mode it also sets the sky; in every mode it can drive Renderer::DirectionalLight.
struct SceneSun {
    // Direction toward the sun: dir = (cos e * cos a, sin e, cos e * sin a), the editor's long-standing convention.
    float azimuth_degrees = 30.0F;
    float elevation_degrees = 55.0F;

    // Preetham turbidity, 2 (clear) to 10 (hazy).
    float turbidity = 2.5F;
    glm::vec3 ground_albedo{0.3F};

    // The disc's angular radius. The real sun is about 0.27 degrees.
    float angular_radius_degrees = 0.27F;

    glm::vec3 colour{1.0F, 0.97F, 0.92F};
    float intensity = 3.0F;

    // Procedural only: tint `colour` by the atmosphere's transmittance, so the light reddens with the sky at sunset.
    bool derive_colour_from_sky = true;

    auto operator==(SceneSun const &) const -> bool = default;
};

// Exponential distance fog. These mirror Renderer's FogSettings and are saved with the scene.
struct SceneFog {
    bool enabled = false;
    glm::vec3 colour{0.5F};
    float extinction = 0.003F;
    float inscattering = 1.0F;

    // Multiplies the in-scattered colour by the environment's blurred colour along the view ray, so fog follows
    // sunsets and HDRIs. `colour` then acts as a tint.
    bool from_environment = false;

    auto operator==(SceneFog const &) const -> bool = default;
};

struct SceneEnvironment {
    EnvironmentSource source = EnvironmentSource::flat_ambient;

    // The flat fallback, and the ambient used while an environment is still building.
    float ambient_intensity = 0.15F;

    // hdr_image only: path as the user gave it.
    std::string hdr_source;

    // hdr_image only: yaw of the environment about +Y.
    float rotation_degrees = 0.0F;

    float exposure_ev = 0.0F;

    // Scale on the diffuse (SH) and specular (prefilter) terms.
    float diffuse_intensity = 1.0F;
    float specular_intensity = 1.0F;

    // Strength of the AO-derived specular occlusion; 0 disables it.
    float specular_occlusion = 1.0F;

    // Procedural only: scales the sky's radiance (and so the lighting derived from it).
    float sky_intensity = 1.0F;

    bool draw_skybox = true;
    bool fog_sky = false;
    bool sun_drives_directional_light = true;

    // Energy compensation for IBL specular, from the BRDF LUT.
    bool multi_scatter = true;

    // hdr_image only, for equirect sources: the radiance cube's face size, 512 or 1024. A cubemap source uses its own.
    std::uint32_t hdr_cube_size = 512;

    SceneSun sun;
    SceneFog fog;

    auto operator==(SceneEnvironment const &) const -> bool = default;
};

// What a new scene starts with in the editor: the procedural sky at the default sun. A scene decoded from a file starts
// from SceneEnvironment{} instead (flat ambient), so a file without an environment section looks as it always did.
[[nodiscard]]
inline auto new_scene_environment() -> SceneEnvironment {
    SceneEnvironment environment;
    environment.source = EnvironmentSource::procedural_sky;

    return environment;
}
