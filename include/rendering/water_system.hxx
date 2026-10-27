#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "assets/material.hxx"
#include "assets/material_storage.hxx"
#include "assets/model.hxx"
#include "scene/components.hxx"

struct Renderer;

// Draws every entity's Components::Water (docs/water.md). Per body it makes and owns a grid mesh and a water material
// (MaterialCreateInfo::water, an internal detail of this system), keeps them in step with the component, and submits
// the surface through the ordinary blended path, where the vertex stage displaces it and the fragment stage shades it
// (assets/shaders/water.slang). A body whose component goes away, or whose entity is destroyed (Ctrl-R, a scene
// clear), has its mesh and material released a few frames later, once no frame in flight still draws them.
//
// The static queries read only the components, so gameplay (buoyancy, boats, Lua's scene.water_height) can ask where
// the surface is: they run the CPU copy of the shader's wave field (rendering/water_waves.hxx).
class WaterSystem {
public:
    // The ocean grid: `ocean_cells` cells across `ocean_size` metres, packed towards the middle (the camera) and
    // widening exponentially by `ocean_curve`: about 5 cm apart under the camera, 1.4 m at 30 m, 26 m at the edge.
    static constexpr float ocean_size = 1200.0F;
    static constexpr std::uint32_t ocean_cells = 320;
    static constexpr float ocean_curve = 7.0F;
    // The ocean grid follows the camera in steps of this many metres, so it does not slide (and its vertices do not
    // resample the waves) every frame.
    static constexpr float ocean_snap = 8.0F;
    // A lake's grid spacing (m), and the cell count it is held within along each side.
    static constexpr float lake_spacing = 0.125F;
    static constexpr std::uint32_t lake_min_cells = 8;
    static constexpr std::uint32_t lake_max_cells = 256;
    // Submits a released mesh or material waits before it is freed (more than the frames in flight).
    static constexpr std::uint32_t retire_frames = 4;

    WaterSystem() = default;
    WaterSystem(WaterSystem const &) = delete;
    auto operator=(WaterSystem const &) -> WaterSystem & = delete;

    // Brings the bodies in `registry` up to date and submits them. `camera_position` places the ocean's grid. Call
    // once per frame with the other scene submissions, on the render thread.
    auto submit(Renderer &renderer, entt::registry const &registry, glm::vec3 camera_position) -> void;

    // Releases every mesh and material now (the renderer must be idle).
    auto clear(Renderer &renderer) -> void;

    [[nodiscard]] auto body_count() const noexcept -> std::size_t { return bodies_.size(); }

    // The material a body draws with.
    [[nodiscard]] static auto material_info(Renderer &renderer, Components::Water const &water, glm::mat4 const &world)
            -> MaterialCreateInfo;

    // The cells along x and z of a lake's grid.
    [[nodiscard]] static auto lake_cells(glm::vec2 extent) noexcept -> glm::uvec2;

    // Where the ocean grid is centred for a camera at `camera_xz`.
    [[nodiscard]] static auto ocean_centre(glm::vec2 camera_xz) noexcept -> glm::vec2;

    // The surface height of one body at world `xz` and shader time `time` (Renderer::frame_time), or nothing when
    // `xz` is outside it (a lake's rectangle, an ocean's zone). `world` is the entity's world transform. The waves
    // are the ones the mesh shows close to the camera (the shortest are faded as the vertex stage fades them there).
    [[nodiscard]] static auto surface_height(Components::Water const &water, glm::mat4 const &world, glm::vec2 xz,
                                             float time) -> std::optional<float>;
    // The displaced surface's normal there (the slope of the geometry, without the shading's extra detail).
    [[nodiscard]] static auto surface_normal(Components::Water const &water, glm::mat4 const &world, glm::vec2 xz,
                                             float time) -> std::optional<glm::vec3>;

    // The water surface at `xz` over every visible body in `registry`: a lake wins over an ocean where both cover the
    // point, and among lakes the highest. Nothing when no water covers it.
    [[nodiscard]] static auto water_height(entt::registry const &registry, glm::vec2 xz, float time)
            -> std::optional<float>;
    [[nodiscard]] static auto water_normal(entt::registry const &registry, glm::vec2 xz, float time)
            -> std::optional<glm::vec3>;
    // Whether `point` is below the water surface over it.
    [[nodiscard]] static auto is_underwater(entt::registry const &registry, glm::vec3 point, float time) -> bool;

private:
    struct Body {
        Components::Water water;
        MaterialCreateInfo material_info;
        MaterialHandle material{};
        ModelHandle model{};
        glm::uvec2 cells{0U};
        bool seen = false;
    };

    struct Retired {
        MaterialHandle material{};
        ModelHandle model{};
        std::uint32_t frames_left = retire_frames;
    };

    auto retire(Body &body) -> void;
    auto age_retired(Renderer &renderer, bool everything) -> void;
    auto ocean_model(Renderer &renderer) -> ModelHandle;

    std::unordered_map<entt::entity, Body> bodies_;
    std::vector<Retired> retired_;
    ModelHandle ocean_model_{};
};
