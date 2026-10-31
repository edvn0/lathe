#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>

#include <glm/mat4x4.hpp>

#include "core/renderer_error.hxx"
#include "rendering/game_graph.hxx"

struct Renderer;

// Ray query checks of the scene acceleration structure (docs/ray-tracing.md, docs/probe-gi.md phase 1). It turns the
// renderer's ray tracing scene on (a BLAS per submesh, a TLAS of the frame's instances: SceneAccelerationStructure) and
// adds one compute pass in GameSlot::after_lighting that traces a camera ray per pixel and a sun ray from each hit with
// VK_KHR_ray_query. Nothing here is a VkPipeline: the shader is an ordinary compute shader object.
class RayTracingPoc {
public:
    RayTracingPoc() = default;

    // Registers the shader and enables the renderer's ray tracing scene. Fails, harmlessly, without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;

    // Declares this frame's pass. Does nothing outside GameSlot::after_lighting or before create succeeded.
    // `sun_direction` points towards the sun (Renderer::directional_light().direction).
    auto declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection, glm::vec3 sun_direction)
            -> void;

    [[nodiscard]] auto ready() const noexcept -> bool { return ready_; }

    // Debug views, in the order of the `view_*` constants in rt_poc.slang. 0 leaves the picture alone.
    static constexpr auto view_names = std::array<std::string_view, 7>{
            "beauty", "shadow-mask", "split", "depth-match", "normals", "instance", "overlay"};
    auto set_view(std::uint32_t view) noexcept -> void { view_ = view; }

    // Nothing to free: the scene acceleration structure belongs to the renderer.
    auto destroy() noexcept -> void { ready_ = false; }

private:
    Renderer *renderer_ = nullptr;
    GameComputeShader shader_{};
    std::uint32_t view_ = 0;
    bool ready_ = false;
};
