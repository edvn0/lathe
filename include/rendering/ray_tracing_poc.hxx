#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <expected>

#include <glm/mat4x4.hpp>

#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "rendering/game_graph.hxx"

struct Renderer;

// Proof of concept for hardware ray tracing (docs/ray-tracing.md). It builds a bottom level acceleration structure of
// one hardcoded sphere and a top level one holding a single instance of it, once, at startup. Every frame a compute
// pass in GameSlot::after_lighting traces rays with VK_KHR_ray_query: a camera ray per pixel draws the sphere into the
// scene colour, and a ray from every scene pixel towards the sun darkens what the sphere shadows.
//
// Built on the GameGraph like ParticleSystem. Nothing here is a VkPipeline: the shader is an ordinary compute shader
// object, and the acceleration structure reaches it as a device address in the push constants.
class RayTracingPoc {
public:
    RayTracingPoc() = default;
    ~RayTracingPoc();

    RayTracingPoc(RayTracingPoc const &) = delete;
    auto operator=(RayTracingPoc const &) -> RayTracingPoc & = delete;

    // Builds the acceleration structures and registers the shader. Fails, harmlessly, on a device without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;

    // Declares this frame's pass. Does nothing outside GameSlot::after_lighting or before create succeeded.
    // `sun_direction` points towards the sun (Renderer::directional_light().direction).
    auto declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection, glm::vec3 sun_direction)
            -> void;

    [[nodiscard]] auto ready() const noexcept -> bool { return ready_; }

    // Debug views, in the order of the `view_*` constants in rt_poc.slang. 0 is the normal picture.
    static constexpr auto view_names = std::array<std::string_view, 7>{
            "beauty", "normals", "distance", "shadow-mask", "triangle", "barycentrics", "depth-test"};
    auto set_view(std::uint32_t view) noexcept -> void { view_ = view; }

    // Waits for nothing: the caller has idled the device. Safe to call twice.
    auto destroy() noexcept -> void;

    // Where the sphere sits and how big it is, in world units.
    static constexpr auto sphere_centre = glm::vec3{0.0F, 3.0F, 0.0F};
    static constexpr auto sphere_radius = 1.5F;

private:
    struct AccelerationStructure {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        Buffer storage;
        VkDeviceAddress address = 0;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    AccelerationStructure blas_;
    AccelerationStructure tlas_;
    Buffer vertices_;
    Buffer indices_;
    Buffer instances_;
    GameComputeShader shader_{};
    std::uint32_t view_ = 0;
    bool ready_ = false;
};
