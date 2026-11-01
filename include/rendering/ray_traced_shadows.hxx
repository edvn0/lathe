#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "core/renderer_error.hxx"
#include "rendering/pipeline_graph_repository.hxx"

class Renderer;

// Ray traced sun shadows (docs/ray-traced-shadows.md): where the device supports ray query, the sun's shadow is a ray from
// every pixel's surface point toward the sun against the scene acceleration structure, instead of the cascaded shadow map.
// The renderer owns it and declares its pass (renderer_frame_graph.cxx); the forward pass reads the mask it writes.
class RayTracedShadows {
public:
    struct Settings {
        bool enabled = true;
        // Half the angle of the cone of the sun's rays, in radians. The sun is about 0.0047 across; the default is wider
        // than that, so the penumbra shows (0 gives hard shadows).
        float sun_spread = 0.006F;
    };

    RayTracedShadows() = default;

    RayTracedShadows(RayTracedShadows const &) = delete;
    auto operator=(RayTracedShadows const &) -> RayTracedShadows & = delete;

    // Registers the shader and turns on the renderer's ray tracing scene. Fails, harmlessly, without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;
    auto destroy() noexcept -> void { renderer_ = nullptr; }

    [[nodiscard]] auto ready() const noexcept -> bool { return renderer_ != nullptr; }
    [[nodiscard]] auto settings() noexcept -> Settings & { return settings_; }
    [[nodiscard]] auto settings() const noexcept -> Settings const & { return settings_; }

    struct Inputs {
        glm::mat4 inverse_view_projection{1.0F};
        VkDeviceAddress instance_table = 0;
        VkDeviceAddress materials = 0;
        // Bindless indices: the resolved depth (sampled_2d_depth) and the mask the pass writes (storage_2d).
        std::uint32_t depth_texture = 0;
        std::uint32_t mask_storage = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        // Towards the sun.
        glm::vec3 sun{0.0F, 1.0F, 0.0F};
    };

    auto record(VkCommandBuffer command_buffer, std::uint32_t frame_index, Inputs const &inputs) -> void;

private:
    Renderer *renderer_ = nullptr;
    Settings settings_{};
    PipelineNodeHandle shader_{};
    std::uint32_t frame_ = 0;
};
