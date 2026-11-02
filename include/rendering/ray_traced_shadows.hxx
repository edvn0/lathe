#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "core/renderer_error.hxx"
#include "gpu/image_storage.hxx"
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
        // Also trace the shadow of the lights that have a shadow tile (the first three per pixel); the rest keep the atlas.
        bool local_lights = true;
        // The mask is traced at the screen size >> this: 0 full, 1 half, 2 quarter resolution (at most max_resolution_shift).
        // Each mask pixel traces from one pixel of the block it covers, a different one every frame.
        std::uint32_t resolution_shift = 0;
        // Filter the traced mask over time: the last frames' masks are reprojected and blended with this frame's, which
        // is noisy (a few jittered rays per pixel), and a pixel with no usable history gets a depth aware blur instead.
        // Off shows the raw mask, which flickers.
        bool temporal_filter = true;
        // How much of this frame's raw mask each frame brings in (0.1 converges in about ten frames).
        float temporal_blend = 0.1F;
    };

    static constexpr std::uint32_t max_resolution_shift = 2;

    // The extent of the mask for a screen of this size and a resolution shift.
    [[nodiscard]] static constexpr auto mask_extent(std::uint32_t size, std::uint32_t shift) noexcept -> std::uint32_t {
        auto const block = std::uint32_t{1} << shift;
        return (size + block - 1U) / block;
    }

    RayTracedShadows() = default;

    RayTracedShadows(RayTracedShadows const &) = delete;
    auto operator=(RayTracedShadows const &) -> RayTracedShadows & = delete;

    // Registers the shader and turns on the renderer's ray tracing scene. Fails, harmlessly, without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;
    auto destroy() noexcept -> void {
        renderer_ = nullptr;
        mask_history_ = {};
        depth_history_ = {};
        history_extent_ = {};
        history_valid_ = false;
        history_initialised_ = false;
        history_slot_ = 0;
    }

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
        // The screen (depth buffer) size; the mask is smaller by Settings::resolution_shift.
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        // Towards the sun.
        glm::vec3 sun{0.0F, 1.0F, 0.0F};
        bool trace_sun = true;
        // Local lights: the UBO, the light array and the cluster light lists the forward pass reads (cluster_lights is 0
        // without clustered lighting). Ignored when trace_local is false.
        bool trace_local = false;
        VkDeviceAddress ubo = 0;
        VkDeviceAddress lights = 0;
        VkDeviceAddress cluster_lights = 0;
        std::uint32_t light_count = 0;
    };

    auto record(VkCommandBuffer command_buffer, std::uint32_t frame_index, Inputs const &inputs) -> void;

    // The temporal filter's history: the resolved mask and the view depth of each pixel, two of each, one read while the
    // other is written, swapped every frame. They are created (and the history dropped) when the mask's extent changes.
    // `prepare_history` returns false when the filter is off or its images could not be made.
    [[nodiscard]] auto prepare_history(VkExtent2D screen, VkExtent2D mask) -> bool;
    [[nodiscard]] auto history_mask(std::uint32_t slot) const -> Image const & { return *mask_history_[slot]; }
    [[nodiscard]] auto history_depth(std::uint32_t slot) const -> Image const & { return *depth_history_[slot]; }
    // The slot the resolve pass reads (the other is written, and read by the forward pass after it).
    [[nodiscard]] auto history_read_slot() const noexcept -> std::uint32_t { return history_slot_; }
    [[nodiscard]] auto history_write_slot() const noexcept -> std::uint32_t { return history_slot_ ^ 1U; }
    [[nodiscard]] auto history_initialised() const noexcept -> bool { return history_initialised_; }
    [[nodiscard]] auto history_mask_index(std::uint32_t slot) const -> std::uint32_t { return mask_history_[slot].handle().index; }
    // A frame that skips the pass leaves the history behind: the next one starts over.
    auto invalidate_history() noexcept -> void { history_valid_ = false; }
    // Called once the frame has declared the resolve pass: flips the slots and remembers the camera.
    auto finish_history(glm::mat4 const &view_projection) -> void;

    struct ResolveInputs {
        glm::mat4 inverse_view_projection{1.0F};
        glm::mat4 view_projection{1.0F};
        // Bindless indices: the resolved depth and the raw mask (sampled_2d_depth, sampled_2d), the history read
        // (sampled_2d) and written (storage_2d).
        std::uint32_t depth_texture = 0;
        std::uint32_t raw_mask = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };
    // Records the temporal resolve for the slot prepare_history set up; call after `record` of the same frame.
    auto record_resolve(VkCommandBuffer command_buffer, std::uint32_t frame_index, ResolveInputs const &inputs) -> void;

private:
    Renderer *renderer_ = nullptr;
    Settings settings_{};
    PipelineNodeHandle shader_{};
    PipelineNodeHandle resolve_shader_{};
    std::uint32_t frame_ = 0;
    // The frame number `record` traced with last, which the resolve pass needs to find the pixel each mask pixel stands for.
    std::uint32_t traced_frame_ = 0;

    std::array<ImageHolder, 2> mask_history_{};
    std::array<ImageHolder, 2> depth_history_{};
    VkExtent2D history_extent_{};
    std::uint32_t history_slot_ = 0;
    bool history_initialised_ = false;
    // The history holds a mask made last frame with previous_view_projection_.
    bool history_valid_ = false;
    glm::mat4 previous_view_projection_{1.0F};
};
