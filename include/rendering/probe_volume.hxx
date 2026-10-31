#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "core/renderer_error.hxx"
#include "rendering/game_graph.hxx"

struct Renderer;

// The probe volume of the probe GI (docs/probe-gi.md, phase 2): a camera following grid of probes, addressed toroidally
// (a probe keeps its slot while the grid scrolls), and per frame a ray query pass that shoots rays_per_probe rays from
// every probe against the scene TLAS (SceneAccelerationStructure), then a pass that classifies the probes from those
// rays. Hits are shaded with the sun and the sky only for now; there is no blend, no atlas and no shading from the
// volume yet (phase 3).
//
// Everything is game graph passes in GameSlot::after_lighting on the graphics queue, behind the scene acceleration
// structure build, like RayTracingPoc. Resources: a persistent probe state buffer (two float4 per probe) and a
// transient ray data buffer (8 bytes per ray).
class ProbeVolume {
public:
    struct Settings {
        std::array<std::uint32_t, 3> counts{32, 8, 32};
        float spacing = 1.0F;
        // A multiple of 64.
        std::uint32_t rays_per_probe = 64;
        // A probe seeing more back faces than this fraction of its rays is inside geometry and is switched off.
        float back_face_limit = 0.25F;
    };

    ProbeVolume() = default;

    // Registers the shaders and turns the renderer's ray tracing scene on. Fails, harmlessly, without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;

    // Declares this frame's passes. Does nothing outside GameSlot::after_lighting or before create succeeded.
    auto declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection) -> void;

    [[nodiscard]] auto ready() const noexcept -> bool { return ready_; }
    [[nodiscard]] auto settings() noexcept -> Settings & { return settings_; }

    [[nodiscard]] auto probe_count() const noexcept -> std::uint32_t {
        return settings_.counts[0] * settings_.counts[1] * settings_.counts[2];
    }

    // Debug views, in the order of the modes in gi_probe_debug.slang. 0 leaves the picture alone.
    static constexpr auto view_names = std::array<std::string_view, 4>{"off", "radiance", "state", "back-faces"};
    auto set_view(std::uint32_t view) noexcept -> void { view_ = view; }

    auto destroy() noexcept -> void;

private:
    Renderer *renderer_ = nullptr;
    GameComputeShader trace_{};
    GameComputeShader classify_{};
    GameComputeShader debug_{};
    Settings settings_{};
    std::uint32_t view_ = 0;
    std::uint32_t frame_ = 0;
    bool ready_ = false;
};
