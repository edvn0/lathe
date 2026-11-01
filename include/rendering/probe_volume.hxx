#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "core/config.hxx"
#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "gpu/image_storage.hxx"
#include "rendering/environment.hxx"
#include "rendering/pipeline_graph_repository.hxx"

struct Renderer;

// The probe GI volume (docs/probe-gi.md): a camera following grid of probes, addressed toroidally (a probe keeps its
// slot while the grid scrolls), lit by ray queries against the scene TLAS (SceneAccelerationStructure) and blended into
// two octahedral atlases the forward shader samples (gi_probes.slang).
//
// The renderer owns it and declares its engine passes (renderer_frame_graph.cxx): trace (one ray per thread), classify
// (per probe: relocation, inside geometry), blend (irradiance and distance atlases), border (seamless filtering). It
// holds the resources: the probe state buffer, the ray data buffer and the atlases (renderer owned images imported
// into the frame graph). Hits are shaded with the sun and the sky and the surface's base colour; emissive and local lights come in phase 4.
class ProbeVolume {
public:
    struct Settings {
        // Fixed when the volume is created.
        std::array<std::uint32_t, 3> counts{32, 8, 32};
        float spacing = 1.0F;
        // A multiple of 64 up to max_rays_per_probe.
        std::uint32_t rays_per_probe = 64;
        // A probe seeing more back faces than this fraction of its rays is inside geometry.
        float back_face_limit = 0.25F;
        float hysteresis = 0.97F;
        float intensity = 1.0F;
        float normal_bias = 0.25F;
        float view_bias = 0.1F;
        // Strength of a small emissive piece (a bulb) as a light, like a point light's intensity. A bulb's physical
        // emission lights nothing, so this is an art setting; the loader's synthesised lights use 25 with no probe
        // smoothing and no bounce.
        float emitter_intensity = 200.0F;
        // 0 off, 1 irradiance only (the diffuse term on grey), 2 probe state (green active, red off).
        std::uint32_t debug_view = 0;
        // 0 the near cascade, 1 the coarser one: the trace samples its own cascade for the bounce.
        std::uint32_t cascade = 0;
        // The passes run every this many frames; in between the volume is still sampled, and does not scroll.
        std::uint32_t update_interval = 1;
        // A probe whose lighting stayed the same for this many traced frames sleeps: the trace, classify and blend skip it
        // except one frame in sleep_refresh (staggered by slot), and a refresh that finds the lighting changed wakes it.
        // 0 keeps every probe awake.
        std::uint32_t sleep_after = 24;
        std::uint32_t sleep_refresh = 8;
    };

    // Mirrors GpuGiBlock in scene_types.slang: the end of the UBO.
    struct GpuBlock {
        VkDeviceAddress probe_state = 0;
        std::uint32_t irradiance_texture = 0;
        std::uint32_t distance_texture = 0;
        std::int32_t base_x = 0;
        std::int32_t base_y = 0;
        std::int32_t base_z = 0;
        std::uint32_t flags = 0;
        std::uint32_t count_x = 0;
        std::uint32_t count_y = 0;
        std::uint32_t count_z = 0;
        float spacing = 1.0F;
        float intensity = 1.0F;
        float normal_bias = 0.0F;
        float view_bias = 0.0F;
        float padding = 0.0F;
    };
    static_assert(sizeof(GpuBlock) == 64);

    // An emissive piece lit through the explicit emitter list (a bulb): centre and bounding radius in world space, and the
    // material its emission comes from. Mirrors GpuEmitter in gi_probe_trace.slang.
    struct Emitter {
        glm::vec3 centre{0.0F};
        float radius = 0.1F;
        std::uint32_t material_index = 0;
        std::uint32_t padding[3]{};
    };
    static_assert(sizeof(Emitter) == 32);

    // Emissive pieces at most this big (bounding radius, metres) are explicit emitters.
    static constexpr auto small_emitter_radius = 0.3F;
    static constexpr auto max_emitters = std::uint32_t{4096};
    // The emitter buffer of a frame slot, in 32 bit words (mirrored in gi_probe_trace.slang): the emitters, then a header
    // of the uniform grid over them (origin xyz, 1 / cell size, cell counts xyz, total cells), the cells' (first index,
    // count) pairs, and the emitter indices of every cell. A hit looks at the emitters of its own cell only.
    static constexpr auto emitter_words = std::uint32_t{8};
    static constexpr auto max_grid_cells = std::uint32_t{16384};
    static constexpr auto grid_header_word = max_emitters * emitter_words;
    static constexpr auto grid_ranges_word = grid_header_word + 8;
    static constexpr auto grid_indices_word = grid_ranges_word + max_grid_cells * 2;
    static constexpr auto emitter_buffer_words = grid_indices_word + max_emitters * 8;

    static constexpr auto max_rays_per_probe = std::uint32_t{256};
    static constexpr auto irradiance_tile = std::uint32_t{10};
    static constexpr auto distance_tile = std::uint32_t{18};
    static constexpr auto debug_view_names = std::array<std::string_view, 3>{"off", "irradiance", "state"};

    ProbeVolume() = default;
    ~ProbeVolume();

    ProbeVolume(ProbeVolume const &) = delete;
    auto operator=(ProbeVolume const &) -> ProbeVolume & = delete;

    // Registers the shaders and creates the buffers and atlases. Fails, harmlessly, without ray query.
    [[nodiscard]] auto create(Renderer &renderer) -> std::expected<void, RendererError>;
    auto destroy() noexcept -> void;

    [[nodiscard]] auto ready() const noexcept -> bool { return renderer_ != nullptr; }
    [[nodiscard]] auto settings() noexcept -> Settings & { return settings_; }
    [[nodiscard]] auto settings() const noexcept -> Settings const & { return settings_; }
    [[nodiscard]] auto probe_count() const noexcept -> std::uint32_t {
        return settings_.counts[0] * settings_.counts[1] * settings_.counts[2];
    }

    // Per frame, before the UBO is written: places the volume around `camera` and picks this frame's ray rotation.
    auto begin_frame(glm::vec3 camera, glm::vec3 sun_direction, glm::vec3 sun_radiance,
                     EnvironmentUboBlock const &environment) -> void;
    // True when the frame has a volume to trace and sample.
    [[nodiscard]] auto active() const noexcept -> bool { return active_; }
    // True on the frames the passes run (see Settings::update_interval).
    [[nodiscard]] auto updating() const noexcept -> bool { return active_ && updating_; }
    [[nodiscard]] auto gpu_block() const noexcept -> GpuBlock;

    // The resources the frame graph imports.
    [[nodiscard]] auto state_buffer() const noexcept -> Buffer const & { return state_; }
    [[nodiscard]] auto ray_buffer() const noexcept -> Buffer const & { return rays_; }
    [[nodiscard]] auto irradiance_image() const -> Image const &;
    [[nodiscard]] auto distance_image() const -> Image const &;
    // False until the atlases have been through one frame: their first use starts from an undefined layout.
    [[nodiscard]] auto atlases_initialised() const noexcept -> bool { return atlases_initialised_; }
    auto mark_atlases_initialised() noexcept -> void { atlases_initialised_ = true; }

    // The frame's emitters: begin_emitters clears the list of this frame slot, add_emitter appends (false when full).
    auto begin_emitters(std::uint32_t frame_index) -> void;
    auto add_emitter(Emitter const &emitter) -> bool;
    // After the last add_emitter of the frame: builds the grid the trace looks emitters up in.
    auto finish_emitters() -> void;

    // Pass recording. `instance_table` is SceneAccelerationStructure::instance_table_address(), `materials` the address of
    // the material table, `ubo` this frame's UBO and `lights` its GpuLight array.
    struct TraceInputs {
        VkDeviceAddress instance_table = 0;
        VkDeviceAddress materials = 0;
        VkDeviceAddress ubo = 0;
        VkDeviceAddress lights = 0;
        std::uint32_t light_count = 0;
    };
    auto record_trace(VkCommandBuffer command_buffer, std::uint32_t frame_index, TraceInputs const &inputs) -> void;
    auto record_classify(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;
    auto record_blend(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;
    auto record_border(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void;

private:
    [[nodiscard]] auto bind(VkCommandBuffer command_buffer, std::uint32_t frame_index, PipelineNodeHandle node)
            -> VkPipelineLayout;
    [[nodiscard]] auto irradiance_extent() const noexcept -> VkExtent2D;
    [[nodiscard]] auto distance_extent() const noexcept -> VkExtent2D;

    Renderer *renderer_ = nullptr;
    Settings settings_{};

    PipelineNodeHandle trace_{};
    PipelineNodeHandle classify_{};
    PipelineNodeHandle blend_irradiance_{};
    PipelineNodeHandle blend_distance_{};
    PipelineNodeHandle border_{};

    Buffer state_{};
    Buffer rays_{};
    std::array<Buffer, frames_in_flight> emitter_buffers_{};
    std::array<std::uint32_t, frames_in_flight> emitter_counts_{};
    std::vector<Emitter> emitters_this_frame_;
    std::uint32_t emitter_slot_ = 0;
    ImageHolder irradiance_{};
    ImageHolder distance_{};
    bool atlases_initialised_ = false;
    bool state_cleared_ = false;

    // This frame.
    bool active_ = false;
    bool updating_ = false;
    std::uint32_t frame_ = 0;
    std::array<std::int32_t, 3> base_{};
    glm::quat rotation_{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 sun_direction_{0.0F, 1.0F, 0.0F};
    glm::vec3 sun_radiance_{0.0F};
    EnvironmentUboBlock environment_{};
};
