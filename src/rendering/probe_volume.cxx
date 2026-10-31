#include "rendering/probe_volume.hxx"

#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

namespace {
    constexpr auto state_buffer_name = std::string_view{"gi_probe_state"};
    constexpr auto trace_slice = std::uint32_t{64};
    constexpr auto bytes_per_ray = VkDeviceSize{8};
    constexpr auto bytes_per_probe_state = VkDeviceSize{32};

    // Mirrors PushConstants in gi_probe_trace.slang.
    struct TraceParams {
        BufferRead state;
        BufferWrite rays;
        VkDeviceAddress instances = 0;
        std::int32_t base_x = 0;
        std::int32_t base_y = 0;
        std::int32_t base_z = 0;
        std::uint32_t count_x = 0;
        std::uint32_t count_y = 0;
        std::uint32_t count_z = 0;
        float spacing = 1.0F;
        std::uint32_t rays_per_probe = 64;
        float rotation_x = 0.0F;
        float rotation_y = 0.0F;
        float rotation_z = 0.0F;
        float rotation_w = 1.0F;
        float sun_x = 0.0F;
        float sun_y = 1.0F;
        float sun_z = 0.0F;
        float sun_r = 0.0F;
        float sun_g = 0.0F;
        float sun_b = 0.0F;
        std::uint32_t sky_cube = 0;
        std::uint32_t sky_sampler = 0;
        float env_cos = 1.0F;
        float env_sin = 0.0F;
        float sky_intensity = 1.0F;

        using Layout = PushLayout<&TraceParams::state, &TraceParams::rays, &TraceParams::instances,
                                  &TraceParams::base_x, &TraceParams::base_y, &TraceParams::base_z,
                                  &TraceParams::count_x, &TraceParams::count_y, &TraceParams::count_z,
                                  &TraceParams::spacing, &TraceParams::rays_per_probe, &TraceParams::rotation_x,
                                  &TraceParams::rotation_y, &TraceParams::rotation_z, &TraceParams::rotation_w,
                                  &TraceParams::sun_x, &TraceParams::sun_y, &TraceParams::sun_z, &TraceParams::sun_r,
                                  &TraceParams::sun_g, &TraceParams::sun_b, &TraceParams::sky_cube,
                                  &TraceParams::sky_sampler, &TraceParams::env_cos, &TraceParams::env_sin,
                                  &TraceParams::sky_intensity>;
    };

    // Mirrors Push in gi_probe_classify.slang.
    struct ClassifyParams {
        BufferReadWrite state;
        BufferRead rays;
        std::uint32_t probe_count = 0;
        std::uint32_t rays_per_probe = 64;
        float back_face_limit = 0.25F;

        using Layout = PushLayout<&ClassifyParams::state, &ClassifyParams::rays, &ClassifyParams::probe_count,
                                  &ClassifyParams::rays_per_probe, &ClassifyParams::back_face_limit>;
    };

    // Mirrors Push in gi_probe_debug.slang.
    struct DebugParams {
        glm::mat4 inverse_view_projection{1.0F};
        BufferRead state;
        ImageRead hdr;
        ImageRead depth;
        ImageWrite result;
        std::int32_t base_x = 0;
        std::int32_t base_y = 0;
        std::int32_t base_z = 0;
        std::uint32_t count_x = 0;
        std::uint32_t count_y = 0;
        std::uint32_t count_z = 0;
        float spacing = 1.0F;
        std::uint32_t mode = 0;

        using Layout = PushLayout<&DebugParams::inverse_view_projection, &DebugParams::state, &DebugParams::hdr,
                                  &DebugParams::depth, &DebugParams::result, &DebugParams::base_x,
                                  &DebugParams::base_y, &DebugParams::base_z, &DebugParams::count_x,
                                  &DebugParams::count_y, &DebugParams::count_z, &DebugParams::spacing,
                                  &DebugParams::mode>;
    };

    static_assert(GameGraph::push_size<TraceParams>(TraceParams::Layout{}) <= max_game_push_bytes);
    static_assert(GameGraph::push_size<ClassifyParams>(ClassifyParams::Layout{}) <= max_game_push_bytes);
    static_assert(GameGraph::push_size<DebugParams>(DebugParams::Layout{}) <= max_game_push_bytes);

    auto hash(std::uint32_t value) -> std::uint32_t {
        value = value * 747796405U + 2891336453U;
        value = ((value >> ((value >> 28U) + 4U)) ^ value) * 277803737U;
        return (value >> 22U) ^ value;
    }

    auto unit_random(std::uint32_t seed) -> float {
        return static_cast<float>(hash(seed) & 0xFFFFFFU) / 16777216.0F;
    }

    // A uniformly distributed random rotation (Shoemake), a new one per frame so the rays sample the sphere evenly over
    // time.
    auto random_rotation(std::uint32_t frame) -> glm::quat {
        auto const u1 = unit_random(frame * 3U + 1U);
        auto const u2 = unit_random(frame * 3U + 2U) * glm::two_pi<float>();
        auto const u3 = unit_random(frame * 3U + 3U) * glm::two_pi<float>();
        auto const a = std::sqrt(1.0F - u1);
        auto const b = std::sqrt(u1);
        return glm::quat{b * std::cos(u3), a * std::sin(u2), a * std::cos(u2), b * std::sin(u3)};
    }
}

auto ProbeVolume::create(Renderer &renderer) -> std::expected<void, RendererError> {
    if (!renderer.context().ray_query_supported || !renderer.scene_acceleration().ready()) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    auto const make = [&](char const *source, char const *name) {
        return renderer.game_gpu().register_compute(
                {.source = source, .entry_point = "main_cs", .debug_name = name});
    };
    auto trace = make("assets/shaders/gi_probe_trace.slang", "gi_probe_trace");
    if (!trace) {
        return std::unexpected(trace.error());
    }
    auto classify = make("assets/shaders/gi_probe_classify.slang", "gi_probe_classify");
    if (!classify) {
        return std::unexpected(classify.error());
    }
    auto debug = make("assets/shaders/gi_probe_debug.slang", "gi_probe_debug");
    if (!debug) {
        return std::unexpected(debug.error());
    }
    trace_ = *trace;
    classify_ = *classify;
    debug_ = *debug;
    renderer_ = &renderer;

    renderer.set_ray_tracing_scene_enabled(true);

    ready_ = true;
    info("Probe volume ready: {}x{}x{} probes, {} rays each, spacing {} m", settings_.counts[0], settings_.counts[1],
         settings_.counts[2], settings_.rays_per_probe, settings_.spacing);
    return {};
}

auto ProbeVolume::destroy() noexcept -> void {
    if (renderer_ != nullptr && ready_) {
        renderer_->game_gpu().release_persistent_buffer(state_buffer_name);
    }
    ready_ = false;
}

auto ProbeVolume::declare(GameGraph &graph, glm::mat4 const &view, glm::mat4 const &projection) -> void {
    if (!ready_ || graph.slot() != GameSlot::after_lighting) {
        return;
    }
    auto const table = renderer_->scene_acceleration().instance_table_address();
    if (table == 0 || !renderer_->scene_acceleration().active()) {
        return;
    }
    ZoneScopedNC("Probe volume declare", tracy::Color::Gold);

    settings_.rays_per_probe = std::max(trace_slice, settings_.rays_per_probe / trace_slice * trace_slice);
    auto const probes = probe_count();
    auto const rays = probes * settings_.rays_per_probe;

    // The grid is centred on the camera and moves in whole probe steps.
    auto const camera = glm::vec3{glm::inverse(view)[3]};
    auto base = glm::ivec3{};
    for (auto axis = 0; axis < 3; ++axis) {
        auto const centre = static_cast<std::int32_t>(std::floor(camera[axis] / settings_.spacing));
        base[axis] = centre - static_cast<std::int32_t>(settings_.counts[static_cast<std::size_t>(axis)] / 2);
    }

    auto const &light = renderer_->directional_light();
    auto const sun = glm::normalize(light.direction);
    auto const sun_radiance = light.colour * light.intensity;
    auto const environment = renderer_->environment_system().ubo_block();
    auto const rotation = random_rotation(frame_++);

    auto state = graph.persistent_buffer(state_buffer_name, {.size = probes * bytes_per_probe_state});
    auto ray_data = graph.create_buffer({.size = rays * bytes_per_ray}, "gi_ray_data");

    auto trace = TraceParams{
            .state = {state},
            .rays = {ray_data},
            .instances = table,
            .base_x = base.x,
            .base_y = base.y,
            .base_z = base.z,
            .count_x = settings_.counts[0],
            .count_y = settings_.counts[1],
            .count_z = settings_.counts[2],
            .spacing = settings_.spacing,
            .rays_per_probe = settings_.rays_per_probe,
            .rotation_x = rotation.x,
            .rotation_y = rotation.y,
            .rotation_z = rotation.z,
            .rotation_w = rotation.w,
            .sun_x = sun.x,
            .sun_y = sun.y,
            .sun_z = sun.z,
            .sun_r = sun_radiance.r,
            .sun_g = sun_radiance.g,
            .sun_b = sun_radiance.b,
            .sky_cube = environment.prefilter_cube_texture,
            .sky_sampler = environment.sampler,
            .env_cos = environment.rotation.x,
            .env_sin = -environment.rotation.y,
            .sky_intensity = environment.intensity.y,
    };
    // Group x is the probe slot, group y a slice of 64 rays: the dispatch is probes x slices threads-wise.
    graph.add_compute("gi_probe_trace", trace, trace_,
                      Threads{.x = probes, .y = settings_.rays_per_probe / trace_slice, .group_x = 1, .group_y = 1},
                      {.queue = frame_graph::QueueAffinity::graphics, .label = "Probe trace", .color = 0xFFD700});

    auto classify = ClassifyParams{
            .state = {state},
            .rays = {trace.rays.buffer},
            .probe_count = probes,
            .rays_per_probe = settings_.rays_per_probe,
            .back_face_limit = settings_.back_face_limit,
    };
    graph.add_compute("gi_probe_classify", classify, classify_, Threads{.x = probes, .group_x = 64},
                      {.queue = frame_graph::QueueAffinity::graphics,
                       .side_effect = true,
                       .label = "Probe classify",
                       .color = 0xFFD700});

    auto const hdr = graph.scene_hdr();
    auto const depth = graph.scene_depth();
    if (view_ == 0 || !hdr || !depth) {
        return;
    }
    auto debug = DebugParams{
            .inverse_view_projection = glm::inverse(projection * view),
            .state = {classify.state.buffer},
            .hdr = ImageRead{*hdr},
            .depth = ImageRead{*depth},
            .result = ImageWrite{GameImageDesc{.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "gi_probe_debug"}},
            .base_x = base.x,
            .base_y = base.y,
            .base_z = base.z,
            .count_x = settings_.counts[0],
            .count_y = settings_.counts[1],
            .count_z = settings_.counts[2],
            .spacing = settings_.spacing,
            .mode = view_,
    };
    graph.add_compute("gi_probe_debug", debug, debug_, Threads::of_extent(graph.scene_extent(), 8, 8),
                      {.queue = frame_graph::QueueAffinity::graphics, .label = "Probe debug", .color = 0xFFD700});
    graph.replace_scene_colour(debug.result.image);
}
