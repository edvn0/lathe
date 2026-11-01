#include "rendering/ray_traced_shadows.hxx"

#include <tracy/Tracy.hpp>

#include "core/logger.hxx"
#include "core/paths.hxx"
#include "gpu/context.hxx"
#include "gpu/shader_stage.hxx"
#include "rendering/renderer.hxx"

namespace {

    // Mirrors PushConstants in rt_shadows.slang.
    struct Push {
        glm::mat4 inverse_view_projection;
        VkDeviceAddress instances;
        VkDeviceAddress materials;
        std::uint32_t depth;
        std::uint32_t result;
        std::uint32_t width;
        std::uint32_t height;
        float sun[3];
        float sun_spread;
        std::uint32_t frame;
        std::uint32_t light_count;
        VkDeviceAddress ubo;
        VkDeviceAddress lights;
        VkDeviceAddress cluster_lights;
        std::uint32_t flags;
    };
    static_assert(sizeof(Push) <= 256);

    constexpr auto group = 8U;

}

auto RayTracedShadows::create(Renderer &renderer) -> std::expected<void, RendererError> {
    if (!renderer.context().ray_query_supported || !renderer.scene_acceleration().ready()) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }
    auto registered = renderer.register_pipeline(PipelineRegisterInfo{
            .stages = {renderer::ShaderCompileRequest{
                    .source_path = data_path("assets/shaders/rt_shadows.slang"),
                    .entry_point = FlyString{"main_cs"},
                    .stage = renderer::ShaderStage::compute,
                    .include_directories = Paths::current().data_directories("assets/shaders"),
            }},
            .push_constant_ranges = {global_push_constant_range},
            .debug_name = "rt_shadows",
    });
    if (!registered) {
        return std::unexpected(registered.error());
    }
    shader_ = *registered;
    renderer_ = &renderer;
    renderer.set_ray_tracing_scene_enabled(true);
    return {};
}

auto RayTracedShadows::record(VkCommandBuffer command_buffer, std::uint32_t frame_index, Inputs const &inputs) -> void {
    ZoneScopedNC("Ray traced shadows", tracy::Color::DeepPink);
    auto const *set = renderer_->resolve_pipeline(shader_);
    if (set == nullptr || !set->valid()) {
        return;
    }
    set->bind(command_buffer);
    renderer_->resource_table().bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, set->layout());

    auto const push = Push{
            .inverse_view_projection = inputs.inverse_view_projection,
            .instances = inputs.instance_table,
            .materials = inputs.materials,
            .depth = inputs.depth_texture,
            .result = inputs.mask_storage,
            .width = inputs.width,
            .height = inputs.height,
            .sun = {inputs.sun.x, inputs.sun.y, inputs.sun.z},
            .sun_spread = settings_.sun_spread,
            .frame = frame_++,
            .light_count = inputs.light_count,
            .ubo = inputs.ubo,
            .lights = inputs.lights,
            .cluster_lights = inputs.cluster_lights,
            .flags = (inputs.trace_sun ? 1U : 0U) | (inputs.trace_local ? 2U : 0U),
    };
    vkCmdPushConstants(command_buffer, set->layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer, (inputs.width + group - 1U) / group, (inputs.height + group - 1U) / group, 1);
}
