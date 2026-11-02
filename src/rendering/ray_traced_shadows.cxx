#include "rendering/ray_traced_shadows.hxx"

#include <algorithm>

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
        std::uint32_t resolution_shift;
    };
    static_assert(sizeof(Push) <= 256);

    // Mirrors PushConstants in rt_shadow_resolve.slang.
    struct ResolvePush {
        glm::mat4 inverse_view_projection;
        glm::mat4 view_projection;
        glm::mat4 previous_view_projection;
        std::uint32_t depth;
        std::uint32_t raw;
        std::uint32_t history_mask;
        std::uint32_t history_depth;
        std::uint32_t result_mask;
        std::uint32_t result_depth;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t trace_frame;
        std::uint32_t resolution_shift;
        std::uint32_t flags;
        float blend;
    };
    static_assert(sizeof(ResolvePush) <= 256);

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

    auto resolve = renderer.register_pipeline(PipelineRegisterInfo{
            .stages = {renderer::ShaderCompileRequest{
                    .source_path = data_path("assets/shaders/rt_shadow_resolve.slang"),
                    .entry_point = FlyString{"main_cs"},
                    .stage = renderer::ShaderStage::compute,
                    .include_directories = Paths::current().data_directories("assets/shaders"),
            }},
            .push_constant_ranges = {global_push_constant_range},
            .debug_name = "rt_shadow_resolve",
    });
    if (!resolve) {
        return std::unexpected(resolve.error());
    }
    resolve_shader_ = *resolve;
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

    auto const shift = std::min(settings_.resolution_shift, max_resolution_shift);
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
            .frame = (traced_frame_ = frame_++),
            .light_count = inputs.light_count,
            .ubo = inputs.ubo,
            .lights = inputs.lights,
            .cluster_lights = inputs.cluster_lights,
            .flags = (inputs.trace_sun ? 1U : 0U) | (inputs.trace_local ? 2U : 0U),
            .resolution_shift = shift,
    };
    vkCmdPushConstants(command_buffer, set->layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer, (mask_extent(inputs.width, shift) + group - 1U) / group,
                  (mask_extent(inputs.height, shift) + group - 1U) / group, 1);
}

auto RayTracedShadows::prepare_history(VkExtent2D screen, VkExtent2D mask) -> bool {
    if (renderer_ == nullptr || !settings_.temporal_filter) {
        return false;
    }
    if (mask.width == history_extent_.width && mask.height == history_extent_.height && mask_history_[0]) {
        return true;
    }

    // The images are in use by frames in flight: this happens when the viewport is resized, which is rare.
    static_cast<void>(renderer_->wait_idle());
    mask_history_ = {};
    depth_history_ = {};
    history_valid_ = false;
    history_initialised_ = false;
    history_slot_ = 0;
    history_extent_ = {};
    static_cast<void>(screen);

    auto const make = [&](VkFormat format, char const *name) {
        return create_held_image(
                renderer_->image_storage(),
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = mask.width, .height = mask.height, .depth = 1},
                        .format = format,
                        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .image_type = VK_IMAGE_TYPE_2D,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                            image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                        .flags = 0,
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = VK_IMAGE_TILING_OPTIMAL,
                        .mip_levels = 1,
                        .array_layers = 1,
                        .debug_name = name,
                });
    };
    auto mask_a = make(VK_FORMAT_R8G8B8A8_UNORM, "rt_shadow_history_mask_0");
    auto mask_b = make(VK_FORMAT_R8G8B8A8_UNORM, "rt_shadow_history_mask_1");
    auto depth_a = make(VK_FORMAT_R32_SFLOAT, "rt_shadow_history_depth_0");
    auto depth_b = make(VK_FORMAT_R32_SFLOAT, "rt_shadow_history_depth_1");
    if (!mask_a || !mask_b || !depth_a || !depth_b) {
        error("RayTracedShadows: could not create the temporal history, the mask stays unfiltered");
        return false;
    }
    mask_history_ = {std::move(*mask_a), std::move(*mask_b)};
    depth_history_ = {std::move(*depth_a), std::move(*depth_b)};
    history_extent_ = mask;
    return true;
}

auto RayTracedShadows::finish_history(glm::mat4 const &view_projection) -> void {
    history_slot_ ^= 1U;
    history_valid_ = true;
    history_initialised_ = true;
    previous_view_projection_ = view_projection;
}

auto RayTracedShadows::record_resolve(VkCommandBuffer command_buffer, std::uint32_t frame_index,
                                      ResolveInputs const &inputs) -> void {
    ZoneScopedNC("Ray traced shadow resolve", tracy::Color::DeepPink);
    auto const *set = renderer_->resolve_pipeline(resolve_shader_);
    if (set == nullptr || !set->valid()) {
        return;
    }
    set->bind(command_buffer);
    renderer_->resource_table().bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, set->layout());

    auto const shift = std::min(settings_.resolution_shift, max_resolution_shift);
    auto const read = history_read_slot();
    auto const write = history_write_slot();
    auto const push = ResolvePush{
            .inverse_view_projection = inputs.inverse_view_projection,
            .view_projection = inputs.view_projection,
            .previous_view_projection = previous_view_projection_,
            .depth = inputs.depth_texture,
            .raw = inputs.raw_mask,
            .history_mask = mask_history_[read].handle().index,
            .history_depth = depth_history_[read].handle().index,
            .result_mask = mask_history_[write].handle().index,
            .result_depth = depth_history_[write].handle().index,
            .width = inputs.width,
            .height = inputs.height,
            .trace_frame = traced_frame_,
            .resolution_shift = shift,
            .flags = history_valid_ ? 1U : 0U,
            .blend = std::clamp(settings_.temporal_blend, 0.01F, 1.0F),
    };
    vkCmdPushConstants(command_buffer, set->layout(), VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer, (mask_extent(inputs.width, shift) + group - 1U) / group,
                  (mask_extent(inputs.height, shift) + group - 1U) / group, 1);
}
