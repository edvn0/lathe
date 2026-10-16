#include "rendering/overlay_shader.hxx"

#include "core/paths.hxx"
#include "gpu/shader_stage.hxx"
#include "rendering/renderer.hxx"

namespace {
    [[nodiscard]] auto same_target(OverlayScope const &a, OverlayScope const &b) noexcept -> bool {
        return a.colour_format == b.colour_format && a.depth_format == b.depth_format && a.samples == b.samples &&
               a.colour_attachment_count == b.colour_attachment_count;
    }
} // namespace

auto OverlayShader::bind(OverlayRecordContext const &context) -> std::expected<BoundOverlayShader, RendererError> {
    if (!node_.valid() || !same_target(scope_, context.scope)) {
        auto const stage = [&](std::string const &entry, renderer::ShaderStage kind) {
            return renderer::ShaderCompileRequest{
                    .source_path = data_path(info_.source),
                    .entry_point = FlyString{std::string_view{entry}},
                    .stage = kind,
            };
        };

        auto registered = renderer_->register_pipeline(PipelineRegisterInfo{
                .stages = {stage(info_.vertex_entry_point, renderer::ShaderStage::vertex),
                           stage(info_.fragment_entry_point, renderer::ShaderStage::fragment)},
                .push_constant_ranges = {global_push_constant_range},
                .colour_formats = {context.scope.colour_format},
                .dynamic_states = {},
                .depth_format = context.scope.depth_format,
                .stencil_format = VK_FORMAT_UNDEFINED,
                .samples = context.scope.samples,
                .blending = info_.blending,
                .debug_name = info_.debug_name,
        });

        if (!registered) {
            return std::unexpected(registered.error());
        }

        node_ = *registered;
        scope_ = context.scope;
    }

    auto const *pipeline = renderer_->resolve_pipeline(node_);

    if (pipeline == nullptr || !pipeline->valid()) {
        // Still compiling; the overlay skips this frame and draws once the shader is ready.
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_pipeline});
    }

    pipeline->bind(context.command_buffer);
    renderer_->resource_table().bind(context.command_buffer, context.frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     pipeline->layout());

    return BoundOverlayShader{context.command_buffer, pipeline->layout()};
}
