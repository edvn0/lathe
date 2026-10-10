#include "rendering/game_gpu.hxx"

#include "core/paths.hxx"
#include "gpu/context.hxx"
#include "gpu/shader_stage.hxx"
#include "rendering/renderer.hxx"

namespace {
    auto device_failure(DeviceError error) -> RendererError {
        return RendererError{
                .type = RendererErrorType::device_error,
                .cause = ErrorCause{Boxed<DeviceError>{error}},
        };
    }
}

auto GameGpu::register_compute(GameComputeShaderInfo const &info) -> std::expected<GameComputeShader, RendererError> {
    auto registered = renderer_->register_pipeline(PipelineRegisterInfo{
            .stages = {renderer::ShaderCompileRequest{
                    .source_path = data_path(info.source),
                    .entry_point = FlyString{std::string_view{info.entry_point}},
                    .stage = renderer::ShaderStage::compute,
            }},
            .push_constant_ranges = {global_push_constant_range},
            .debug_name = info.debug_name,
    });
    if (!registered) {
        return std::unexpected(registered.error());
    }
    return GameComputeShader{.node = *registered};
}

auto GameGpu::register_graphics(GameGraphicsShaderInfo const &info)
        -> std::expected<GameGraphicsShader, RendererError> {
    auto registered = renderer_->register_pipeline(PipelineRegisterInfo{
            .stages =
                    {
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path(info.source),
                                    .entry_point = FlyString{std::string_view{info.vertex_entry_point}},
                                    .stage = renderer::ShaderStage::vertex,
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path(info.source),
                                    .entry_point = FlyString{std::string_view{info.fragment_entry_point}},
                                    .stage = renderer::ShaderStage::fragment,
                            },
                    },
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {renderer_->hdr_format()},
            .depth_format = renderer_->depth_format(),
            .samples = renderer_->samples(),
            .blending = false,
            .debug_name = info.debug_name,
    });
    if (!registered) {
        return std::unexpected(registered.error());
    }
    return GameGraphicsShader{.node = *registered};
}

auto GameGpu::create_buffer(GameBufferInfo const &info) -> std::expected<GameBufferHandle, RendererError> {
    auto &context = renderer_->context();
    auto entry = Entry{.name = info.debug_name};

    for (auto slot = std::uint32_t{0}; slot < context.swapchain.frame_count(); ++slot) {
        auto create_info = BufferCreateInfo{
                .size = info.size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .memory = BufferMemory::device,
                .debug_name = entry.name,
        };
        // The engine imports frame buffers as concurrent when the queues are in different families.
        if (context.queue_families.compute != context.queue_families.graphics) {
            create_info.concurrent_families = {context.queue_families.graphics, context.queue_families.compute};
            create_info.concurrent_family_count = 2;
        }

        auto created = Buffer::create(context, create_info);
        if (!created) {
            return std::unexpected(device_failure(created.error()));
        }
        entry.copies.push_back(std::move(*created));
    }

    // Zero-filled, so a pass can treat a fresh buffer as empty rather than as garbage.
    context.one_time_submit([&](VkCommandBuffer command_buffer) {
        for (auto const &copy: entry.copies) {
            vkCmdFillBuffer(command_buffer, copy.buffer, 0, VK_WHOLE_SIZE, 0);
        }
    });

    buffers_.push_back(std::move(entry));
    return GameBufferHandle{.index = static_cast<std::uint32_t>(buffers_.size() - 1), .generation = 1};
}

auto GameGpu::buffer(GameBufferHandle handle, std::uint32_t frame_slot) const noexcept -> Buffer const * {
    if (!handle.valid() || handle.index >= buffers_.size() || frame_slot >= buffers_[handle.index].copies.size()) {
        return nullptr;
    }
    return &buffers_[handle.index].copies[frame_slot];
}

auto GameGpu::buffer_name(GameBufferHandle handle) const noexcept -> std::string_view {
    return handle.valid() && handle.index < buffers_.size() ? std::string_view{buffers_[handle.index].name}
                                                            : std::string_view{};
}
