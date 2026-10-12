#include "rendering/game_gpu.hxx"

#include <algorithm>

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
                    .include_directories = {data_path("assets/shaders").absolute()},
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
                                    .include_directories = {data_path("assets/shaders").absolute()},
                            },
                            renderer::ShaderCompileRequest{
                                    .source_path = data_path(info.source),
                                    .entry_point = FlyString{std::string_view{info.fragment_entry_point}},
                                    .stage = renderer::ShaderStage::fragment,
                                    .include_directories = {data_path("assets/shaders").absolute()},
                            },
                    },
            .push_constant_ranges = {global_push_constant_range},
            .colour_formats = {renderer_->hdr_format()},
            .depth_format = renderer_->depth_format(),
            .samples = renderer_->samples(),
            .blending = info.blending,
            .debug_name = info.debug_name,
    });
    if (!registered) {
        return std::unexpected(registered.error());
    }
    return GameGraphicsShader{.node = *registered, .blending = info.blending};
}

auto GameGpu::create_storage_buffer(VkDeviceSize size, std::string_view name, bool zero)
        -> std::expected<Buffer, RendererError> {
    auto &context = renderer_->context();
    auto create_info = BufferCreateInfo{
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memory = BufferMemory::device,
            .debug_name = name,
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
    if (zero) {
        // Nothing has used the new buffer yet, so filling it outside the frame cannot race anything.
        context.one_time_submit([&](VkCommandBuffer command_buffer) {
            vkCmdFillBuffer(command_buffer, created->buffer, 0, VK_WHOLE_SIZE, 0);
        });
    }
    return std::move(*created);
}

auto GameGpu::begin_frame(std::uint32_t frame_slot) -> void {
    if (frame_slot >= frame_buffers_.size()) {
        frame_buffers_.resize(frame_slot + 1);
    }

    // A buffer this slot did not ask for last time round is not coming back soon.
    auto &pool = frame_buffers_[frame_slot];
    std::erase_if(pool, [](FrameBuffer const &entry) { return !entry.used; });
    for (auto &entry: pool) {
        entry.used = false;
    }

    // A retired buffer was last used by a frame recorded before the one that retired it, and every frame in flight
    // has been recorded again by the time the count runs out.
    for (auto &retired: retired_) {
        --retired.frames_left;
    }
    std::erase_if(retired_, [](Retired const &retired) { return retired.frames_left == 0; });
}

auto GameGpu::frame_buffer_count() const noexcept -> std::size_t {
    auto count = std::size_t{0};
    for (auto const &pool: frame_buffers_) {
        count += pool.size();
    }
    return count;
}

auto GameGpu::acquire_frame_buffer(std::uint32_t frame_slot, VkDeviceSize size, std::string_view name)
        -> std::expected<Buffer const *, RendererError> {
    if (frame_slot >= frame_buffers_.size()) {
        frame_buffers_.resize(frame_slot + 1);
    }
    auto &pool = frame_buffers_[frame_slot];

    // The smallest free buffer that is big enough, so one large buffer is not spent on a small request.
    auto *best = static_cast<FrameBuffer *>(nullptr);
    for (auto &entry: pool) {
        if (!entry.used && entry.buffer.size() >= size && (best == nullptr || entry.buffer.size() < best->buffer.size())) {
            best = &entry;
        }
    }
    if (best == nullptr) {
        auto created = create_storage_buffer(size, name, false);
        if (!created) {
            return std::unexpected(created.error());
        }
        best = &pool.emplace_back(FrameBuffer{.buffer = std::move(*created)});
    }
    best->used = true;
    return &best->buffer;
}

auto GameGpu::retire(Buffer buffer) -> void {
    // The old buffer may still be read by frames in flight, so it is destroyed after one full cycle of frame slots.
    auto const frames = std::max(renderer_->context().swapchain.frame_count(), std::uint32_t{1});
    retired_.push_back(Retired{.buffer = std::move(buffer), .frames_left = frames + 1});
}

auto GameGpu::release_persistent_buffer(std::string_view name) -> void {
    auto const found = persistent_.find(name);
    if (found == persistent_.end()) {
        return;
    }
    retire(std::move(found->second));
    persistent_.erase(found);
}

auto GameGpu::acquire_persistent_buffer(std::string_view name, VkDeviceSize size)
        -> std::expected<Buffer const *, RendererError> {
    auto const found = persistent_.find(name);
    if (found != persistent_.end() && found->second.size() == size) {
        return &found->second;
    }

    auto created = create_storage_buffer(size, name, true);
    if (!created) {
        return std::unexpected(created.error());
    }
    if (found != persistent_.end()) {
        retire(std::move(found->second));
        found->second = std::move(*created);
        return &found->second;
    }
    return &persistent_.emplace(std::string{name}, std::move(*created)).first->second;
}
