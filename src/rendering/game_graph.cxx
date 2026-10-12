#include "rendering/game_graph.hxx"

#include <algorithm>
#include <cstring>
#include <format>
#include <variant>

#include "core/logger.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/image.hxx"
#include "gpu/shader_object.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/renderer.hxx"

namespace {
    constexpr auto compute_stage = frame_graph::stages_of(frame_graph::ShaderStage::compute);

    constexpr auto scene_draw_stages = static_cast<frame_graph::ShaderStages>(
            frame_graph::ShaderStage::vertex | frame_graph::ShaderStage::task | frame_graph::ShaderStage::mesh |
            frame_graph::ShaderStage::fragment);
}

auto GameComputeContext::fail(std::string_view what) -> void {
    if (violation_.empty()) {
        violation_ = what;
    }
}

auto GameComputeContext::bind(GameComputeShader shader) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (services_->renderer == nullptr) {
        fail("has no renderer to bind a shader with");
        return;
    }
    auto const *set = services_->renderer->resolve_pipeline(shader.node);
    if (set == nullptr || !set->valid()) {
        skipped_ = true;
        return;
    }

    set->bind(pass_->command_buffer);
    layout_ = set->layout();
    services_->renderer->resource_table().bind(pass_->command_buffer, pass_->frame_index,
                                               VK_PIPELINE_BIND_POINT_COMPUTE, layout_);
}

auto GameComputeContext::begin(GameComputeShader shader, Threads threads) -> void {
    threads_ = threads;
    bind(shader);
}

auto GameComputeContext::dispatch() -> void {
    if (!violation_.empty()) {
        return;
    }
    if (!threads_) {
        fail("dispatched without declared threads");
        return;
    }
    dispatch_threads(threads_->x, threads_->group_x, threads_->y, threads_->group_y);
}

auto GameComputeContext::push_bytes(void const *data, std::size_t size) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (skipped_) {
        return;
    }
    if (layout_ == VK_NULL_HANDLE) {
        fail("pushed constants before binding a shader");
        return;
    }
    vkCmdPushConstants(pass_->command_buffer, layout_, VK_SHADER_STAGE_ALL, 0, static_cast<std::uint32_t>(size), data);
}

auto GameComputeContext::image_index(DeclaredImage image, bool writable) -> std::uint32_t {
    if (!violation_.empty()) {
        return 0;
    }
    if (!pass_->declares(image.resource_, writable) || (writable && !image.writable_)) {
        fail(writable ? "used an image as storage without declaring a write" : "used an image it did not declare");
        return 0;
    }
    if (!services_->bindless_index) {
        fail("has no bindless table");
        return 0;
    }
    return services_->bindless_index(image.resource_);
}

auto GameComputeContext::sampled_index(DeclaredImage image) -> std::uint32_t { return image_index(image, false); }

auto GameComputeContext::storage_index(DeclaredImage image) -> std::uint32_t { return image_index(image, true); }

auto GameComputeContext::address(DeclaredBuffer buffer) -> VkDeviceAddress {
    if (!violation_.empty()) {
        return 0;
    }
    if (!pass_->declares(buffer.resource_, false)) {
        fail("used a buffer it did not declare");
        return 0;
    }
    auto const *physical = pass_->resources != nullptr ? pass_->resources->buffer(buffer.resource_) : nullptr;
    if (physical == nullptr) {
        fail("used a buffer that has no memory");
        return 0;
    }
    return physical->address;
}

auto GameComputeContext::dispatch_threads(std::uint32_t x, std::uint32_t group_x, std::uint32_t y,
                                          std::uint32_t group_y) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (skipped_) {
        return;
    }
    if (layout_ == VK_NULL_HANDLE) {
        fail("dispatched before binding a shader");
        return;
    }
    auto const groups_x = dispatch_group_count(x, group_x, services_->max_group_count[0]);
    auto const groups_y = dispatch_group_count(y, group_y, services_->max_group_count[1]);
    if (!groups_x || !groups_y) {
        fail("dispatched an empty or out of range number of workgroups");
        return;
    }
    vkCmdDispatch(pass_->command_buffer, *groups_x, *groups_y, 1);
}

auto GameDrawContext::fail(std::string_view what) -> void {
    if (violation_.empty()) {
        violation_ = what;
    }
}

auto GameDrawContext::bind(GameGraphicsShader shader) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (services_->renderer == nullptr) {
        fail("has no renderer to bind a shader with");
        return;
    }
    auto const *set = services_->renderer->resolve_pipeline(shader.node);
    if (set == nullptr || !set->valid()) {
        skipped_ = true;
        return;
    }

    set->bind(command_buffer_);
    layout_ = set->layout();
    if (shader.blending) {
        render_pass::set_overlay_blending(command_buffer_, scope_, true);
    }
    services_->renderer->resource_table().bind(command_buffer_, pass_->frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                               layout_);
}

auto GameDrawContext::push_bytes(void const *data, std::size_t size) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (skipped_) {
        return;
    }
    if (layout_ == VK_NULL_HANDLE) {
        fail("pushed constants before binding a shader");
        return;
    }
    vkCmdPushConstants(command_buffer_, layout_, VK_SHADER_STAGE_ALL, 0, static_cast<std::uint32_t>(size), data);
}

auto GameDrawContext::address(DeclaredBuffer buffer) -> VkDeviceAddress {
    if (!violation_.empty()) {
        return 0;
    }
    // Declared by this draw, and by the forward pass that records it, which is what orders the read.
    if (!std::ranges::contains(reads_, buffer.resource_) || !pass_->declares(buffer.resource_, false)) {
        fail("used a buffer it did not declare");
        return 0;
    }
    auto const *physical = pass_->resources != nullptr ? pass_->resources->buffer(buffer.resource_) : nullptr;
    if (physical == nullptr) {
        fail("used a buffer that has no memory");
        return 0;
    }
    return physical->address;
}

auto GameDrawContext::draw(std::uint32_t vertex_count, std::uint32_t instance_count) -> void {
    if (!violation_.empty()) {
        return;
    }
    if (skipped_) {
        return;
    }
    if (layout_ == VK_NULL_HANDLE) {
        fail("drew before binding a shader");
        return;
    }
    if (vertex_count == 0 || instance_count == 0) {
        return;
    }
    vkCmdDraw(command_buffer_, vertex_count, instance_count, 0, 0);
}

auto GameComputeBuilder::sample(EngineImage image) -> DeclaredImage {
    auto const id = pass_->read(image.id_, frame_graph::Use::sampled, compute_stage);
    return DeclaredImage{id.index, false};
}

auto GameComputeBuilder::sample(GameImage image) -> DeclaredImage {
    auto const id = pass_->read(image.id_, frame_graph::Use::sampled, compute_stage);
    return DeclaredImage{id.index, false};
}

auto GameComputeBuilder::touch(GameBuffer buffer, bool reads) -> void {
    auto const resource = buffer.id_.index;
    if (reads) {
        graph_->check_initialised(resource);
    }
    if (auto const *record = graph_->find_buffer(resource); record != nullptr && record->persistent) {
        pass_->queue(frame_graph::QueueAffinity::graphics);
        graphics_only_ = true;
    }
}

auto GameComputeBuilder::read(GameBuffer buffer) -> DeclaredBuffer {
    touch(buffer, true);
    auto const id = pass_->read(buffer.id_, frame_graph::Use::shader_read, compute_stage);
    return DeclaredBuffer{id.index, false};
}

auto GameComputeBuilder::write(GameImage &image) -> DeclaredImage {
    image.id_ = pass_->write(image.id_, frame_graph::Use::storage_write, compute_stage);
    return DeclaredImage{image.id_.index, true};
}

auto GameComputeBuilder::write(GameBuffer &buffer) -> DeclaredBuffer {
    touch(buffer, false);
    buffer.id_ = pass_->write_discard(buffer.id_, frame_graph::Use::shader_write, compute_stage);
    return DeclaredBuffer{buffer.id_.index, true};
}

auto GameComputeBuilder::read_write(GameBuffer &buffer) -> DeclaredBuffer {
    touch(buffer, true);
    buffer.id_ = pass_->write(buffer.id_, frame_graph::Use::shader_read_write, compute_stage);
    return DeclaredBuffer{buffer.id_.index, true};
}

auto GameComputeBuilder::create_image(GameImageDesc const &desc) -> GameImage {
    auto const extent = graph_->services_.scene_extent;
    auto const id = pass_->create(frame_graph::TransientImageDesc{
            .format = desc.format,
            .extent = {desc.extent.width != 0 ? desc.extent.width : extent.width,
                       desc.extent.height != 0 ? desc.extent.height : extent.height, 1},
            .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                image_descriptor_view_bit(ImageDescriptorView::storage_2d),
            .debug_name = graph_->memory_->intern(desc.name),
    });
    return GameImage{id};
}

auto GameDrawBuilder::read(GameBuffer buffer, frame_graph::ShaderStages stages) -> DeclaredBuffer {
    graph_->check_initialised(buffer.id_.index);
    graph_->scene_overlay_reads(buffer, stages);
    reads_.push_back(buffer.id_.index);
    return DeclaredBuffer{buffer.id_.index, false};
}

auto GameGraph::find_buffer(std::uint32_t resource) const -> BufferRecord const * {
    auto const found = std::ranges::find(buffers_, resource, &BufferRecord::resource);
    return found != buffers_.end() ? &*found : nullptr;
}

auto GameGraph::check_initialised(std::uint32_t resource) -> bool {
    auto const *record = find_buffer(resource);
    // Version 1 is the buffer as it was imported; any later version has been written by a pass.
    if (record == nullptr || !record->undefined || graph_->latest_version(resource) != 1) {
        return true;
    }
    problems_.push_back(std::format("read the buffer '{}' before anything wrote it", record->name));
    return false;
}

auto GameGraph::valid_buffer_desc(std::string_view name, GameBufferDesc const &desc) -> bool {
    if (desc.size == 0 || desc.size % 4U != 0) {
        problems_.push_back(std::format("asked for the buffer '{}' with a size that is not a positive multiple of 4",
                                        name));
        return false;
    }
    return true;
}

auto GameGraph::acquire(std::string_view name, GameBufferDesc const &desc, bool persistent) -> GameBuffer {
    auto const interned = memory_->intern(name);
    auto const imported = services_.acquire_buffer
                                  ? services_.acquire_buffer(GameBufferRequest{interned, desc.size, persistent})
                                  : std::nullopt;
    if (!imported) {
        problems_.push_back(std::format("could not get the buffer '{}'", name));
        return GameBuffer{};
    }
    buffers_.push_back(BufferRecord{.name = interned,
                                    .resource = imported->index,
                                    .size = desc.size,
                                    .persistent = persistent,
                                    .undefined = !persistent && !desc.zero});
    return GameBuffer{*imported};
}

auto GameGraph::persistent_buffer(std::string_view name, GameBufferDesc const &desc) -> GameBuffer {
    if (!valid_buffer_desc(name, desc)) {
        return GameBuffer{};
    }
    auto const known = std::ranges::find_if(
            buffers_, [&](BufferRecord const &record) { return record.persistent && record.name == name; });
    if (known != buffers_.end()) {
        if (known->size != desc.size) {
            problems_.push_back(std::format("asked for the persistent buffer '{}' with two different sizes", name));
            return GameBuffer{};
        }
        return GameBuffer{frame_graph::BufferId{.index = known->resource,
                                                .generation = graph_->latest_version(known->resource)}};
    }
    return acquire(name, desc, true);
}

auto GameGraph::create_buffer(GameBufferDesc const &desc, std::string_view name) -> GameBuffer {
    if (!valid_buffer_desc(name, desc)) {
        return GameBuffer{};
    }
    auto buffer = acquire(name, desc, false);
    if (!buffer.valid() || !desc.zero) {
        return buffer;
    }

    // The pool hands out a buffer with someone else's old contents, so the clear is a pass of its own.
    auto const clear_name = memory_->intern(std::format("{}_clear", name));
    graph_->add_pass(clear_name, frame_graph::PassType::transfer,
                     frame_graph::PassProfile{.name_id = clear_name, .label = clear_name, .color = 0xFF1493},
                     frame_graph::Owner::game, [&](frame_graph::PassBuilder &pass) {
                         buffer.id_ = pass.write_discard(buffer.id_, frame_graph::Use::transfer_write);
                         return frame_graph::RecordFn{[resource = buffer.id_.index](frame_graph::PassContext &context) {
                             auto const *physical =
                                     context.resources != nullptr ? context.resources->buffer(resource) : nullptr;
                             if (physical == nullptr || !context.declares(resource, true)) {
                                 return;
                             }
                             vkCmdFillBuffer(context.command_buffer, physical->buffer, 0, VK_WHOLE_SIZE, 0);
                         }};
                     });
    return buffer;
}

auto GameGraph::add_forward_read(frame_graph::BufferId buffer, frame_graph::ShaderStages stages) -> bool {
    if (slot_ > GameSlot::after_depth) {
        problems_.push_back("asked the forward pass to read a buffer after it was declared");
        return false;
    }
    if (!graph_->is_current(buffer) || graph_->owner_of(buffer.index) != frame_graph::Owner::game) {
        problems_.push_back("asked the forward pass to read a buffer that is not a current game buffer");
        return false;
    }
    stages &= scene_draw_stages;
    if (stages == 0) {
        problems_.push_back("asked the forward pass to read a buffer from no graphics stage");
        return false;
    }

    auto const known = std::ranges::find_if(
            forward_reads_, [&](ForwardRead const &read) { return read.buffer.index == buffer.index; });
    if (known != forward_reads_.end()) {
        known->stages |= stages;
    } else {
        forward_reads_.push_back(ForwardRead{.buffer = buffer, .stages = stages});
    }
    return true;
}

auto GameGraph::scene_overlay_reads(GameBuffer buffer, frame_graph::ShaderStages stages) -> void {
    add_forward_read(buffer.id_, stages);
}

auto GameGraph::replace_scene_colour(GameImage image) -> void {
    if (slot_ < GameSlot::after_lighting) {
        problems_.push_back("replaced the scene colour before it exists");
        return;
    }
    if (!graph_->is_current(image.id_) || graph_->owner_of(image.id_.index) != frame_graph::Owner::game) {
        problems_.push_back("replaced the scene colour with an image that is not a current game image");
        return;
    }
    scene_colour_ = image.id_;
}

auto GameGraph::begin_unit() const -> Unit {
    return Unit{
            .mark = graph_->checkpoint(),
            .forward_reads = forward_reads_,
            .scene_colour = scene_colour_,
            .draws = scene_draws_.size(),
            .buffers = buffers_.size(),
            .problems = problems_.size(),
    };
}

auto GameGraph::end_unit(Unit const &unit, bool keep_problems) -> std::optional<std::string> {
    auto const &errors = graph_->declaration_errors();
    if (errors.size() <= unit.mark.errors && problems_.size() <= unit.problems) {
        return std::nullopt;
    }

    // Nothing the game declared touched an engine resource, so dropping it leaves the engine's graph as it was.
    auto message = std::string{};
    for (auto const &rejected: graph_->rollback(unit.mark)) {
        message += std::format(" {};", rejected);
    }
    for (auto index = unit.problems; index < problems_.size(); ++index) {
        message += std::format(" game {};", problems_[index]);
    }
    if (!keep_problems) {
        problems_.resize(unit.problems);
    }
    forward_reads_ = unit.forward_reads;
    scene_colour_ = unit.scene_colour;
    scene_draws_.resize(unit.draws);
    buffers_.resize(unit.buffers);
    return message;
}

auto GameGraph::run_slot(GameSlot slot, std::optional<frame_graph::ImageId> depth,
                         std::optional<frame_graph::ImageId> hdr, std::function<void(GameGraph &)> const &hook)
        -> void {
    slot_ = slot;
    scene_depth_ = depth ? std::optional{EngineImage{*depth}} : std::nullopt;
    scene_hdr_ = hdr ? std::optional{EngineImage{*hdr}} : std::nullopt;

    auto const unit = begin_unit();
    hook(*this);

    auto const rejected = end_unit(unit, true);
    if (!rejected) {
        return;
    }

    auto const message = "Game frame graph slot rejected, its passes are dropped for this frame:" + *rejected;
    ++rolled_back_slots_;
    if (memory_->report_once(message)) {
        error("{}", message);
    }
}

auto GameGraph::isolated(std::string_view what, std::function<void()> const &declare) -> std::optional<std::string> {
    auto const unit = begin_unit();
    declare();

    auto rejected = end_unit(unit, false);
    if (!rejected) {
        return std::nullopt;
    }

    auto const message = std::format("Game frame graph '{}' rejected, it is dropped for this frame:{}", what, *rejected);
    ++rejected_units_;
    if (memory_->report_once(message)) {
        error("{}", message);
    }
    return message;
}

auto GameGraph::scene_colour_source() const -> std::optional<ImageRead> {
    if (scene_colour_) {
        return ImageRead{GameImage{*scene_colour_}};
    }
    if (scene_hdr_) {
        return ImageRead{*scene_hdr_};
    }
    return std::nullopt;
}

auto GameGraph::dynamic_push_size(DynamicParams const &params) noexcept -> std::size_t {
    auto offset = std::size_t{0};
    for (auto const &member: params.members) {
        auto const [align, size] = std::visit(
                [](auto const &param) -> std::pair<std::size_t, std::size_t> {
                    using T = std::remove_cvref_t<decltype(param)>;
                    if constexpr (std::is_same_v<T, PushBytes>) {
                        return {param.align, param.size};
                    } else {
                        return {alignof(typename T::push_type), sizeof(typename T::push_type)};
                    }
                },
                member);
        offset = ((offset + align - 1U) & ~(align - 1U)) + size;
    }
    return offset;
}

auto GameGraph::add_compute(std::string_view name, DynamicParams &params, GameComputeShader shader, Threads threads,
                            GameComputeOptions const &options) -> void {
    if (dynamic_push_size(params) > max_game_push_bytes) {
        problems_.push_back(std::format("pass '{}' has more push constants than fit in {} bytes", name,
                                        max_game_push_bytes));
        return;
    }

    add_compute_pass(
            name, GamePassProfile{.label = options.label.empty() ? name : options.label, .color = options.color},
            [&](GameComputeBuilder &pass) {
                pass.queue(options.queue);
                if (options.side_effect) {
                    pass.side_effect();
                }
                for (auto &member: params.members) {
                    std::visit([&](auto &param) { declare_param(pass, param); }, member);
                }

                return [shader, threads, snapshot = params](GameComputeContext &context) {
                    auto packed = PackedPush{};
                    auto offset = std::size_t{0};
                    for (auto const &member: snapshot.members) {
                        std::visit(
                                [&](auto const &param) {
                                    using T = std::remove_cvref_t<decltype(param)>;
                                    auto const append = [&](auto const &value, std::size_t align) {
                                        offset = (offset + align - 1U) & ~(align - 1U);
                                        std::memcpy(packed.bytes.data() + offset, &value, sizeof(value));
                                        offset += sizeof(value);
                                    };
                                    if constexpr (std::is_same_v<T, PushBytes>) {
                                        offset = (offset + param.align - 1U) & ~(std::size_t{param.align} - 1U);
                                        std::memcpy(packed.bytes.data() + offset, param.bytes.data(), param.size);
                                        offset += param.size;
                                    } else {
                                        append(resolve(context, param), alignof(typename T::push_type));
                                    }
                                },
                                member);
                    }
                    packed.size = offset;

                    context.begin(shader, threads);
                    context.push_layout(packed.bytes.data(), packed.size);
                    context.dispatch();
                };
            });
}

auto GameGraph::report_violation(std::string_view pass, std::string_view violation) -> void {
    if (violation.empty()) {
        return;
    }
    auto message = std::format("Game pass '{}' {}", pass, violation);
    problems_.push_back(message);
    if (memory_->report_once(message)) {
        error("{}", message);
    }
}

auto GameGraph::record_scene_draws(frame_graph::PassContext const &pass, OverlayRecordContext const &overlay)
        -> void {
    for (auto &draw: scene_draws_) {
        render_pass::set_overlay_baseline_state(overlay.command_buffer, OverlayStage::scene, overlay.scope);

        auto context = GameDrawContext{pass, services_, overlay, draw.reads};
        draw.record(context);
        report_violation(draw.name, context.violation());
    }
}
