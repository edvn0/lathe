#include "rendering/game_graph.hxx"

#include <algorithm>
#include <format>

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

auto GameComputeBuilder::read(GameBuffer buffer) -> DeclaredBuffer {
    auto const id = pass_->read(buffer.id_, frame_graph::Use::shader_read, compute_stage);
    return DeclaredBuffer{id.index, false};
}

auto GameComputeBuilder::write(GameImage &image) -> DeclaredImage {
    image.id_ = pass_->write(image.id_, frame_graph::Use::storage_write, compute_stage);
    return DeclaredImage{image.id_.index, true};
}

auto GameComputeBuilder::read_write(GameBuffer &buffer) -> DeclaredBuffer {
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
    graph_->scene_overlay_reads(buffer, stages);
    reads_.push_back(buffer.id_.index);
    return DeclaredBuffer{buffer.id_.index, false};
}

auto GameGraph::import(GameBufferHandle buffer) -> GameBuffer {
    auto const known = std::ranges::find_if(imported_, [&](auto const &entry) { return entry.first == buffer.index; });
    if (known != imported_.end()) {
        return GameBuffer{frame_graph::BufferId{.index = known->second,
                                                .generation = graph_->latest_version(known->second)}};
    }

    auto const imported = services_.import_buffer ? services_.import_buffer(buffer) : std::nullopt;
    if (!imported) {
        problems_.push_back("imported a game buffer that does not exist");
        return GameBuffer{};
    }
    imported_.emplace_back(buffer.index, imported->index);
    return GameBuffer{*imported};
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

auto GameGraph::run_slot(GameSlot slot, std::optional<frame_graph::ImageId> depth,
                         std::optional<frame_graph::ImageId> hdr, std::function<void(GameGraph &)> const &hook)
        -> void {
    slot_ = slot;
    scene_depth_ = depth ? std::optional{EngineImage{*depth}} : std::nullopt;
    scene_hdr_ = hdr ? std::optional{EngineImage{*hdr}} : std::nullopt;

    auto const mark = graph_->checkpoint();
    auto const saved_reads = forward_reads_;
    auto const saved_colour = scene_colour_;
    auto const draw_count = scene_draws_.size();
    auto const imported_count = imported_.size();
    auto const problem_count = problems_.size();

    hook(*this);

    auto const &errors = graph_->declaration_errors();
    if (errors.size() <= mark.errors && problems_.size() <= problem_count) {
        return;
    }

    // Nothing the game declared touched an engine resource, so dropping it leaves the engine's graph as it was.
    auto message = std::string{"Game frame graph slot rejected, its passes are dropped for this frame:"};
    for (auto const &rejected: graph_->rollback(mark)) {
        message += std::format(" {};", rejected);
    }
    for (auto index = problem_count; index < problems_.size(); ++index) {
        message += std::format(" game {};", problems_[index]);
    }
    forward_reads_ = saved_reads;
    scene_colour_ = saved_colour;
    scene_draws_.resize(draw_count);
    imported_.resize(imported_count);
    ++rolled_back_slots_;
    if (memory_->report_once(message)) {
        error("{}", message);
    }
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
