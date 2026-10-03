#include "rendering/frame_graph/frame_graph.hxx"

#include <algorithm>

namespace frame_graph {

    auto FrameGraph::reset() -> void {
        desc_.resources.clear();
        desc_.passes.clear();
        desc_.producers.clear();
        records_.clear();
        latest_.clear();
        written_.clear();
        errors_.clear();
    }

    auto FrameGraph::add_resource(ResourceDesc resource, bool produced) -> std::uint32_t {
        auto const index = static_cast<std::uint32_t>(desc_.resources.size());
        desc_.resources.push_back(std::move(resource));
        // Version 1 exists from the start; imports and tokens have it written, transients do not.
        desc_.producers.push_back({-1, -1});
        latest_.push_back(1);
        written_.push_back(produced);
        return index;
    }

    auto FrameGraph::latest_version(std::uint32_t resource) const -> std::uint32_t { return latest_[resource]; }

    auto FrameGraph::record_error(FrameGraphErrorType type, PassDesc const &pass, std::uint32_t resource) -> void {
        errors_.push_back(FrameGraphError{
                .type = type,
                .pass = pass.name,
                .resource = resource < desc_.resources.size() ? desc_.resources[resource].name : std::string{},
        });
    }

    auto FrameGraph::import_image(ImportedDesc const &desc) -> ImageId {
        auto const index = add_resource(
                ResourceDesc{
                        .name = std::string{desc.debug_name},
                        .kind = ResourceKind::image,
                        .imported = true,
                        .swapchain = desc.swapchain,
                        .read_only = desc.read_only,
                        .sharing = desc.sharing,
                        .entry = desc.entry,
                        .exit = desc.exit,
                        .image = desc.image,
                },
                true);
        if (desc.exit.queue != LogicalQueue::graphics) {
            errors_.push_back({FrameGraphErrorType::import_exit_not_on_graphics, {}, std::string{desc.debug_name}});
        }
        return ImageId{.index = index, .generation = 1};
    }

    auto FrameGraph::import_buffer(ImportedDesc const &desc) -> BufferId {
        auto const index = add_resource(
                ResourceDesc{
                        .name = std::string{desc.debug_name},
                        .kind = ResourceKind::buffer,
                        .imported = true,
                        .read_only = desc.read_only,
                        .sharing = desc.sharing,
                        .entry = desc.entry,
                        .exit = desc.exit,
                        .buffer = desc.buffer,
                },
                true);
        if (desc.exit.queue != LogicalQueue::graphics) {
            errors_.push_back({FrameGraphErrorType::import_exit_not_on_graphics, {}, std::string{desc.debug_name}});
        }
        return BufferId{.index = index, .generation = 1};
    }

    auto FrameGraph::import_token(std::string_view name, ResourceState entry, ResourceState exit) -> BufferId {
        auto const index = add_resource(
                ResourceDesc{
                        .name = std::string{name},
                        .kind = ResourceKind::token,
                        .imported = true,
                        .entry = entry,
                        .exit = exit,
                },
                true);
        return BufferId{.index = index, .generation = 1};
    }

    auto FrameGraph::begin_pass(std::string_view name, PassType type, PassProfile profile) -> PassDesc & {
        auto const duplicate = std::ranges::any_of(desc_.passes, [&](PassDesc const &p) { return p.name == name; });
        auto &pass = desc_.passes.emplace_back();
        pass.name = std::string{name};
        pass.type = type;
        pass.profile = profile;
        if (duplicate) {
            errors_.push_back({FrameGraphErrorType::duplicate_pass_name, pass.name, {}});
        }
        return pass;
    }

    auto FrameGraph::validate_access(PassDesc const &pass, std::uint32_t resource, std::uint32_t version) -> bool {
        if (resource >= desc_.resources.size() || version == 0) {
            record_error(FrameGraphErrorType::invalid_handle, pass, resource);
            return false;
        }
        if (version != latest_[resource]) {
            record_error(FrameGraphErrorType::stale_version, pass, resource);
            return false;
        }
        return true;
    }

    auto FrameGraph::produce(std::uint32_t resource) -> std::uint32_t {
        auto const next = latest_[resource] + 1;
        latest_[resource] = next;
        written_[resource] = true;
        auto const pass_index = static_cast<std::int64_t>(desc_.passes.size() - 1);
        auto &chain = desc_.producers[resource];
        chain.resize(next + 1, -1);
        chain[next] = pass_index;
        return next;
    }

    auto PassBuilder::queue(QueueAffinity affinity) -> void { pass_->affinity = affinity; }

    auto PassBuilder::side_effect() -> void { pass_->side_effect = true; }

    auto PassBuilder::legacy() -> void { pass_->legacy = true; }

    auto PassBuilder::pinned() -> void { pass_->pinned = true; }

    auto PassBuilder::access(std::uint32_t resource, std::uint32_t version, Use use, ShaderStages stages, bool discard)
            -> bool {
        auto &graph = *graph_;
        auto const &pass = *pass_;
        if (!graph.validate_access(pass, resource, version)) {
            return false;
        }

        auto const &desc = graph.desc_.resources[resource];
        auto const info = use_info(use, stages);
        auto const kind_matches = info.is_token   ? desc.kind == ResourceKind::token
                                  : info.is_image ? desc.kind == ResourceKind::image
                                                  : desc.kind == ResourceKind::buffer;
        if (!kind_matches) {
            graph.record_error(FrameGraphErrorType::wrong_resource_kind, pass, resource);
            return false;
        }
        if (info.needs_shader_stages && stages == 0) {
            graph.record_error(FrameGraphErrorType::missing_shader_stages, pass, resource);
            return false;
        }
        if (std::ranges::any_of(pass.accesses, [&](AccessDesc const &a) { return a.resource == resource; })) {
            graph.record_error(FrameGraphErrorType::conflicting_use, pass, resource);
            return false;
        }
        if (info.writes && desc.read_only) {
            graph.record_error(FrameGraphErrorType::write_to_read_only_import, pass, resource);
            return false;
        }

        auto const written = graph.written_[resource];
        if (!written && !info.writes) {
            graph.record_error(FrameGraphErrorType::read_before_write, pass, resource);
            return false;
        }

        pass_->accesses.push_back(AccessDesc{
                .resource = resource,
                .version = version,
                .use = use,
                .stages = stages,
                // A transient's first write has no earlier contents to preserve.
                .discard = discard || !written,
                .produces = info.writes,
        });
        if (info.writes) {
            graph.produce(resource);
        }
        return true;
    }

    auto PassBuilder::read(ImageId image, Use use, ShaderStages stages) -> ImageId {
        if (use_info(use, stages).writes) {
            graph_->record_error(FrameGraphErrorType::conflicting_use, *pass_, image.index);
            return image;
        }
        access(image.index, image.generation, use, stages, false);
        return image;
    }

    auto PassBuilder::write(ImageId image, Use use, ShaderStages stages) -> ImageId {
        auto const discard = discards_contents(use, LoadOp::load);
        access(image.index, image.generation, use, stages, discard);
        return ImageId{.index = image.index, .generation = graph_->latest_version(image.index)};
    }

    auto PassBuilder::write(ImageId image, Use use, ShaderStages stages, ExitUse exit) -> ImageId {
        auto const written = write(image, use, stages);
        // The access recorded above is the pass's last, unless validation rejected it.
        if (!pass_->accesses.empty() && pass_->accesses.back().resource == image.index) {
            pass_->accesses.back().exit_use = exit.use;
        }
        return written;
    }

    auto PassBuilder::read(BufferId buffer, Use use, ShaderStages stages) -> BufferId {
        if (use_info(use, stages).writes) {
            graph_->record_error(FrameGraphErrorType::conflicting_use, *pass_, buffer.index);
            return buffer;
        }
        access(buffer.index, buffer.generation, use, stages, false);
        return buffer;
    }

    auto PassBuilder::write(BufferId buffer, Use use, ShaderStages stages) -> BufferId {
        access(buffer.index, buffer.generation, use, stages, false);
        return BufferId{.index = buffer.index, .generation = graph_->latest_version(buffer.index)};
    }

    auto PassBuilder::write_discard(BufferId buffer, Use use, ShaderStages stages) -> BufferId {
        access(buffer.index, buffer.generation, use, stages, true);
        return BufferId{.index = buffer.index, .generation = graph_->latest_version(buffer.index)};
    }

    auto PassBuilder::rendering() -> RenderingDesc & {
        if (!pass_->rendering) {
            pass_->rendering.emplace();
        }
        return *pass_->rendering;
    }

    auto PassBuilder::color(ImageId image, LoadOp load, StoreOp store, VkClearValue clear) -> ImageId {
        if (access(image.index, image.generation, Use::color_attachment, 0,
                   discards_contents(Use::color_attachment, load))) {
            rendering().colors.push_back(
                    AttachmentDesc{.resource = image.index, .load = load, .store = store, .clear = clear});
        }
        return ImageId{.index = image.index, .generation = graph_->latest_version(image.index)};
    }

    auto PassBuilder::write_depth(ImageId image, LoadOp load, StoreOp store, VkClearValue clear) -> ImageId {
        if (access(image.index, image.generation, Use::depth_attachment, 0,
                   discards_contents(Use::depth_attachment, load))) {
            rendering().depth = AttachmentDesc{.resource = image.index, .load = load, .store = store, .clear = clear};
        }
        return ImageId{.index = image.index, .generation = graph_->latest_version(image.index)};
    }

    auto PassBuilder::resolve(ImageId attachment, ImageId target, VkResolveModeFlagBits mode) -> ImageId {
        auto *attached = static_cast<AttachmentDesc *>(nullptr);
        auto is_depth = false;
        if (pass_->rendering) {
            auto &desc = *pass_->rendering;
            if (desc.depth && desc.depth->resource == attachment.index) {
                attached = &*desc.depth;
                is_depth = true;
            } else if (auto const found = std::ranges::find_if(
                               desc.colors, [&](AttachmentDesc const &c) { return c.resource == attachment.index; });
                       found != desc.colors.end()) {
                attached = &*found;
            }
        }
        if (attached == nullptr) {
            // Not an attachment of this pass, or one that already has its resolve.
            graph_->record_error(FrameGraphErrorType::conflicting_use, *pass_, attachment.index);
            return target;
        }

        auto const use = is_depth ? Use::depth_resolve : Use::color_resolve;
        if (access(target.index, target.generation, use, 0, true)) {
            attached->resolve = AttachmentResolve{.resource = target.index, .mode = mode};
        }
        return ImageId{.index = target.index, .generation = graph_->latest_version(target.index)};
    }

    auto PassBuilder::render_area(VkRect2D area) -> void { rendering().render_area = area; }

    auto PassBuilder::view_mask(std::uint32_t mask) -> void { rendering().view_mask = mask; }

    auto PassBuilder::create(TransientImageDesc const &desc) -> ImageId {
        auto const index = graph_->add_resource(
                ResourceDesc{
                        .name = std::string{desc.debug_name},
                        .kind = ResourceKind::image,
                        .transient_image = desc,
                },
                false);
        return ImageId{.index = index, .generation = 1};
    }

} // namespace frame_graph
