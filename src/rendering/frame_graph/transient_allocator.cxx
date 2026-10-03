#include "rendering/frame_graph/transient_allocator.hxx"

#include <format>

#include "gpu/context.hxx"

namespace frame_graph {
    namespace {

        auto mix(std::uint64_t &state, std::uint64_t value) -> void {
            state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
        }

        // Everything that decides what the slot's images are: the compiled plan (which fixes lifetimes) and each
        // transient's description.
        auto key_of(GraphDesc const &graph, CompiledGraph const &compiled, bool alias) -> std::uint64_t {
            auto key = compiled.hash;
            mix(key, alias ? 1U : 0U);
            for (auto const &resource: graph.resources) {
                if (!resource.transient_image) {
                    continue;
                }
                auto const &desc = *resource.transient_image;
                mix(key, static_cast<std::uint64_t>(desc.format));
                mix(key, (std::uint64_t{desc.extent.width} << 32U) | desc.extent.height);
                mix(key, desc.extent.depth);
                mix(key, (std::uint64_t{desc.mip_levels} << 32U) | desc.array_layers);
                mix(key, static_cast<std::uint64_t>(desc.samples));
                mix(key, (std::uint64_t{desc.descriptor_views} << 2U) | (desc.mip_layer_views ? 2U : 0U) |
                                 (desc.mip_slots ? 1U : 0U));
            }
            return key;
        }

        auto create_info_of(TransientImageDesc const &desc, VkImageUsageFlags usage) -> ImageCreateInfo {
            // Bindless views need the matching usage whether or not a pass declared the use.
            if ((desc.descriptor_views & image_descriptor_view_bit(ImageDescriptorView::sampled_2d)) != 0) {
                usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            }
            if ((desc.descriptor_views & image_descriptor_view_bit(ImageDescriptorView::storage_2d)) != 0) {
                usage |= VK_IMAGE_USAGE_STORAGE_BIT;
            }
            return ImageCreateInfo{
                    .extent = desc.extent,
                    .format = desc.format,
                    .usage = usage,
                    .image_type = VK_IMAGE_TYPE_2D,
                    .view_type = VK_IMAGE_VIEW_TYPE_2D,
                    .descriptor_views = desc.descriptor_views,
                    .samples = desc.samples,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .mip_levels = desc.mip_levels,
                    .array_layers = desc.array_layers,
                    .create_mip_layer_views = desc.mip_layer_views,
                    .debug_name = desc.debug_name,
            };
        }

        auto empty_plan() -> TransientPlan const & {
            static TransientPlan const plan{};
            return plan;
        }

    } // namespace

    auto TransientAllocator::initialize(VulkanContext &context, ImageStorage &images, std::uint32_t slot_count)
            -> void {
        release_all();
        context_ = &context;
        images_ = &images;
        slots_.clear();
        slots_.resize(slot_count);
    }

    auto TransientAllocator::release(std::uint32_t slot) -> void {
        if (slot >= slots_.size()) {
            return;
        }

        auto &state = slots_[slot];
        // Images first: they are bound into the blocks.
        state.entries.clear();
        for (auto const block: state.blocks) {
            if (context_ != nullptr && context_->allocator != VK_NULL_HANDLE) {
                vmaFreeMemory(context_->allocator, block);
            }
        }
        state.blocks.clear();
        state.plan = {};
        state.key = 0;
        state.valid = false;
    }

    auto TransientAllocator::release_all() -> void {
        for (auto slot = std::uint32_t{0}; slot < slots_.size(); ++slot) {
            release(slot);
        }
    }

    auto TransientAllocator::prepare(std::uint32_t slot, GraphDesc const &graph, CompiledGraph const &compiled,
                                     bool alias) -> std::expected<void, TransientAllocationError> {
        if (context_ == nullptr || slot >= slots_.size()) {
            return std::unexpected(TransientAllocationError{"transient allocator not initialised"});
        }

        auto &state = slots_[slot];
        auto const key = key_of(graph, compiled, alias);
        if (state.valid && state.key == key) {
            return {};
        }

        release(slot);

        // What each live transient needs.
        auto requirements = std::vector<MemoryRequirement>(graph.resources.size());
        auto infos = std::vector<ImageCreateInfo>(graph.resources.size());
        for (auto resource = std::uint32_t{0}; resource < graph.resources.size(); ++resource) {
            auto const &desc = graph.resources[resource].transient_image;
            if (!desc) {
                continue;
            }
            auto const usage = transient_usage(graph, compiled, resource);
            if (usage == 0) {
                continue; // culled away
            }
            infos[resource] = create_info_of(*desc, usage);
            auto const memory = Image::memory_requirements(*context_, infos[resource]);
            requirements[resource] = MemoryRequirement{
                    .size = memory.size,
                    .alignment = memory.alignment,
                    .memory_type_bits = memory.memoryTypeBits,
            };
        }

        state.plan = plan_transients(graph, compiled, requirements, alias);

        for (auto const &block: state.plan.blocks) {
            VkMemoryRequirements const memory{
                    .size = block.size,
                    .alignment = block.alignment,
                    .memoryTypeBits = block.memory_type_bits,
            };
            VmaAllocationCreateInfo const allocation_info{
                    .flags = VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    .requiredFlags = 0,
                    .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    .memoryTypeBits = block.memory_type_bits,
                    .pool = VK_NULL_HANDLE,
                    .pUserData = nullptr,
                    .priority = 1.0F,
            };
            VmaAllocation allocation = VK_NULL_HANDLE;
            auto const result = vmaAllocateMemory(context_->allocator, &memory, &allocation_info, &allocation, nullptr);
            if (result != VK_SUCCESS) {
                release(slot);
                return std::unexpected(TransientAllocationError{
                        std::format("vmaAllocateMemory failed for a {} byte transient block (VkResult {})", block.size,
                                    static_cast<int>(result))});
            }
            state.blocks.push_back(allocation);
        }

        state.entries.clear();
        state.entries.resize(graph.resources.size());
        for (auto const &placement: state.plan.placements) {
            auto info = infos[placement.resource];
            info.alias = ImageAliasing{.allocation = state.blocks[placement.block], .offset = placement.offset};

            auto image = create_held_image(*images_, info);
            if (!image) {
                release(slot);
                return std::unexpected(TransientAllocationError{
                        std::format("could not create transient image '{}'", info.debug_name)});
            }

            auto &entry = state.entries[placement.resource];
            entry.image = std::move(*image);

            auto const &desc = *graph.resources[placement.resource].transient_image;
            if (desc.mip_slots) {
                for (auto mip = std::uint32_t{0}; mip < desc.mip_levels; ++mip) {
                    auto const view = entry.image.get()->mip_layer_view(mip, 0);
                    auto mip_slot = register_held_view(*images_, ImageViewRegistration{
                                                                         .sampled_2d = view,
                                                                         .storage_2d = view,
                                                                 });
                    if (!mip_slot) {
                        release(slot);
                        return std::unexpected(TransientAllocationError{std::format(
                                "could not register mip {} of transient image '{}'", mip, info.debug_name)});
                    }
                    entry.mip_slots.push_back(std::move(*mip_slot));
                }
            }
        }

        state.key = key;
        state.valid = true;
        return {};
    }

    auto TransientAllocator::image(std::uint32_t slot, std::uint32_t resource) const noexcept -> Image const * {
        if (slot >= slots_.size() || resource >= slots_[slot].entries.size()) {
            return nullptr;
        }
        return slots_[slot].entries[resource].image.get();
    }

    auto TransientAllocator::handle(std::uint32_t slot, std::uint32_t resource) const noexcept -> ImageHandle {
        if (slot >= slots_.size() || resource >= slots_[slot].entries.size()) {
            return {};
        }
        return slots_[slot].entries[resource].image.handle();
    }

    auto TransientAllocator::mip_handle(std::uint32_t slot, std::uint32_t resource, std::uint32_t mip) const noexcept
            -> ImageHandle {
        if (slot >= slots_.size() || resource >= slots_[slot].entries.size() ||
            mip >= slots_[slot].entries[resource].mip_slots.size()) {
            return {};
        }
        return slots_[slot].entries[resource].mip_slots[mip].handle();
    }

    auto TransientAllocator::plan(std::uint32_t slot) const noexcept -> TransientPlan const & {
        return slot < slots_.size() ? slots_[slot].plan : empty_plan();
    }

    auto TransientAllocator::fill(std::uint32_t slot, PhysicalResources &resources) const -> void {
        if (slot >= slots_.size()) {
            return;
        }
        auto const &entries = slots_[slot].entries;
        if (resources.images.size() < entries.size()) {
            resources.images.resize(entries.size());
        }
        for (auto resource = std::size_t{0}; resource < entries.size(); ++resource) {
            auto const *image = entries[resource].image.get();
            if (image == nullptr) {
                continue;
            }
            resources.images[resource] = PhysicalImage{
                    .image = image->image(),
                    .view = image->view(),
                    .format = image->format(),
                    .extent = image->extent(),
                    .mip_levels = image->mip_levels(),
                    .array_layers = image->array_layers(),
            };
        }
    }

} // namespace frame_graph
