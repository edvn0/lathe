#include "rendering/frame_graph/vk_translate.hxx"

namespace frame_graph {
    namespace {

        auto load_op(LoadOp load) noexcept -> VkAttachmentLoadOp {
            switch (load) {
                case LoadOp::load:
                    return VK_ATTACHMENT_LOAD_OP_LOAD;
                case LoadOp::clear:
                    return VK_ATTACHMENT_LOAD_OP_CLEAR;
                case LoadOp::dont_care:
                    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        }

        auto store_op(StoreOp store) noexcept -> VkAttachmentStoreOp {
            switch (store) {
                case StoreOp::store:
                    return VK_ATTACHMENT_STORE_OP_STORE;
                case StoreOp::dont_care:
                    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
            return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }

        auto attachment_info(AttachmentDesc const &attachment, VkImageLayout layout, PhysicalResources const &resources)
                -> std::expected<VkRenderingAttachmentInfo, TranslateFailure> {
            auto const *image = resources.image(attachment.resource);
            if (image == nullptr) {
                return std::unexpected(
                        TranslateFailure{.kind = TranslateFailureKind::missing_image, .resource = attachment.resource});
            }

            auto info = VkRenderingAttachmentInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .pNext = nullptr,
                    .imageView = image->view,
                    .imageLayout = layout,
                    .resolveMode = VK_RESOLVE_MODE_NONE,
                    .resolveImageView = VK_NULL_HANDLE,
                    .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp = load_op(attachment.load),
                    .storeOp = store_op(attachment.store),
                    .clearValue = attachment.clear,
            };

            if (attachment.resolve) {
                auto const *target = resources.image(attachment.resolve->resource);
                if (target == nullptr) {
                    return std::unexpected(TranslateFailure{.kind = TranslateFailureKind::missing_image,
                                                            .resource = attachment.resolve->resource});
                }
                info.resolveMode = attachment.resolve->mode;
                info.resolveImageView = target->view;
                info.resolveImageLayout = layout;
            }
            return info;
        }

    }

    auto physical_resources_of(GraphDesc const &graph) -> PhysicalResources {
        auto resources = PhysicalResources{};
        resources.images.resize(graph.resources.size());
        resources.buffers.resize(graph.resources.size());
        for (auto index = std::size_t{0}; index < graph.resources.size(); ++index) {
            auto const &resource = graph.resources[index];
            if (resource.kind == ResourceKind::image) {
                resources.images[index] = resource.image;
            } else if (resource.kind == ResourceKind::buffer) {
                resources.buffers[index] = resource.buffer;
            }
        }
        return resources;
    }

    auto image_aspect(VkFormat format) noexcept -> VkImageAspectFlags {
        switch (format) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_X8_D24_UNORM_PACK32:
            case VK_FORMAT_D32_SFLOAT:
                return VK_IMAGE_ASPECT_DEPTH_BIT;
            case VK_FORMAT_D16_UNORM_S8_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            case VK_FORMAT_S8_UINT:
                return VK_IMAGE_ASPECT_STENCIL_BIT;
            default:
                return VK_IMAGE_ASPECT_COLOR_BIT;
        }
    }

    auto DependencyStorage::info() const noexcept -> VkDependencyInfo {
        return VkDependencyInfo{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = static_cast<std::uint32_t>(memory.size()),
                .pMemoryBarriers = memory.data(),
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(buffers.size()),
                .pBufferMemoryBarriers = buffers.data(),
                .imageMemoryBarrierCount = static_cast<std::uint32_t>(images.size()),
                .pImageMemoryBarriers = images.data(),
        };
    }

    auto translate(BarrierSet const &barriers, PhysicalResources const &resources)
            -> std::expected<DependencyStorage, TranslateFailure> {
        auto storage = DependencyStorage{};

        for (auto const &barrier: barriers.memory) {
            storage.memory.push_back(VkMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = barrier.src_stages,
                    .srcAccessMask = barrier.src_access,
                    .dstStageMask = barrier.dst_stages,
                    .dstAccessMask = barrier.dst_access,
            });
        }

        for (auto const &barrier: barriers.images) {
            auto const *image = resources.image(barrier.resource);
            if (image == nullptr) {
                return std::unexpected(
                        TranslateFailure{.kind = TranslateFailureKind::missing_image, .resource = barrier.resource});
            }
            storage.images.push_back(VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = barrier.src_stages,
                    .srcAccessMask = barrier.src_access,
                    .dstStageMask = barrier.dst_stages,
                    .dstAccessMask = barrier.dst_access,
                    .oldLayout = barrier.old_layout,
                    .newLayout = barrier.new_layout,
                    .srcQueueFamilyIndex = barrier.src_family,
                    .dstQueueFamilyIndex = barrier.dst_family,
                    .image = image->image,
                    .subresourceRange =
                            VkImageSubresourceRange{
                                    .aspectMask = image_aspect(image->format),
                                    .baseMipLevel = 0,
                                    .levelCount = image->mip_levels,
                                    .baseArrayLayer = 0,
                                    .layerCount = image->array_layers,
                            },
            });
        }

        for (auto const &barrier: barriers.buffers) {
            auto const *buffer = resources.buffer(barrier.resource);
            if (buffer == nullptr) {
                return std::unexpected(
                        TranslateFailure{.kind = TranslateFailureKind::missing_buffer, .resource = barrier.resource});
            }
            storage.buffers.push_back(VkBufferMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = barrier.src_stages,
                    .srcAccessMask = barrier.src_access,
                    .dstStageMask = barrier.dst_stages,
                    .dstAccessMask = barrier.dst_access,
                    .srcQueueFamilyIndex = barrier.src_family,
                    .dstQueueFamilyIndex = barrier.dst_family,
                    .buffer = buffer->buffer,
                    .offset = 0,
                    .size = VK_WHOLE_SIZE,
            });
        }

        return storage;
    }

    auto RenderingStorage::info() const noexcept -> VkRenderingInfo {
        return VkRenderingInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = render_area,
                .layerCount = layer_count,
                .viewMask = view_mask,
                .colorAttachmentCount = static_cast<std::uint32_t>(colors.size()),
                .pColorAttachments = colors.data(),
                .pDepthAttachment = has_depth ? &depth : nullptr,
                .pStencilAttachment = nullptr,
        };
    }

    auto translate(RenderingDesc const &rendering, PhysicalResources const &resources)
            -> std::expected<RenderingStorage, TranslateFailure> {
        auto storage = RenderingStorage{
                .render_area = rendering.render_area,
                .layer_count = rendering.layer_count,
                .view_mask = rendering.view_mask,
        };

        for (auto const &color: rendering.colors) {
            auto info = attachment_info(color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, resources);
            if (!info) {
                return std::unexpected(info.error());
            }
            storage.colors.push_back(*info);
        }

        if (rendering.depth) {
            auto info = attachment_info(*rendering.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, resources);
            if (!info) {
                return std::unexpected(info.error());
            }
            storage.depth = *info;
            storage.has_depth = true;
        }

        return storage;
    }

}
