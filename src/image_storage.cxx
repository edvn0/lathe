#include "image_storage.hxx"

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include "context.hxx"

namespace {
    constexpr std::array<std::array<std::uint8_t, 4>, default_image_count> default_pixels{{
            // 0: white
            {255, 255, 255, 255},

            // 1: black
            {0, 0, 0, 255},

            // 2: flat tangent-space normal
            {128, 128, 255, 255},

            // 3: glTF metallic-roughness
            // R unused, G roughness=1, B metallic=0
            {0, 255, 0, 255},

            // 4: full occlusion contribution
            {255, 255, 255, 255},

            // 5: no emissive contribution
            {0, 0, 0, 255},
    }};

    constexpr std::array<std::string_view, default_image_count> default_names{
            "white", "black", "flat_normal", "metallic_roughness", "occlusion", "emissive",
    };

    auto make_error(ImageStorageErrorType type) noexcept -> ImageStorageError {
        return ImageStorageError{
                .type = type,
        };
    }

    auto make_image_error(ImageError error) noexcept -> ImageStorageError {
        return ImageStorageError{
                .type = ImageStorageErrorType::image_error,
                .cause = ErrorCause{Boxed<ImageError>{std::move(error)}},
        };
    }

    auto make_device_error(DeviceError error) noexcept -> ImageStorageError {
        return ImageStorageError{
                .type = ImageStorageErrorType::device_error,
                .cause = ErrorCause{Boxed<DeviceError>{std::move(error)}},
        };
    }
} // namespace

ImageStorage::~ImageStorage() { destroy(); }

ImageStorage::ImageStorage(ImageStorage &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), slots_(std::move(other.slots_)),
    default_upload_buffer_(std::move(other.default_upload_buffer_)), free_head_(std::exchange(other.free_head_, 0)),
    capacity_(std::exchange(other.capacity_, 0)), size_(std::exchange(other.size_, 0)),
    defaults_uploaded_(std::exchange(other.defaults_uploaded_, false)),
    pending_uploads_(std::move(other.pending_uploads_)), debug_name_(std::move(other.debug_name_)) {}

auto ImageStorage::operator=(ImageStorage &&other) noexcept -> ImageStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

    default_upload_buffer_ = std::move(other.default_upload_buffer_);

    free_head_ = std::exchange(other.free_head_, 0);

    capacity_ = std::exchange(other.capacity_, 0);

    size_ = std::exchange(other.size_, 0);

    defaults_uploaded_ = std::exchange(other.defaults_uploaded_, false);

    debug_name_ = std::move(other.debug_name_);

    return *this;
}

auto ImageStorage::create(VulkanContext &context, ImageStorageCreateInfo const &create_info)
        -> std::expected<ImageStorage, ImageStorageError> {
    if (create_info.capacity < default_image_count || context.device == VK_NULL_HANDLE ||
        context.allocator == VK_NULL_HANDLE) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    ImageStorage storage;

    storage.context_ = &context;
    storage.capacity_ = create_info.capacity;

    storage.debug_name_ = std::string{create_info.debug_name};

    storage.slots_.resize(create_info.capacity);

    auto defaults = storage.create_default_images();

    if (!defaults) {
        storage.destroy();

        return std::unexpected(defaults.error());
    }

    /*
     * Slots 0 through 5 are occupied by defaults.
     * The ordinary free list begins at slot 6.
     */
    if (create_info.capacity > default_image_count) {
        storage.free_head_ = default_image_count;

        for (std::uint32_t index = default_image_count; index < create_info.capacity; ++index) {
            storage.slots_[index].next_free = index + 1 < create_info.capacity ? index + 1 : 0;
        }
    }

    return storage;
}

auto ImageStorage::create_default_images() -> std::expected<void, ImageStorageError> {
    constexpr auto upload_size = static_cast<VkDeviceSize>(default_pixels.size() * sizeof(default_pixels[0]));

    auto upload = Buffer::create(*context_, BufferCreateInfo{
                                                    .size = upload_size,
                                                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    .memory = BufferMemory::upload,
                                                    .debug_name = "image_storage.default_upload",
                                            });

    if (!upload) {
        return std::unexpected(make_device_error(upload.error()));
    }

    auto *mapped = static_cast<std::byte *>(upload->allocation_info.pMappedData);

    if (mapped == nullptr) {
        upload->destroy();

        return std::unexpected(make_error(ImageStorageErrorType::device_error));
    }

    std::memcpy(mapped, default_pixels.data(), static_cast<std::size_t>(upload_size));

    auto const flush_result = vmaFlushAllocation(context_->allocator, upload->allocation, 0, upload_size);

    if (flush_result != VK_SUCCESS) {
        upload->destroy();

        return std::unexpected(make_error(ImageStorageErrorType::device_error));
    }

    default_upload_buffer_ = std::move(*upload);

    for (std::uint32_t index = 0; index < default_image_count; ++index) {
        auto const name = debug_name_ + ".default." + std::string{default_names[index]};

        auto image = Image::create(
                *context_, ImageCreateInfo{
                                   .extent =
                                           VkExtent3D{
                                                   .width = 1,
                                                   .height = 1,
                                                   .depth = 1,
                                           },
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                   .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .image_type = VK_IMAGE_TYPE_2D,
                                   .view_type = VK_IMAGE_VIEW_TYPE_2D,
                                   .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                                   .flags = 0,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling = VK_IMAGE_TILING_OPTIMAL,
                                   .mip_levels = 1,
                                   .array_layers = 1,

                                   .debug_name = name,

                           });

        if (!image) {
            return std::unexpected(make_image_error(image.error()));
        }

        auto &slot = slots_[index];

        slot.image = std::move(*image);
        slot.generation = 1;
        slot.next_free = 0;
        slot.occupied = true;
        slot.protected_default = true;

        ++size_;
    }

    return {};
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info) -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (free_head_ == 0) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    auto image = Image::create(*context_, create_info);

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    free_head_ = slot.next_free;

    slot.image = std::move(*image);
    slot.next_free = 0;
    slot.occupied = true;
    slot.protected_default = false;

    bump_revision(slot);

    ++size_;

    return ImageHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels)
        -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (free_head_ == 0) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    auto image = Image::create(*context_, create_info, pixels);

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    free_head_ = slot.next_free;

    slot.image = std::move(*image);
    slot.next_free = 0;
    slot.occupied = true;
    slot.protected_default = false;

    bump_revision(slot);

    ++size_;

    return ImageHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                                VkCommandBuffer command_buffer) -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr || command_buffer == VK_NULL_HANDLE) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (free_head_ == 0) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto const upload_size = static_cast<VkDeviceSize>(pixels.size_bytes());

    auto staging = Buffer::create(*context_, BufferCreateInfo{
                                                     .size = upload_size,
                                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     .memory = BufferMemory::upload,
                                                     .debug_name = "image_storage.pending_upload",
                                             });

    if (!staging) {
        return std::unexpected(make_device_error(staging.error()));
    }

    auto *mapped = static_cast<std::byte *>(staging->allocation_info.pMappedData);

    if (mapped == nullptr) {
        staging->destroy();

        return std::unexpected(make_error(ImageStorageErrorType::device_error));
    }

    std::memcpy(mapped, pixels.data(), static_cast<std::size_t>(upload_size));

    auto const flush_result = vmaFlushAllocation(context_->allocator, staging->allocation, 0, upload_size);

    if (flush_result != VK_SUCCESS) {
        staging->destroy();

        return std::unexpected(make_error(ImageStorageErrorType::device_error));
    }

    auto image = Image::create(*context_, create_info);

    if (!image) {
        staging->destroy();

        return std::unexpected(make_image_error(image.error()));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    free_head_ = slot.next_free;

    slot.image = std::move(*image);
    slot.next_free = 0;
    slot.occupied = true;
    slot.protected_default = false;

    bump_revision(slot);

    ++size_;

    auto const mip_levels = create_info.mip_levels;

    VkImageMemoryBarrier2 const to_transfer{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = slot.image.image(),
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = mip_levels,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    VkDependencyInfo const to_transfer_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_transfer,
    };

    vkCmdPipelineBarrier2(command_buffer, &to_transfer_info);

    VkBufferImageCopy2 const region{
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .pNext = nullptr,
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                    VkImageSubresourceLayers{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
            .imageOffset = {0, 0, 0},
            .imageExtent = create_info.extent,
    };

    VkCopyBufferToImageInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .pNext = nullptr,
            .srcBuffer = staging->buffer,
            .dstImage = slot.image.image(),
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = 1,
            .pRegions = &region,
    };

    vkCmdCopyBufferToImage2(command_buffer, &copy_info);

    auto mip_width = static_cast<std::int32_t>(create_info.extent.width);
    auto mip_height = static_cast<std::int32_t>(create_info.extent.height);

    VkImageMemoryBarrier2 mip_barrier{};
    mip_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    mip_barrier.pNext = nullptr;
    mip_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mip_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mip_barrier.image = slot.image.image();

    VkDependencyInfo const mip_dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &mip_barrier,
    };

    for (std::uint32_t mip = 1; mip < mip_levels; ++mip) {
        mip_barrier.subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mip - 1,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
        };
        mip_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        mip_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mip_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        mip_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        mip_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        mip_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        vkCmdPipelineBarrier2(command_buffer, &mip_dependency_info);

        auto const next_width = mip_width > 1 ? mip_width / 2 : 1;
        auto const next_height = mip_height > 1 ? mip_height / 2 : 1;

        VkImageBlit const blit{
                .srcSubresource =
                        VkImageSubresourceLayers{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .mipLevel = mip - 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
                .srcOffsets = {VkOffset3D{0, 0, 0}, VkOffset3D{mip_width, mip_height, 1}},
                .dstSubresource =
                        VkImageSubresourceLayers{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .mipLevel = mip,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
                .dstOffsets = {VkOffset3D{0, 0, 0}, VkOffset3D{next_width, next_height, 1}},
        };

        vkCmdBlitImage(command_buffer, slot.image.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.image.image(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        mip_barrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        mip_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        mip_barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mip_barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        mip_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        mip_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        vkCmdPipelineBarrier2(command_buffer, &mip_dependency_info);

        mip_width = next_width;
        mip_height = next_height;
    }


    mip_barrier.subresourceRange = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = mip_levels - 1,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
    };
    mip_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
    mip_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mip_barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mip_barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    mip_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    mip_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier2(command_buffer, &mip_dependency_info);


    pending_uploads_.push_back(std::move(*staging));

    return ImageHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto ImageStorage::release_completed_uploads() -> void {
    for (auto &buffer: pending_uploads_) {
        buffer.destroy();
    }

    pending_uploads_.clear();
}

auto ImageStorage::destroy_image(ImageHandle handle) -> std::expected<void, ImageStorageError> {
    auto *slot = slot_for(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_handle));
    }

    if (slot->protected_default) {
        return std::unexpected(make_error(ImageStorageErrorType::protected_default));
    }

    slot->image.destroy();
    slot->occupied = false;

    bump_revision(*slot);

    ++slot->generation;

    if (slot->generation == 0) {
        slot->generation = 1;
    }

    slot->next_free = free_head_;
    free_head_ = handle.index;

    --size_;

    return {};
}

auto ImageStorage::prepare_frame(VkCommandBuffer command_buffer) -> std::expected<void, ImageStorageError> {
    if (defaults_uploaded_) {
        return {};
    }

    if (context_ == nullptr || command_buffer == VK_NULL_HANDLE || default_upload_buffer_.buffer == VK_NULL_HANDLE) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    std::array<VkImageMemoryBarrier2, default_image_count> to_transfer{};

    for (std::uint32_t index = 0; index < default_image_count; ++index) {
        to_transfer[index] = VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                .srcAccessMask = VK_ACCESS_2_NONE,
                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = slots_[index].image.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };
    }

    VkDependencyInfo const to_transfer_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = default_image_count,
            .pImageMemoryBarriers = to_transfer.data(),
    };

    vkCmdPipelineBarrier2(command_buffer, &to_transfer_info);

    for (std::uint32_t index = 0; index < default_image_count; ++index) {
        VkBufferImageCopy2 const region{
                .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .pNext = nullptr,
                .bufferOffset = static_cast<VkDeviceSize>(index * sizeof(default_pixels[0])),
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                        VkImageSubresourceLayers{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .mipLevel = 0,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
                .imageOffset = {0, 0, 0},
                .imageExtent =
                        VkExtent3D{
                                .width = 1,
                                .height = 1,
                                .depth = 1,
                        },
        };

        VkCopyBufferToImageInfo2 const copy_info{
                .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                .pNext = nullptr,
                .srcBuffer = default_upload_buffer_.buffer,
                .dstImage = slots_[index].image.image(),
                .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .regionCount = 1,
                .pRegions = &region,
        };

        vkCmdCopyBufferToImage2(command_buffer, &copy_info);
    }

    std::array<VkImageMemoryBarrier2, default_image_count> to_sampled{};

    for (std::uint32_t index = 0; index < default_image_count; ++index) {
        to_sampled[index] = VkImageMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = slots_[index].image.image(),
                .subresourceRange =
                        VkImageSubresourceRange{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
        };
    }

    VkDependencyInfo const to_sampled_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = default_image_count,
            .pImageMemoryBarriers = to_sampled.data(),
    };

    vkCmdPipelineBarrier2(command_buffer, &to_sampled_info);

    defaults_uploaded_ = true;

    return {};
}

auto ImageStorage::get(ImageHandle handle) noexcept -> Image * {
    auto *slot = slot_for(handle);

    return slot != nullptr ? &slot->image : nullptr;
}

auto ImageStorage::get(ImageHandle handle) const noexcept -> Image const * {
    auto const *slot = slot_for(handle);

    return slot != nullptr ? &slot->image : nullptr;
}

auto ImageStorage::destroy() noexcept -> void {
    default_upload_buffer_.destroy();

    for (auto &buffer: pending_uploads_) {
        buffer.destroy();
    }
    pending_uploads_.clear();

    for (auto &slot: slots_) {
        if (!slot.occupied) {
            continue;
        }

        slot.image.destroy();
        slot.occupied = false;
    }

    slots_.clear();

    context_ = nullptr;

    free_head_ = 0;
    capacity_ = 0;
    size_ = 0;

    defaults_uploaded_ = false;

    debug_name_.clear();
}

auto ImageStorage::slot_for(ImageHandle handle) noexcept -> Slot * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto ImageStorage::slot_for(ImageHandle handle) const noexcept -> Slot const * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto const &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto ImageStorage::descriptor_record(std::uint32_t index) const noexcept -> ImageDescriptorRecord {
    if (index >= slots_.size()) {
        return {};
    }

    auto const &slot = slots_[index];

    if (!slot.occupied) {
        return ImageDescriptorRecord{
                .revision = slot.revision,
                .occupied = false,
        };
    }

    return ImageDescriptorRecord{
            .sampled_2d = slot.image.descriptor_view(ImageDescriptorView::sampled_2d),
            .sampled_cube = slot.image.descriptor_view(ImageDescriptorView::sampled_cube),
            .sampled_2d_array = slot.image.descriptor_view(ImageDescriptorView::sampled_2d_array),
            .storage_2d = slot.image.descriptor_view(ImageDescriptorView::storage_2d),
            .storage_2d_array = slot.image.descriptor_view(ImageDescriptorView::storage_2d_array),
            .revision = slot.revision,
            .occupied = true,
    };
}

auto ImageStorage::descriptor_revision(std::uint32_t index) const noexcept -> std::uint64_t {
    if (index >= slots_.size()) {
        return 0;
    }

    return slots_[index].revision;
}

auto ImageStorage::occupied(std::uint32_t index) const noexcept -> bool {
    return index < slots_.size() && slots_[index].occupied;
}
