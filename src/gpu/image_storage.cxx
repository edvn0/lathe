#include "gpu/image_storage.hxx"

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include "gpu/context.hxx"

namespace {
    constexpr std::array<std::array<std::uint8_t, 4>, default_image_count> default_pixels{{
            {255, 255, 255, 255},

            {0, 0, 0, 255},

            {128, 128, 255, 255},

            {0, 255, 0, 255},

            {255, 255, 255, 255},

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
    default_upload_buffer_(std::move(other.default_upload_buffer_)),
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

    storage.debug_name_ = std::string{create_info.debug_name};

    storage.slots_ = ObjectPool<ImageSlotData>::create(create_info.capacity);

    // The six default images are allocated first, in order, out of a
    // freshly-created pool, so they always land on indices 0-5 -- matching
    // default_image_handle()'s hard-coded indices.
    auto defaults = storage.create_default_images();

    if (!defaults) {
        storage.destroy();

        return std::unexpected(defaults.error());
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

        auto &slot = slots_.allocate()->second;

        slot.image = std::move(*image);
        slot.protected_default = true;
    }

    return {};
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info) -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (slots_.size() >= slots_.capacity()) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto image = Image::create(*context_, create_info);

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    auto [handle, slot] = *slots_.allocate();

    slot.image = std::move(*image);
    slot.protected_default = false;

    bump_revision(slot);

    return handle;
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels)
        -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (slots_.size() >= slots_.capacity()) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto image = Image::create(*context_, create_info, pixels);

    if (!image) {
        return std::unexpected(make_image_error(image.error()));
    }

    auto [handle, slot] = *slots_.allocate();

    slot.image = std::move(*image);
    slot.protected_default = false;

    bump_revision(slot);

    return handle;
}

auto ImageStorage::create_image(ImageCreateInfo const &create_info, std::span<const std::byte> pixels,
                                VkCommandBuffer command_buffer) -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr || command_buffer == VK_NULL_HANDLE) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    if (slots_.size() >= slots_.capacity()) {
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

    auto [handle, slot] = *slots_.allocate();

    slot.image = std::move(*image);
    slot.protected_default = false;

    bump_revision(slot);

    auto const mip_levels = create_info.mip_levels;

    VkImageMemoryBarrier2 const to_transfer{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT,
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

    return handle;
}

auto ImageStorage::upgrade_pending_image(ImageHandle handle, CompressedTexture const &texture,
                                         VkCommandBuffer command_buffer) -> std::expected<Buffer, ImageStorageError> {
    if (context_ == nullptr || command_buffer == VK_NULL_HANDLE || texture.mips.empty() ||
        texture.format == VK_FORMAT_UNDEFINED) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_handle));
    }

    if (slot->protected_default) {
        return std::unexpected(make_error(ImageStorageErrorType::protected_default));
    }

    auto const upload_size = static_cast<VkDeviceSize>(texture.data.size());

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

    std::memcpy(mapped, texture.data.data(), texture.data.size());

    auto const flush_result = vmaFlushAllocation(context_->allocator, staging->allocation, 0, upload_size);

    if (flush_result != VK_SUCCESS) {
        staging->destroy();

        return std::unexpected(make_error(ImageStorageErrorType::device_error));
    }

    auto const mip_levels = static_cast<std::uint32_t>(texture.mips.size());

    auto image = Image::create(*context_,
                               ImageCreateInfo{
                                       .extent =
                                               VkExtent3D{
                                                       .width = texture.width,
                                                       .height = texture.height,
                                                       .depth = 1,
                                               },
                                       .format = texture.format,
                                       .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                       .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .image_type = VK_IMAGE_TYPE_2D,
                                       .view_type = VK_IMAGE_VIEW_TYPE_2D,
                                       .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                                       .samples = VK_SAMPLE_COUNT_1_BIT,
                                       .tiling = VK_IMAGE_TILING_OPTIMAL,
                                       .mip_levels = mip_levels,
                                       .array_layers = 1,
                                       .debug_name = texture.debug_name,
                               });

    if (!image) {
        staging->destroy();

        return std::unexpected(make_image_error(image.error()));
    }

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
            .image = image->image(),
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

    std::vector<VkBufferImageCopy2> regions;
    regions.reserve(mip_levels);

    for (std::uint32_t level = 0; level < mip_levels; ++level) {
        auto const &mip = texture.mips[level];

        regions.push_back(VkBufferImageCopy2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                .pNext = nullptr,
                .bufferOffset = mip.byte_offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                        VkImageSubresourceLayers{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .mipLevel = level,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                        },
                .imageOffset = {0, 0, 0},
                .imageExtent = {mip.width, mip.height, 1},
        });
    }

    VkCopyBufferToImageInfo2 const copy_info{
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .pNext = nullptr,
            .srcBuffer = staging->buffer,
            .dstImage = image->image(),
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount = mip_levels,
            .pRegions = regions.data(),
    };

    vkCmdCopyBufferToImage2(command_buffer, &copy_info);

    VkImageMemoryBarrier2 const to_shader_read{
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
            .image = image->image(),
            .subresourceRange =
                    VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = mip_levels,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                    },
    };

    VkDependencyInfo const to_shader_read_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = 0,
            .memoryBarrierCount = 0,
            .pMemoryBarriers = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &to_shader_read,
    };

    vkCmdPipelineBarrier2(command_buffer, &to_shader_read_info);

    if (slot->is_alias) {
        slot->alias_sampled_2d = VK_NULL_HANDLE;
        slot->alias_storage_2d = VK_NULL_HANDLE;
        slot->is_alias = false;
    } else {
        slot->image.destroy();
    }

    slot->image = std::move(*image);

    bump_revision(*slot);

    return std::move(*staging);
}

auto ImageStorage::create_pending_image(ImageHandle fallback) -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    auto const *fallback_slot = slots_.get(fallback);

    if (fallback_slot == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_handle));
    }

    if (slots_.size() >= slots_.capacity()) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    // slots_'s backing storage is sized once at create() and never resized
    // by allocate()/release(), so fallback_slot stays valid across this call.
    auto [handle, slot] = *slots_.allocate();

    if (fallback_slot->is_alias) {
        slot.alias_sampled_2d = fallback_slot->alias_sampled_2d;
        slot.alias_storage_2d = fallback_slot->alias_storage_2d;
    } else {
        slot.alias_sampled_2d = fallback_slot->image.descriptor_view(ImageDescriptorView::sampled_2d);
        slot.alias_storage_2d = fallback_slot->image.descriptor_view(ImageDescriptorView::storage_2d);
    }

    slot.is_alias = true;
    slot.protected_default = false;

    bump_revision(slot);

    return handle;
}

auto ImageStorage::release_completed_uploads() -> void {
    for (auto &buffer: pending_uploads_) {
        buffer.destroy();
    }

    pending_uploads_.clear();
}

auto ImageStorage::destroy_image(ImageHandle handle) -> std::expected<void, ImageStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_handle));
    }

    if (slot->protected_default) {
        return std::unexpected(make_error(ImageStorageErrorType::protected_default));
    }

    if (slot->is_alias) {
        slot->alias_sampled_2d = VK_NULL_HANDLE;
        slot->alias_storage_2d = VK_NULL_HANDLE;
        slot->is_alias = false;
    } else {
        slot->image.destroy();
    }

    bump_revision(*slot);

    static_cast<void>(slots_.release(handle));

    return {};
}

auto ImageStorage::prepare_frame(VkCommandBuffer command_buffer) -> std::expected<void, ImageStorageError> {
    if (defaults_uploaded_) [[likely]] {
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
                .image = slots_.get_at(index)->image.image(),
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
                .dstImage = slots_.get_at(index)->image.image(),
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
                .image = slots_.get_at(index)->image.image(),
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
    auto *slot = slots_.get(handle);

    return slot != nullptr ? &slot->image : nullptr;
}

auto ImageStorage::get(ImageHandle handle) const noexcept -> Image const * {
    auto const *slot = slots_.get(handle);

    return slot != nullptr ? &slot->image : nullptr;
}

auto ImageStorage::destroy() noexcept -> void {
    default_upload_buffer_.destroy();

    for (auto &buffer: pending_uploads_) {
        buffer.destroy();
    }
    pending_uploads_.clear();

    for (std::uint32_t index = 0; index < slots_.capacity(); ++index) {
        if (!slots_.occupied_at(index)) {
            continue;
        }

        auto *slot = slots_.get_at(index);

        if (!slot->is_alias) {
            slot->image.destroy();
        }
    }

    slots_ = ObjectPool<ImageSlotData>{};

    context_ = nullptr;

    defaults_uploaded_ = false;

    debug_name_.clear();
}

auto ImageStorage::descriptor_record(std::uint32_t index) const noexcept -> ImageDescriptorRecord {
    auto const *slot = slots_.get_at(index);

    if (slot == nullptr) {
        return {};
    }

    auto const occupied = slots_.occupied_at(index);

    if (!occupied) {
        return ImageDescriptorRecord{
                .revision = slot->revision,
                .occupied = false,
        };
    }

    if (slot->is_alias) {
        return ImageDescriptorRecord{
                .sampled_2d = slot->alias_sampled_2d,
                .storage_2d = slot->alias_storage_2d,
                .revision = slot->revision,
                .occupied = true,
        };
    }

    return ImageDescriptorRecord{
            .sampled_2d = slot->image.descriptor_view(ImageDescriptorView::sampled_2d),
            .sampled_cube = slot->image.descriptor_view(ImageDescriptorView::sampled_cube),
            .sampled_2d_array = slot->image.descriptor_view(ImageDescriptorView::sampled_2d_array),
            .storage_2d = slot->image.descriptor_view(ImageDescriptorView::storage_2d),
            .storage_2d_array = slot->image.descriptor_view(ImageDescriptorView::storage_2d_array),
            .revision = slot->revision,
            .occupied = true,
    };
}

auto ImageStorage::descriptor_revision(std::uint32_t index) const noexcept -> std::uint64_t {
    auto const *slot = slots_.get_at(index);

    return slot != nullptr ? slot->revision : 0;
}

auto ImageStorage::occupied(std::uint32_t index) const noexcept -> bool { return slots_.occupied_at(index); }

auto ImageStorage::register_view(ImageViewRegistration const &registration)
        -> std::expected<ImageHandle, ImageStorageError> {
    if (context_ == nullptr ||
        (registration.sampled_2d == VK_NULL_HANDLE && registration.storage_2d == VK_NULL_HANDLE)) {
        return std::unexpected(make_error(ImageStorageErrorType::invalid_argument));
    }

    auto allocation = slots_.allocate();

    if (!allocation) {
        return std::unexpected(make_error(ImageStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot.alias_sampled_2d = registration.sampled_2d;
    slot.alias_storage_2d = registration.storage_2d;
    slot.is_alias = true;
    slot.protected_default = false;

    bump_revision(slot);

    return handle;
}
