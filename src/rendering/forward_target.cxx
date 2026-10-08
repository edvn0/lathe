#include "rendering/forward_target.hxx"

#include <format>
#include <string>
#include <utility>
#include "core/logger.hxx"

#include "gpu/image_storage.hxx"

namespace {

    auto make_error(ForwardTargetErrorType type) noexcept -> ForwardTargetError {
        return ForwardTargetError{
                .type = type,
        };
    }

    auto make_image_error(ImageStorageError error) noexcept -> ForwardTargetError {
        return ForwardTargetError{
                .type = ForwardTargetErrorType::image_error,
                .cause = ErrorCause{Boxed<ImageStorageError>{std::move(error)}},
        };
    }

}

auto ForwardTarget::create(ImageStorage &image_storage, ForwardTargetCreateInfo const &create_info)
        -> std::expected<ForwardTarget, ForwardTargetError> {
    if (create_info.extent.width == 0 || create_info.extent.height == 0 ||
        create_info.hdr_format == VK_FORMAT_UNDEFINED || create_info.depth_format == VK_FORMAT_UNDEFINED) {
        return std::unexpected(make_error(ForwardTargetErrorType::invalid_argument));
    }

    bool const is_msaa = create_info.samples > VK_SAMPLE_COUNT_1_BIT;

    auto const resolved_hdr_name = std::string{create_info.debug_name} + ".hdr";

    auto resolved_hdr = create_held_image(
            image_storage, ImageCreateInfo{
                                   .extent =
                                           VkExtent3D{
                                                   .width = create_info.extent.width,
                                                   .height = create_info.extent.height,
                                                   .depth = 1,
                                           },
                                   .format = create_info.hdr_format,
                                   .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                   .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                   .image_type = VK_IMAGE_TYPE_2D,
                                   .view_type = VK_IMAGE_VIEW_TYPE_2D,
                                   .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                                   .flags = 0,
                                   .samples = VK_SAMPLE_COUNT_1_BIT,
                                   .tiling = VK_IMAGE_TILING_OPTIMAL,
                                   .mip_levels = 1,
                                   .array_layers = 1,
                                   .debug_name = resolved_hdr_name,
                           });

    if (!resolved_hdr) {
        warn("Could not create the resolved HDR image (1 sample)");
        return std::unexpected(make_image_error(resolved_hdr.error()));
    }

    std::expected<ImageHolder, ImageStorageError> msaa_hdr;

    if (is_msaa) {
        auto const msaa_hdr_name = std::string{create_info.debug_name} + ".hdr_msaa";

        msaa_hdr = create_held_image(
                image_storage, ImageCreateInfo{
                                       .extent =
                                               VkExtent3D{
                                                       .width = create_info.extent.width,
                                                       .height = create_info.extent.height,
                                                       .depth = 1,
                                               },
                                       .format = create_info.hdr_format,
                                       .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                       .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .image_type = VK_IMAGE_TYPE_2D,
                                       .view_type = VK_IMAGE_VIEW_TYPE_2D,
                                       .descriptor_views = 0,
                                       .flags = 0,
                                       .samples = create_info.samples,
                                       .tiling = VK_IMAGE_TILING_OPTIMAL,
                                       .mip_levels = 1,
                                       .array_layers = 1,
                                       .debug_name = msaa_hdr_name,
                               });

        if (!msaa_hdr) {
            warn("Could not create the MSAA HDR image ({} samples)", static_cast<std::uint32_t>(create_info.samples));

            return std::unexpected(make_image_error(msaa_hdr.error()));
        }
    }

    auto const depth_name = std::string{create_info.debug_name} + (is_msaa ? ".depth_msaa" : ".depth");

    auto depth = create_held_image(
            image_storage,
            ImageCreateInfo{
                    .extent =
                            VkExtent3D{
                                    .width = create_info.extent.width,
                                    .height = create_info.extent.height,
                                    .depth = 1,
                            },
                    .format = create_info.depth_format,
                    .usage = VkImageUsageFlags{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT} |
                             (is_msaa ? VkImageUsageFlags{} : VkImageUsageFlags{VK_IMAGE_USAGE_SAMPLED_BIT}),
                    .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .image_type = VK_IMAGE_TYPE_2D,
                    .view_type = VK_IMAGE_VIEW_TYPE_2D,
                    .descriptor_views = is_msaa ? 0u : image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                    .flags = 0,
                    .samples = create_info.samples,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .mip_levels = 1,
                    .array_layers = 1,
                    .debug_name = depth_name,
            });

    if (!depth) {
        return std::unexpected(make_image_error(depth.error()));
    }

    std::expected<ImageHolder, ImageStorageError> resolved_depth;

    if (is_msaa) {
        auto const resolved_depth_name = std::string{create_info.debug_name} + ".depth";

        resolved_depth = create_held_image(
                image_storage,
                ImageCreateInfo{
                        .extent =
                                VkExtent3D{
                                        .width = create_info.extent.width,
                                        .height = create_info.extent.height,
                                        .depth = 1,
                                },
                        .format = create_info.depth_format,
                        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .image_type = VK_IMAGE_TYPE_2D,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .flags = 0,
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = VK_IMAGE_TILING_OPTIMAL,
                        .mip_levels = 1,
                        .array_layers = 1,
                        .debug_name = resolved_depth_name,
                });

        if (!resolved_depth) {
            return std::unexpected(make_image_error(resolved_depth.error()));
        }
    }

    ForwardTarget target;

    if (is_msaa) {
        target.hdr_ = std::move(*msaa_hdr);
        target.resolved_hdr_ = std::move(*resolved_hdr);
        target.resolved_depth_ = std::move(*resolved_depth);
    } else {
        target.hdr_ = std::move(*resolved_hdr);
    }

    target.depth_ = std::move(*depth);

    target.extent_ = create_info.extent;
    target.hdr_format_ = create_info.hdr_format;
    target.depth_format_ = create_info.depth_format;
    target.samples_ = create_info.samples;

    return target;
}
