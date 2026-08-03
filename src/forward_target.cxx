#include "forward_target.hxx"

#include <format>
#include <string>
#include <utility>
#include "logger.hxx"

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

} // namespace

ForwardTarget::ForwardTarget(ForwardTarget &&other) noexcept :
    hdr_(std::exchange(other.hdr_, ImageHandle{})), resolved_hdr_(std::exchange(other.resolved_hdr_, ImageHandle{})),
    depth_(std::exchange(other.depth_, ImageHandle{})),
    resolved_depth_(std::exchange(other.resolved_depth_, ImageHandle{})),
    extent_(std::exchange(other.extent_, VkExtent2D{})),
    hdr_format_(std::exchange(other.hdr_format_, VK_FORMAT_UNDEFINED)),
    depth_format_(std::exchange(other.depth_format_, VK_FORMAT_UNDEFINED)),
    samples_(std::exchange(other.samples_, VK_SAMPLE_COUNT_1_BIT)) {}

auto ForwardTarget::operator=(ForwardTarget &&other) noexcept -> ForwardTarget & {
    if (this == &other) {
        return *this;
    }

    /*
     * ForwardTarget does not directly own ImageStorage,
     * so the current handles must already have been
     * destroyed before move-assignment.
     */
    hdr_ = std::exchange(other.hdr_, ImageHandle{});
    depth_ = std::exchange(other.depth_, ImageHandle{});
    resolved_hdr_ = std::exchange(other.resolved_hdr_, ImageHandle{});
    resolved_depth_ = std::exchange(other.resolved_depth_, ImageHandle{});

    extent_ = std::exchange(other.extent_, VkExtent2D{});
    hdr_format_ = std::exchange(other.hdr_format_, VK_FORMAT_UNDEFINED);
    depth_format_ = std::exchange(other.depth_format_, VK_FORMAT_UNDEFINED);
    samples_ = std::exchange(other.samples_, VK_SAMPLE_COUNT_1_BIT);

    return *this;
}

auto ForwardTarget::create(ImageStorage &image_storage, ForwardTargetCreateInfo const &create_info)
        -> std::expected<ForwardTarget, ForwardTargetError> {
    if (create_info.extent.width == 0 || create_info.extent.height == 0 ||
        create_info.hdr_format == VK_FORMAT_UNDEFINED || create_info.depth_format == VK_FORMAT_UNDEFINED) {
        return std::unexpected(make_error(ForwardTargetErrorType::invalid_argument));
    }

    bool const is_msaa = create_info.samples > VK_SAMPLE_COUNT_1_BIT;

    // Always create the single-sample HDR image: it's the render target when
    // there's no MSAA, and the resolve destination when there is.
    auto const resolved_hdr_name = std::string{create_info.debug_name} + ".hdr";

    auto resolved_hdr = image_storage.create_image(ImageCreateInfo{
            .extent =
                    VkExtent3D{
                            .width = create_info.extent.width,
                            .height = create_info.extent.height,
                            .depth = 1,
                    },
            .format = create_info.hdr_format,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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

    // Only create a distinct multisample render target when MSAA is on. It's
    // never sampled directly, so no SAMPLED usage / descriptor view for it.
    std::expected<ImageHandle, ImageStorageError> msaa_hdr;

    if (is_msaa) {
        auto const msaa_hdr_name = std::string{create_info.debug_name} + ".hdr_msaa";

        msaa_hdr = image_storage.create_image(ImageCreateInfo{
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

            static_cast<void>(image_storage.destroy_image(*resolved_hdr));

            return std::unexpected(make_image_error(msaa_hdr.error()));
        }
    }

    auto const depth_name = std::string{create_info.debug_name} + (is_msaa ? ".depth_msaa" : ".depth");

    auto depth = image_storage.create_image(ImageCreateInfo{
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
        if (is_msaa) {
            static_cast<void>(image_storage.destroy_image(*msaa_hdr));
        }

        static_cast<void>(image_storage.destroy_image(*resolved_hdr));

        return std::unexpected(make_image_error(depth.error()));
    }

    std::expected<ImageHandle, ImageStorageError> resolved_depth;

    if (is_msaa) {
        auto const resolved_depth_name = std::string{create_info.debug_name} + ".depth";

        resolved_depth = image_storage.create_image(ImageCreateInfo{
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
            static_cast<void>(image_storage.destroy_image(*depth));
            static_cast<void>(image_storage.destroy_image(*msaa_hdr));
            static_cast<void>(image_storage.destroy_image(*resolved_hdr));

            return std::unexpected(make_image_error(resolved_depth.error()));
        }
    }

    ForwardTarget target;

    target.hdr_ = is_msaa ? *msaa_hdr : *resolved_hdr;
    target.resolved_hdr_ = is_msaa ? *resolved_hdr : ImageHandle{};
    target.depth_ = *depth;
    target.resolved_depth_ = is_msaa ? *resolved_depth : ImageHandle{};

    target.extent_ = create_info.extent;
    target.hdr_format_ = create_info.hdr_format;
    target.depth_format_ = create_info.depth_format;
    target.samples_ = create_info.samples;

    return target;
}

auto ForwardTarget::destroy(ImageStorage &image_storage) noexcept -> void {
    if (depth_.valid()) {
        static_cast<void>(image_storage.destroy_image(depth_));
    }

    if (resolved_depth_.valid()) {
        static_cast<void>(image_storage.destroy_image(resolved_depth_));
    }

    if (hdr_.valid()) {
        static_cast<void>(image_storage.destroy_image(hdr_));
    }

    if (resolved_hdr_.valid()) {
        static_cast<void>(image_storage.destroy_image(resolved_hdr_));
    }

    hdr_ = {};
    resolved_hdr_ = {};
    depth_ = {};
    resolved_depth_ = {};

    extent_ = {};
    hdr_format_ = VK_FORMAT_UNDEFINED;
    depth_format_ = VK_FORMAT_UNDEFINED;
    samples_ = VK_SAMPLE_COUNT_1_BIT;
}
