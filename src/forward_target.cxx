#include "forward_target.hxx"

#include <string>
#include <utility>

namespace {

    auto make_error(ForwardTargetErrorType type) noexcept -> ForwardTargetError {
        return ForwardTargetError{
                .type = type,
        };
    }

    auto make_image_error(ImageStorageError const &error) noexcept -> ForwardTargetError {
        return ForwardTargetError{
                .type = ForwardTargetErrorType::image_error,
                .image_error = error,
        };
    }

} // namespace

ForwardTarget::ForwardTarget(ForwardTarget &&other) noexcept :
    hdr_(std::exchange(other.hdr_, ImageHandle{})), depth_(std::exchange(other.depth_, ImageHandle{})),
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

    auto const hdr_name = std::string{create_info.debug_name} + ".hdr";

    auto hdr = image_storage.create_image(ImageCreateInfo{
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
            .samples = create_info.samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .mip_levels = 1,
            .array_layers = 1,
            .debug_name = hdr_name,
    });

    if (!hdr) {
        return std::unexpected(make_image_error(hdr.error()));
    }

    auto const depth_name = std::string{create_info.debug_name} + ".depth";

    auto depth = image_storage.create_image(ImageCreateInfo{
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
            .samples = create_info.samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .mip_levels = 1,
            .array_layers = 1,
            .debug_name = depth_name,
    });

    if (!depth) {
        static_cast<void>(image_storage.destroy_image(*hdr));

        return std::unexpected(make_image_error(depth.error()));
    }

    ForwardTarget target;

    target.hdr_ = *hdr;
    target.depth_ = *depth;

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

    if (hdr_.valid()) {
        static_cast<void>(image_storage.destroy_image(hdr_));
    }

    hdr_ = {};
    depth_ = {};

    extent_ = {};

    hdr_format_ = VK_FORMAT_UNDEFINED;

    depth_format_ = VK_FORMAT_UNDEFINED;

    samples_ = VK_SAMPLE_COUNT_1_BIT;
}
