#pragma once

#include <cstdint>
#include <expected>
#include <string_view>

#include <volk.h>

#include "image_storage.hxx"

enum class ForwardTargetErrorType : std::uint8_t {
    invalid_argument,
    image_error,
};

struct ForwardTargetError {
    ForwardTargetErrorType type = ForwardTargetErrorType::invalid_argument;

    ImageStorageError image_error{};
};

struct ForwardTargetCreateInfo {
    VkExtent2D extent{};

    VkFormat hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    std::string_view debug_name = "forward_target";
};

class ForwardTarget {
public:
    ForwardTarget() = default;

    ForwardTarget(ForwardTarget const &) = delete;

    auto operator=(ForwardTarget const &) -> ForwardTarget & = delete;

    ForwardTarget(ForwardTarget &&other) noexcept;

    auto operator=(ForwardTarget &&other) noexcept -> ForwardTarget &;

    [[nodiscard]]
    static auto create(ImageStorage &image_storage, ForwardTargetCreateInfo const &create_info)
            -> std::expected<ForwardTarget, ForwardTargetError>;

    auto destroy(ImageStorage &image_storage) noexcept -> void;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return hdr_.valid() && depth_.valid();
    }

    [[nodiscard]]
    auto hdr() const noexcept -> ImageHandle {
        return hdr_;
    }

    [[nodiscard]]
    auto depth() const noexcept -> ImageHandle {
        return depth_;
    }

    [[nodiscard]]
    auto extent() const noexcept -> VkExtent2D {
        return extent_;
    }

    [[nodiscard]]
    auto hdr_format() const noexcept -> VkFormat {
        return hdr_format_;
    }

    [[nodiscard]]
    auto depth_format() const noexcept -> VkFormat {
        return depth_format_;
    }

    [[nodiscard]]
    auto samples() const noexcept -> VkSampleCountFlagBits {
        return samples_;
    }

private:
    ImageHandle hdr_{};
    ImageHandle depth_{};

    VkExtent2D extent_{};

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;

    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;

    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
};
