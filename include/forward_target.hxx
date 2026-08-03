#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string_view>

#include <volk.h>

#include "error_context.hxx"
#include "image_storage.hxx"

enum class ForwardTargetErrorType : std::uint8_t {
    invalid_argument,
    image_error,
};

struct ForwardTargetError {
    ForwardTargetErrorType type = ForwardTargetErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<ForwardTargetErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ForwardTargetErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ForwardTargetErrorType::invalid_argument:
                    return "invalid_argument";
                case ForwardTargetErrorType::image_error:
                    return "image_error";
            }

            return "unknown_forward_target_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
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
        bool const resolve_ok = samples_ <= VK_SAMPLE_COUNT_1_BIT || (resolved_hdr_.valid() && resolved_depth_.valid());
        return hdr_.valid() && depth_.valid() && resolve_ok;
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

    [[nodiscard]]
    auto hdr() const noexcept -> ImageHandle {
        return hdr_;
    }

    [[nodiscard]]
    auto resolved_hdr() const noexcept -> ImageHandle {
        return resolved_hdr_.valid() ? resolved_hdr_ : hdr_;
    }

    [[nodiscard]]
    auto is_multisampled() const noexcept -> bool {
        return samples_ > VK_SAMPLE_COUNT_1_BIT;
    }

    [[nodiscard]]
    auto resolved_depth() const noexcept -> ImageHandle {
        return resolved_depth_.valid() ? resolved_depth_ : depth_;
    }

private:
    ImageHandle hdr_{};
    ImageHandle resolved_hdr_{};
    ImageHandle depth_{};
    ImageHandle resolved_depth_{};

    VkExtent2D extent_{};

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
};
