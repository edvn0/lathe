#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "error_context.hxx"
#include "forward.hxx"
#include "pipeline.hxx"

struct PipelineHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0;
    }

    auto operator==(PipelineHandle const &) const -> bool = default;
};

enum class PipelineStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
    pipeline_error,
};

struct PipelineStorageError {
    PipelineStorageErrorType type = PipelineStorageErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<PipelineStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(PipelineStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case PipelineStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case PipelineStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case PipelineStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case PipelineStorageErrorType::pipeline_error:
                    return "pipeline_error";
            }

            return "unknown_pipeline_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct PipelineStorageCreateInfo {
    std::uint32_t capacity = 0;

    VkDescriptorSetLayout global_descriptor_set_layout = VK_NULL_HANDLE;

    std::string_view debug_name = "pipeline_storage";
};

class PipelineStorage {
public:
    PipelineStorage() = default;
    ~PipelineStorage();

    PipelineStorage(PipelineStorage const &) = delete;

    auto operator=(PipelineStorage const &) -> PipelineStorage & = delete;

    PipelineStorage(PipelineStorage &&other) noexcept;

    auto operator=(PipelineStorage &&other) noexcept -> PipelineStorage &;

    [[nodiscard]]
    static auto create(VulkanContext &context, PipelineStorageCreateInfo const &create_info)
            -> std::expected<PipelineStorage, PipelineStorageError>;

    [[nodiscard]]
    auto create_graphics(GraphicsPipelineCreateInfo const &create_info)
            -> std::expected<PipelineHandle, PipelineStorageError>;

    [[nodiscard]]
    auto destroy_pipeline(PipelineHandle handle) -> std::expected<void, PipelineStorageError>;

    [[nodiscard]]
    auto get(PipelineHandle handle) noexcept -> Pipeline *;

    [[nodiscard]]
    auto get(PipelineHandle handle) const noexcept -> Pipeline const *;

    [[nodiscard]]
    auto contains(PipelineHandle handle) const noexcept -> bool {
        return get(handle) != nullptr;
    }

    [[nodiscard]]
    auto size() const noexcept -> std::uint32_t {
        return size_;
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return capacity_;
    }

    [[nodiscard]]
    auto global_descriptor_set_layout() const noexcept -> VkDescriptorSetLayout {
        return global_descriptor_set_layout_;
    }

    auto destroy() noexcept -> void;

private:
    struct Slot {
        Pipeline pipeline{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
    };

    [[nodiscard]]
    auto slot_for(PipelineHandle handle) noexcept -> Slot *;

    [[nodiscard]]
    auto slot_for(PipelineHandle handle) const noexcept -> Slot const *;

    VulkanContext *context_ = nullptr;

    std::vector<Slot> slots_;

    std::uint32_t free_head_ = 0;
    std::uint32_t capacity_ = 0;
    std::uint32_t size_ = 0;

    VkDescriptorSetLayout global_descriptor_set_layout_ = VK_NULL_HANDLE;
    std::string debug_name_;
};
