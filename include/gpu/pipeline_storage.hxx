#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"
#include "core/forward.hxx"
#include "core/object_pool.hxx"
#include "gpu/pipeline.hxx"

using PipelineHandle = Handle<Pipeline>;

enum class PipelineStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
    pipeline_error,
};

struct PipelineStorageError {
    PipelineStorageErrorType type = PipelineStorageErrorType::invalid_argument;

    std::optional<ErrorCause> cause{std::nullopt};
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
    std::filesystem::path cache_file_path;
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
    auto create_compute(ComputePipelineCreateInfo const &create_info)
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
        return slots_.size();
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return slots_.capacity();
    }

    [[nodiscard]]
    auto global_descriptor_set_layout() const noexcept -> VkDescriptorSetLayout {
        return global_descriptor_set_layout_;
    }

    [[nodiscard]]
    auto save_cache_to_disk() const -> std::expected<void, PipelineStorageError>;

    auto destroy() noexcept -> void;

private:
    VulkanContext *context_ = nullptr;

    ObjectPool<Pipeline> slots_;

    std::mutex slot_mutex_;

    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::filesystem::path cache_file_path_;

    VkDescriptorSetLayout global_descriptor_set_layout_ = VK_NULL_HANDLE;
    std::string debug_name_;
};
