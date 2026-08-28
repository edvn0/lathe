#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "error_context.hxx"
#include "forward.hxx"
#include "object_pool.hxx"
#include "shader_object.hxx"

// Sentinel stays the default (u32 max): index 0 is a legitimate slot here
// (unlike MaterialHandle), so validity is effectively generation != 0
// alone -- see PipelineHandle's identical reasoning.
using ShaderObjectHandle = Handle<ShaderObjectSet>;

enum class ShaderObjectStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
    shader_object_error,
};

struct ShaderObjectStorageError {
    ShaderObjectStorageErrorType type = ShaderObjectStorageErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<ShaderObjectStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ShaderObjectStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ShaderObjectStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case ShaderObjectStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case ShaderObjectStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case ShaderObjectStorageErrorType::shader_object_error:
                    return "shader_object_error";
            }

            return "unknown_shader_object_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct ShaderObjectStorageCreateInfo {
    std::uint32_t capacity = 0;

    VkDescriptorSetLayout global_descriptor_set_layout = VK_NULL_HANDLE;

    std::string_view debug_name = "shader_object_storage";
};

// Parallel to PipelineStorage: a generational-handle slot map owning
// ShaderObjectSet instances. See docs/pipeline_to_shader_objects.md Phase 4.
class ShaderObjectStorage {
public:
    ShaderObjectStorage() = default;
    ~ShaderObjectStorage();

    ShaderObjectStorage(ShaderObjectStorage const &) = delete;

    auto operator=(ShaderObjectStorage const &) -> ShaderObjectStorage & = delete;

    ShaderObjectStorage(ShaderObjectStorage &&other) noexcept;

    auto operator=(ShaderObjectStorage &&other) noexcept -> ShaderObjectStorage &;

    [[nodiscard]]
    static auto create(VulkanContext &context, ShaderObjectStorageCreateInfo const &create_info)
            -> std::expected<ShaderObjectStorage, ShaderObjectStorageError>;

    [[nodiscard]]
    auto create_linked(ShaderObjectCreateInfo const &create_info)
            -> std::expected<ShaderObjectHandle, ShaderObjectStorageError>;

    [[nodiscard]]
    auto create_compute(ComputeShaderCreateInfo const &create_info)
            -> std::expected<ShaderObjectHandle, ShaderObjectStorageError>;

    [[nodiscard]]
    auto destroy_shader_object(ShaderObjectHandle handle) -> std::expected<void, ShaderObjectStorageError>;

    [[nodiscard]]
    auto get(ShaderObjectHandle handle) noexcept -> ShaderObjectSet *;

    [[nodiscard]]
    auto get(ShaderObjectHandle handle) const noexcept -> ShaderObjectSet const *;

    [[nodiscard]]
    auto contains(ShaderObjectHandle handle) const noexcept -> bool {
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

    auto destroy() noexcept -> void;

private:
    VulkanContext *context_ = nullptr;

    ObjectPool<ShaderObjectSet> slots_;

    // Guards slots_.allocate()/release() -- see the identical comment on
    // PipelineStorage::slot_mutex_. ShaderObjectSet creation has no shared
    // VkPipelineCache-equivalent (confirmed: neither create_linked nor
    // create_compute touch anything but per-call state and the VkDevice
    // itself), so this is the only synchronization ShaderObjectStorage needs
    // for concurrent create_linked/create_compute.
    std::mutex slot_mutex_;

    VkDescriptorSetLayout global_descriptor_set_layout_ = VK_NULL_HANDLE;
    std::string debug_name_;
};
