#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "forward.hxx"
#include "image_storage.hxx"
#include "sampler_storage.hxx"
/*
enum class GpuResourceBinding :
    std::uint32_t {
    sampled_2d = 0,
    sampled_cube = 1,
    sampled_2d_array = 2,
    storage_2d = 3,
    storage_2d_array = 4,
    samplers = 5,
    comparison_samplers = 6,
};
 */

enum class GpuResourceBinding : std::uint32_t {
    sampled_2d = 0,
    samplers = 1,
    comparison_samplers = 2,
};

enum class GpuResourceTableErrorType : std::uint8_t {
    invalid_argument,
    capacity_exceeded,
    descriptor_layout_creation_failed,
    descriptor_pool_creation_failed,
    descriptor_set_allocation_failed,
};

struct GpuResourceTableError {
    GpuResourceTableErrorType type = GpuResourceTableErrorType::invalid_argument;

    VkResult result = VK_SUCCESS;
};

struct GpuResourceTableCreateInfo {
    std::uint32_t frames_in_flight = 0;

    std::uint32_t image_capacity = 4096;
    std::uint32_t sampler_capacity = 64;

    std::string_view debug_name = "gpu_resource_table";
};

class GpuResourceTable {
public:
    GpuResourceTable() = default;
    ~GpuResourceTable();

    GpuResourceTable(GpuResourceTable const &) = delete;

    auto operator=(GpuResourceTable const &) -> GpuResourceTable & = delete;

    GpuResourceTable(GpuResourceTable &&other) noexcept;

    auto operator=(GpuResourceTable &&other) noexcept -> GpuResourceTable &;

    [[nodiscard]]
    static auto create(VulkanContext &context, GpuResourceTableCreateInfo const &create_info)
            -> std::expected<GpuResourceTable, GpuResourceTableError>;

    /*
     * Call only after the frame fence associated with
     * frame_index has completed.
     */
    [[nodiscard]]
    auto prepare_frame(std::uint32_t frame_index, ImageStorage const &images, SamplerStorage const &samplers)
            -> std::expected<void, GpuResourceTableError>;

    auto bind(VkCommandBuffer command_buffer, std::uint32_t frame_index, VkPipelineBindPoint bind_point,
              VkPipelineLayout pipeline_layout) const noexcept -> void;

    [[nodiscard]]
    auto layout() const noexcept -> VkDescriptorSetLayout {
        return layout_;
    }

    [[nodiscard]]
    auto descriptor_set(std::uint32_t frame_index) const noexcept -> VkDescriptorSet;

    [[nodiscard]]
    auto image_capacity() const noexcept -> std::uint32_t {
        return image_capacity_;
    }

    [[nodiscard]]
    auto sampler_capacity() const noexcept -> std::uint32_t {
        return sampler_capacity_;
    }

    auto destroy() noexcept -> void;

private:
    struct FrameState {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        std::vector<std::uint64_t> image_revisions;

        std::vector<std::uint64_t> sampler_revisions;
    };

    VulkanContext *context_ = nullptr;

    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;

    std::vector<FrameState> frames_;

    std::uint32_t image_capacity_ = 0;
    std::uint32_t sampler_capacity_ = 0;

    std::string debug_name_;
};
