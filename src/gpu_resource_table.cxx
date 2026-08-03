#include "gpu_resource_table.hxx"

#include <array>
#include <format>
#include <source_location>
#include <string_view>
#include <utility>

#include "context.hxx"

namespace {

    auto make_error(GpuResourceTableErrorType type, std::string_view message = {}, VkResult result = VK_SUCCESS,
                    std::source_location location = std::source_location::current()) noexcept -> GpuResourceTableError {
        return {
                .type = type,
                .context =
                        ErrorContext{
                                .message = FlyString{message},
                                .vk_result = result != VK_SUCCESS ? std::optional{result} : std::nullopt,
                                .location = location,
                        },
        };
    }

    constexpr auto binding_index(GpuResourceBinding binding) noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(binding);
    }

} // namespace

GpuResourceTable::~GpuResourceTable() { destroy(); }

auto GpuResourceTable::create(VulkanContext &context, GpuResourceTableCreateInfo const &create_info)
        -> std::expected<GpuResourceTable, GpuResourceTableError> {
    if (context.device == VK_NULL_HANDLE || create_info.frames_in_flight == 0 || create_info.image_capacity == 0 ||
        create_info.sampler_capacity == 0) {
        return std::unexpected(make_error(GpuResourceTableErrorType::invalid_argument));
    }

    GpuResourceTable table;

    table.context_ = &context;

    table.image_capacity_ = create_info.image_capacity;

    table.sampler_capacity_ = create_info.sampler_capacity;

    table.debug_name_ = std::string{create_info.debug_name};

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{
            VkDescriptorSetLayoutBinding{
                    .binding = binding_index(GpuResourceBinding::sampled_2d),
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .descriptorCount = table.image_capacity_,
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                    .binding = binding_index(GpuResourceBinding::samplers),
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .descriptorCount = table.sampler_capacity_,
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .pImmutableSamplers = nullptr,
            },
            VkDescriptorSetLayoutBinding{
                    .binding = binding_index(GpuResourceBinding::comparison_samplers),
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .descriptorCount =

                            table.sampler_capacity_,
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .pImmutableSamplers = nullptr,
            },
    };

    /*
     * Descriptor arrays are fixed-capacity but can be
     * sparsely populated. We fill missing entries with
     * defaults, so PARTIALLY_BOUND is not required for
     * correctness, but remains useful while bringing the
     * table online.
     */
    VkDescriptorSetLayoutCreateInfo const layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = static_cast<std::uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
    };

    auto result = vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &table.layout_);

    if (result != VK_SUCCESS) {
        table.destroy();

        return std::unexpected(make_error(GpuResourceTableErrorType::descriptor_layout_creation_failed,
                                          "vkCreateDescriptorSetLayout failed", result));
    }

    auto const frame_count = create_info.frames_in_flight;

    std::array<VkDescriptorPoolSize, 3> pool_sizes{
            VkDescriptorPoolSize{
                    .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .descriptorCount = table.image_capacity_ * 3 * frame_count,
            },
            VkDescriptorPoolSize{
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = table.image_capacity_ * 2 * frame_count,
            },
            VkDescriptorPoolSize{
                    .type = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .descriptorCount = table.sampler_capacity_ * 2 * frame_count,
            },
    };

    VkDescriptorPoolCreateInfo const pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = frame_count,
            .poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
    };

    result = vkCreateDescriptorPool(context.device, &pool_info, nullptr, &table.pool_);

    if (result != VK_SUCCESS) {
        table.destroy();

        return std::unexpected(
                make_error(GpuResourceTableErrorType::descriptor_pool_creation_failed, "vkCreateDescriptorPool failed",
                          result));
    }

    auto layouts = std::vector<VkDescriptorSetLayout>(frame_count, table.layout_);

    auto descriptor_sets = std::vector<VkDescriptorSet>(frame_count, VK_NULL_HANDLE);

    VkDescriptorSetAllocateInfo const allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = table.pool_,
            .descriptorSetCount = frame_count,
            .pSetLayouts = layouts.data(),
    };

    result = vkAllocateDescriptorSets(context.device, &allocate_info, descriptor_sets.data());

    if (result != VK_SUCCESS) {
        table.destroy();

        return std::unexpected(make_error(GpuResourceTableErrorType::descriptor_set_allocation_failed,
                                          "vkAllocateDescriptorSets failed", result));
    }

    table.frames_.resize(frame_count);

    for (std::uint32_t index = 0; index < frame_count; ++index) {
        auto &frame = table.frames_[index];

        frame.descriptor_set = descriptor_sets[index];

        frame.image_revisions.resize(table.image_capacity_, 0);

        frame.sampler_revisions.resize(table.sampler_capacity_, 0);
    }

    return table;
}

auto GpuResourceTable::prepare_frame(std::uint32_t frame_index, ImageStorage const &images,
                                     SamplerStorage const &samplers) -> std::expected<void, GpuResourceTableError> {
    if (context_ == nullptr || frame_index >= frames_.size() || images.capacity() > image_capacity_ ||
        samplers.capacity() > sampler_capacity_) {
        return std::unexpected(make_error(images.capacity() > image_capacity_ || samplers.capacity() > sampler_capacity_
                                                  ? GpuResourceTableErrorType::capacity_exceeded
                                                  : GpuResourceTableErrorType::invalid_argument));
    }

    auto &frame = frames_[frame_index];

    auto image_infos = std::vector<VkDescriptorImageInfo>{};

    auto writes = std::vector<VkWriteDescriptorSet>{};

    /*
     * Reserve enough that pImageInfo pointers remain
     * stable until vkUpdateDescriptorSets().
     */
    image_infos.reserve(images.capacity() * 5 + samplers.capacity() * 2);

    writes.reserve(images.capacity() * 5 + samplers.capacity() * 2);

    auto append_image_write = [&](std::uint32_t binding, std::uint32_t array_index, VkDescriptorType descriptor_type,
                                  VkImageView view, VkImageLayout layout) {
        image_infos.push_back(VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = view,
                .imageLayout = layout,
        });

        writes.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.descriptor_set,
                .dstBinding = binding,
                .dstArrayElement = array_index,
                .descriptorCount = 1,
                .descriptorType = descriptor_type,
                .pImageInfo = &image_infos.back(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
        });
    };

    auto const fallback = images.descriptor_record(images.white().index);

    for (std::uint32_t index = 0; index < images.capacity(); ++index) {
        auto const record = images.descriptor_record(index);

        if (frame.image_revisions[index] == record.revision) {
            continue;
        }

        auto const sampled_2d = record.sampled_2d != VK_NULL_HANDLE ? record.sampled_2d : fallback.sampled_2d;

        append_image_write(binding_index(GpuResourceBinding::sampled_2d), index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                           sampled_2d, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        /*
         * A production implementation should use dedicated
         * fallback cube/array/storage images here. Until
         * those defaults exist, only write categories for
         * which a valid view exists.
         */

        /*
          if (record.sampled_cube != VK_NULL_HANDLE) {
            append_image_write(binding_index(GpuResourceBinding::sampled_cube),
          index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, record.sampled_cube,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
          }

          if (record.sampled_2d_array != VK_NULL_HANDLE) {
            append_image_write(binding_index(GpuResourceBinding::sampled_2d_array),
                               index, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                               record.sampled_2d_array,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
          }

          if (record.storage_2d != VK_NULL_HANDLE) {
            append_image_write(binding_index(GpuResourceBinding::storage_2d), index,
                               VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, record.storage_2d,
                               VK_IMAGE_LAYOUT_GENERAL);
          }

          if (record.storage_2d_array != VK_NULL_HANDLE) {
            append_image_write(binding_index(GpuResourceBinding::storage_2d_array),
                               index, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               record.storage_2d_array, VK_IMAGE_LAYOUT_GENERAL);
          }
          */

        frame.image_revisions[index] = record.revision;
    }

    auto const regular_fallback = samplers.descriptor_record(samplers.linear_repeat().index).sampler;

    auto const comparison_fallback = samplers.descriptor_record(samplers.shadow_compare().index).sampler;

    auto append_sampler_write = [&](std::uint32_t binding, std::uint32_t array_index, VkSampler sampler) {
        image_infos.push_back(VkDescriptorImageInfo{
                .sampler = sampler,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });

        writes.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.descriptor_set,
                .dstBinding = binding,
                .dstArrayElement = array_index,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .pImageInfo = &image_infos.back(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
        });
    };
    for (std::uint32_t index = 0; index < samplers.capacity(); ++index) {
        auto const record = samplers.descriptor_record(index);

        if (frame.sampler_revisions[index] == record.revision) {
            continue;
        }

        auto const regular_sampler =
                record.occupied && record.sampler_class == SamplerClass::regular ? record.sampler : regular_fallback;

        auto const comparison_sampler = record.occupied && record.sampler_class == SamplerClass::comparison
                                                ? record.sampler
                                                : comparison_fallback;

        append_sampler_write(binding_index(GpuResourceBinding::samplers), index, regular_sampler);

        append_sampler_write(binding_index(GpuResourceBinding::comparison_samplers), index, comparison_sampler);

        frame.sampler_revisions[index] = record.revision;
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(context_->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return {};
}

auto GpuResourceTable::bind(VkCommandBuffer command_buffer, std::uint32_t frame_index, VkPipelineBindPoint bind_point,
                            VkPipelineLayout pipeline_layout) const noexcept -> void {
    if (command_buffer == VK_NULL_HANDLE || pipeline_layout == VK_NULL_HANDLE || frame_index >= frames_.size()) {
        return;
    }

    auto const descriptor_set = frames_[frame_index].descriptor_set;

    vkCmdBindDescriptorSets(command_buffer, bind_point, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
}

GpuResourceTable::GpuResourceTable(GpuResourceTable &&other) noexcept :
    context_{std::exchange(other.context_, nullptr)}, layout_{std::exchange(other.layout_, VK_NULL_HANDLE)},
    pool_{std::exchange(other.pool_, VK_NULL_HANDLE)}, frames_{std::move(other.frames_)},
    image_capacity_{std::exchange(other.image_capacity_, 0)},
    sampler_capacity_{std::exchange(other.sampler_capacity_, 0)}, debug_name_{std::move(other.debug_name_)} {}

auto GpuResourceTable::operator=(GpuResourceTable &&other) noexcept -> GpuResourceTable & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);

    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);

    frames_ = std::move(other.frames_);

    image_capacity_ = std::exchange(other.image_capacity_, 0);

    sampler_capacity_ = std::exchange(other.sampler_capacity_, 0);

    debug_name_ = std::move(other.debug_name_);

    return *this;
}

auto GpuResourceTable::destroy() noexcept -> void {
    frames_.clear();

    if (context_ != nullptr && context_->device != VK_NULL_HANDLE) {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(context_->device, pool_, nullptr);
        }

        if (layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context_->device, layout_, nullptr);
        }
    }

    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;

    image_capacity_ = 0;
    sampler_capacity_ = 0;

    debug_name_.clear();

    context_ = nullptr;
}
