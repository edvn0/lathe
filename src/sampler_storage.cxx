#include "sampler_storage.hxx"

#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "context.hxx"

namespace {

    auto make_error(SamplerStorageErrorType type, VkResult result = VK_SUCCESS) noexcept -> SamplerStorageError {
        return SamplerStorageError{
                .type = type,
                .result = result,
        };
    }

    template<typename Handle>
    auto object_handle(Handle handle) noexcept -> std::uint64_t {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<std::uint64_t>(handle);
        } else {
            return static_cast<std::uint64_t>(handle);
        }
    }

    auto set_object_name(VkDevice device, VkObjectType object_type, std::uint64_t handle,
                         std::string_view name) noexcept -> void {
        if (device == VK_NULL_HANDLE || handle == 0 || name.empty() || vkSetDebugUtilsObjectNameEXT == nullptr) {
            return;
        }

        auto null_terminated_name = std::string{name};

        VkDebugUtilsObjectNameInfoEXT const info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = object_type,
                .objectHandle = handle,
                .pObjectName = null_terminated_name.c_str(),
        };

        static_cast<void>(vkSetDebugUtilsObjectNameEXT(device, &info));
    }

    auto bump_revision(std::uint64_t &revision) noexcept -> void {
        ++revision;

        if (revision == 0) {
            revision = 1;
        }
    }

    struct DefaultSamplerInfo {
        SamplerCreateInfo create_info;
        SamplerClass sampler_class = SamplerClass::regular;
    };

    auto default_sampler_infos() -> std::array<DefaultSamplerInfo, default_sampler_count> {
        return {
                DefaultSamplerInfo{
                        .create_info =
                                SamplerCreateInfo{
                                        .mag_filter = VK_FILTER_LINEAR,
                                        .min_filter = VK_FILTER_LINEAR,
                                        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                        .address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .mip_lod_bias = 0.0F,
                                        .max_anisotropy = 1.0F,
                                        .compare_op = VK_COMPARE_OP_NEVER,
                                        .min_lod = 0.0F,
                                        .max_lod = VK_LOD_CLAMP_NONE,
                                        .sampler_class = SamplerClass::regular,
                                        .debug_name = "linear_repeat",
                                },
                        .sampler_class = SamplerClass::regular,
                },
                DefaultSamplerInfo{
                        .create_info =
                                SamplerCreateInfo{
                                        .mag_filter = VK_FILTER_LINEAR,
                                        .min_filter = VK_FILTER_LINEAR,
                                        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                        .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .mip_lod_bias = 0.0F,
                                        .max_anisotropy = 1.0F,
                                        .compare_op = VK_COMPARE_OP_NEVER,
                                        .min_lod = 0.0F,
                                        .max_lod = VK_LOD_CLAMP_NONE,
                                        .sampler_class = SamplerClass::regular,
                                        .debug_name = "linear_clamp",
                                },
                        .sampler_class = SamplerClass::regular,
                },
                DefaultSamplerInfo{
                        .create_info =
                                SamplerCreateInfo{
                                        .mag_filter = VK_FILTER_NEAREST,
                                        .min_filter = VK_FILTER_NEAREST,
                                        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        .address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        .mip_lod_bias = 0.0F,
                                        .max_anisotropy = 1.0F,
                                        .compare_op = VK_COMPARE_OP_NEVER,
                                        .min_lod = 0.0F,
                                        .max_lod = VK_LOD_CLAMP_NONE,
                                        .sampler_class = SamplerClass::regular,
                                        .debug_name = "nearest_repeat",
                                },
                        .sampler_class = SamplerClass::regular,
                },
                DefaultSamplerInfo{
                        .create_info =
                                SamplerCreateInfo{
                                        .mag_filter = VK_FILTER_NEAREST,
                                        .min_filter = VK_FILTER_NEAREST,
                                        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                        .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                        .mip_lod_bias = 0.0F,
                                        .max_anisotropy = 1.0F,
                                        .compare_op = VK_COMPARE_OP_NEVER,
                                        .min_lod = 0.0F,
                                        .max_lod = VK_LOD_CLAMP_NONE,
                                        .sampler_class = SamplerClass::regular,
                                        .debug_name = "nearest_clamp",
                                },
                        .sampler_class = SamplerClass::regular,
                },
                DefaultSamplerInfo{
                        .create_info =
                                SamplerCreateInfo{
                                        .mag_filter = VK_FILTER_LINEAR,
                                        .min_filter = VK_FILTER_LINEAR,
                                        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                        .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                        .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                        .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                                        .mip_lod_bias = 0.0F,
                                        .max_anisotropy = 1.0F,
                                        .compare_op = VK_COMPARE_OP_GREATER_OR_EQUAL,
                                        .min_lod = 0.0F,
                                        .max_lod = VK_LOD_CLAMP_NONE,
                                        .sampler_class = SamplerClass::comparison,
                                        .debug_name = "shadow_compare",
                                },
                        .sampler_class = SamplerClass::comparison,
                },
        };
    }

    auto create_vk_sampler(VulkanContext &context, SamplerCreateInfo const &create_info)
            -> std::expected<VkSampler, SamplerStorageError> {
        if (context.device == VK_NULL_HANDLE || create_info.max_anisotropy < 1.0F ||
            create_info.min_lod > create_info.max_lod) {
            return std::unexpected(make_error(SamplerStorageErrorType::invalid_argument));
        }

        auto const comparison = create_info.sampler_class == SamplerClass::comparison;

        VkSamplerCreateInfo const sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .magFilter = create_info.mag_filter,
                .minFilter = create_info.min_filter,
                .mipmapMode = create_info.mipmap_mode,
                .addressModeU = create_info.address_mode_u,
                .addressModeV = create_info.address_mode_v,
                .addressModeW = create_info.address_mode_w,
                .mipLodBias = create_info.mip_lod_bias,
                .anisotropyEnable = create_info.max_anisotropy > 1.0F ? VK_TRUE : VK_FALSE,
                .maxAnisotropy = create_info.max_anisotropy,
                .compareEnable = comparison ? VK_TRUE : VK_FALSE,
                .compareOp = comparison ? create_info.compare_op : VK_COMPARE_OP_NEVER,
                .minLod = create_info.min_lod,
                .maxLod = create_info.max_lod,
                .borderColor =
                        comparison ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
                .unnormalizedCoordinates = VK_FALSE,
        };

        auto sampler = VkSampler{VK_NULL_HANDLE};

        auto const result = vkCreateSampler(context.device, &sampler_info, nullptr, &sampler);

        if (result != VK_SUCCESS) {
            return std::unexpected(make_error(SamplerStorageErrorType::sampler_creation_failed, result));
        }

        set_object_name(context.device, VK_OBJECT_TYPE_SAMPLER, object_handle(sampler), create_info.debug_name);

        return sampler;
    }

} // namespace

SamplerStorage::~SamplerStorage() { destroy(); }

SamplerStorage::SamplerStorage(SamplerStorage &&other) noexcept :
    context_{std::exchange(other.context_, nullptr)}, slots_{std::move(other.slots_)},
    free_head_{std::exchange(other.free_head_, 0)}, capacity_{std::exchange(other.capacity_, 0)},
    debug_name_{std::move(other.debug_name_)} {}

auto SamplerStorage::operator=(SamplerStorage &&other) noexcept -> SamplerStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

    free_head_ = std::exchange(other.free_head_, 0);

    capacity_ = std::exchange(other.capacity_, 0);

    debug_name_ = std::move(other.debug_name_);

    return *this;
}

auto SamplerStorage::create(VulkanContext &context, std::uint32_t capacity, std::string_view debug_name)
        -> std::expected<SamplerStorage, SamplerStorageError> {
    if (context.device == VK_NULL_HANDLE || capacity < default_sampler_count) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_argument));
    }

    SamplerStorage storage;

    storage.context_ = &context;
    storage.capacity_ = capacity;
    storage.debug_name_ = std::string{debug_name};

    storage.slots_.resize(capacity);

    auto const defaults = default_sampler_infos();

    for (std::uint32_t index = 0; index < default_sampler_count; ++index) {
        auto create_info = defaults[index].create_info;

        auto const full_name = storage.debug_name_ + ".default." + std::string{create_info.debug_name};

        create_info.debug_name = full_name;

        auto sampler = create_vk_sampler(context, create_info);

        if (!sampler) {
            storage.destroy();

            return std::unexpected(sampler.error());
        }

        auto &slot = storage.slots_[index];

        slot.sampler = *sampler;
        slot.generation = 1;
        slot.next_free = 0;
        slot.descriptor_revision = 1;
        slot.sampler_class = defaults[index].sampler_class;
        slot.occupied = true;
        slot.protected_default = true;
    }

    if (capacity > default_sampler_count) {
        storage.free_head_ = default_sampler_count;

        for (std::uint32_t index = default_sampler_count; index < capacity; ++index) {
            storage.slots_[index].next_free = index + 1 < capacity ? index + 1 : 0;
        }
    }

    return storage;
}

auto SamplerStorage::create_sampler(SamplerCreateInfo const &create_info)
        -> std::expected<SamplerHandle, SamplerStorageError> {
    if (context_ == nullptr || context_->device == VK_NULL_HANDLE) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_argument));
    }

    if (free_head_ == 0) {
        return std::unexpected(make_error(SamplerStorageErrorType::capacity_exceeded));
    }

    auto sampler = create_vk_sampler(*context_, create_info);

    if (!sampler) {
        return std::unexpected(sampler.error());
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    free_head_ = slot.next_free;

    slot.sampler = *sampler;
    slot.next_free = 0;
    slot.sampler_class = create_info.sampler_class;
    slot.occupied = true;
    slot.protected_default = false;

    bump_revision(slot.descriptor_revision);

    return SamplerHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto SamplerStorage::descriptor_record(std::uint32_t index) const noexcept -> SamplerDescriptorRecord {
    if (index >= slots_.size()) {
        return {};
    }

    auto const &slot = slots_[index];

    return SamplerDescriptorRecord{
            .sampler = slot.occupied ? slot.sampler : VK_NULL_HANDLE,
            .revision = slot.descriptor_revision,
            .sampler_class = slot.sampler_class,
            .occupied = slot.occupied,
    };
}

auto SamplerStorage::destroy() noexcept -> void {
    if (context_ != nullptr && context_->device != VK_NULL_HANDLE) {
        for (auto &slot: slots_) {
            if (slot.sampler != VK_NULL_HANDLE) {
                vkDestroySampler(context_->device, slot.sampler, nullptr);
            }

            slot.sampler = VK_NULL_HANDLE;

            slot.occupied = false;
            slot.protected_default = false;
        }
    }

    slots_.clear();

    context_ = nullptr;
    free_head_ = 0;
    capacity_ = 0;

    debug_name_.clear();
}

auto SamplerStorage::destroy_sampler(SamplerHandle handle) -> std::expected<void, SamplerStorageError> {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_handle));
    }

    auto &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_handle));
    }

    if (slot.protected_default) {
        return std::unexpected(make_error(SamplerStorageErrorType::protected_default));
    }

    if (context_ != nullptr && context_->device != VK_NULL_HANDLE && slot.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(context_->device, slot.sampler, nullptr);
    }

    slot.sampler = VK_NULL_HANDLE;

    slot.occupied = false;
    slot.protected_default = false;

    ++slot.generation;

    if (slot.generation == 0) {
        slot.generation = 1;
    }

    bump_revision(slot.descriptor_revision);

    slot.next_free = free_head_;

    free_head_ = handle.index;

    return {};
}
