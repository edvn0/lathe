#include "gpu/sampler_storage.hxx"

#include <array>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "gpu/vk_object_name.hxx"

#include "gpu/context.hxx"

namespace {

    auto make_error(SamplerStorageErrorType type, std::string_view message = {}, VkResult result = VK_SUCCESS,
                    std::source_location location = std::source_location::current()) noexcept -> SamplerStorageError {
        return SamplerStorageError{
                .type = type,
                .context =
                        ErrorContext{
                                .message = FlyString{message},
                                .vk_result = result != VK_SUCCESS ? std::optional{result} : std::nullopt,
                                .location = location,
                        },
        };
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
                                        .max_anisotropy = 16.0F,
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
                                        .max_anisotropy = 16.0F,
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
                                        .max_anisotropy = 16.0F,
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
                                        .max_anisotropy = 16.0F,
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
                                        .max_anisotropy = 16.0F,
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
            return std::unexpected(
                    make_error(SamplerStorageErrorType::sampler_creation_failed,
                               std::format("vkCreateSampler failed for sampler '{}'", create_info.debug_name), result));
        }

        vk::set_object_name(context.device, VK_OBJECT_TYPE_SAMPLER, vk::object_handle(sampler), create_info.debug_name);

        return sampler;
    }

} // namespace

SamplerStorage::~SamplerStorage() { destroy(); }

SamplerStorage::SamplerStorage(SamplerStorage &&other) noexcept :
    context_{std::exchange(other.context_, nullptr)}, slots_{std::move(other.slots_)},
    debug_name_{std::move(other.debug_name_)} {}

auto SamplerStorage::operator=(SamplerStorage &&other) noexcept -> SamplerStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

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
    storage.debug_name_ = std::string{debug_name};

    storage.slots_ = ObjectPool<SamplerSlotData>::create(capacity);

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

        // The default samplers are allocated first, in order, out of a
        // freshly-created pool, so this always lands on index == `index`
        // with generation == 1 -- matching linear_repeat()/linear_clamp()/
        // etc.'s hard-coded handles below.
        auto &slot = storage.slots_.allocate()->second;

        slot.sampler = *sampler;
        slot.descriptor_revision = 1;
        slot.sampler_class = defaults[index].sampler_class;
        slot.protected_default = true;
    }

    return storage;
}

auto SamplerStorage::create_sampler(SamplerCreateInfo const &create_info)
        -> std::expected<SamplerHandle, SamplerStorageError> {
    if (context_ == nullptr || context_->device == VK_NULL_HANDLE) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_argument));
    }

    if (slots_.size() >= slots_.capacity()) {
        return std::unexpected(make_error(SamplerStorageErrorType::capacity_exceeded));
    }

    auto sampler = create_vk_sampler(*context_, create_info);

    if (!sampler) {
        return std::unexpected(sampler.error());
    }

    auto [handle, slot] = *slots_.allocate();

    slot.sampler = *sampler;
    slot.sampler_class = create_info.sampler_class;
    slot.protected_default = false;

    bump_revision(slot.descriptor_revision);

    return handle;
}

auto SamplerStorage::descriptor_record(std::uint32_t index) const noexcept -> SamplerDescriptorRecord {
    auto const *slot = slots_.get_at(index);

    if (slot == nullptr) {
        return {};
    }

    auto const occupied = slots_.occupied_at(index);

    return SamplerDescriptorRecord{
            .sampler = occupied ? slot->sampler : VK_NULL_HANDLE,
            .revision = slot->descriptor_revision,
            .sampler_class = slot->sampler_class,
            .occupied = occupied,
    };
}

auto SamplerStorage::destroy() noexcept -> void {
    if (context_ != nullptr && context_->device != VK_NULL_HANDLE) {
        for (std::uint32_t index = 0; index < slots_.capacity(); ++index) {
            auto const *slot = slots_.get_at(index);

            if (slot != nullptr && slot->sampler != VK_NULL_HANDLE) {
                vkDestroySampler(context_->device, slot->sampler, nullptr);
            }
        }
    }

    slots_ = ObjectPool<SamplerSlotData>{};

    context_ = nullptr;

    debug_name_.clear();
}

auto SamplerStorage::destroy_sampler(SamplerHandle handle) -> std::expected<void, SamplerStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(SamplerStorageErrorType::invalid_handle));
    }

    if (slot->protected_default) {
        return std::unexpected(make_error(SamplerStorageErrorType::protected_default));
    }

    if (context_ != nullptr && context_->device != VK_NULL_HANDLE && slot->sampler != VK_NULL_HANDLE) {
        vkDestroySampler(context_->device, slot->sampler, nullptr);
    }

    slot->sampler = VK_NULL_HANDLE;

    bump_revision(slot->descriptor_revision);

    static_cast<void>(slots_.release(handle));

    return {};
}
