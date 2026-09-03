#pragma once

#include <volk.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace vk {

    template<typename Handle>
    auto object_handle(Handle handle) noexcept -> std::uint64_t {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<std::uint64_t>(handle);
        } else {
            return static_cast<std::uint64_t>(handle);
        }
    }

    inline auto set_object_name(VkDevice device, VkObjectType object_type, std::uint64_t handle,
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

} // namespace vk
