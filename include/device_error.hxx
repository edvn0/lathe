#pragma once

#include <volk.h>

#include "fly_string.hxx"

#include <format>
#include <source_location>
#include <string_view>

struct DeviceError {
    enum class Type {
        Unknown,
        BufferCreation,
        AllocationFailure,
        AddressRetrieval,
    };

    Type type = Type::Unknown;
    FlyString message;
    VkResult vk_result = VK_SUCCESS;
    std::source_location location = std::source_location::current();

    static auto buffer_creation(const std::string_view message, const VkResult result = VK_SUCCESS,
                                const std::source_location location = std::source_location::current()) -> DeviceError {
        return DeviceError{
                .type = Type::BufferCreation,
                .message = FlyString{message},
                .vk_result = result,
                .location = location,
        };
    }
};

template<>
struct std::formatter<DeviceError::Type> : std::formatter<std::string_view> {
    constexpr auto format(DeviceError::Type error, std::format_context &context) const {
        auto const name = [&]() constexpr -> FlyString {
            switch (error) {
                case DeviceError::Type::Unknown:
                    return FlyString("Unknown");
                case DeviceError::Type::BufferCreation:
                    return FlyString("BufferCreation");
                case DeviceError::Type::AllocationFailure:
                    return FlyString("AllocationFailure");
                case DeviceError::Type::AddressRetrieval:
                    return FlyString("AddressRetrieval");
            }

            return FlyString("UnknownDeviceError");
        }();

        return std::formatter<std::string_view>::format(name.view(), context);
    }
};
