#pragma once

#include <volk.h>

#include "fly_string.hxx"

#include <source_location>

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
