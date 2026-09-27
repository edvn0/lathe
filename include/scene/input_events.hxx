#pragma once

#include <cstdint>

struct KeyPressedEvent {
    std::int32_t key{};
    std::int32_t modifiers{};
};

struct KeyReleasedEvent {
    std::int32_t key{};
    std::int32_t modifiers{};
};

// Cursor delta in pixels since the previous callback.
struct MouseMovedEvent {
    double delta_x{};
    double delta_y{};
};

struct MouseScrolledEvent {
    double delta_x{};
    double delta_y{};
};

struct MouseButtonPressedEvent {
    std::int32_t button{};
    std::int32_t modifiers{};
};

struct MouseButtonReleasedEvent {
    std::int32_t button{};
    std::int32_t modifiers{};
};
