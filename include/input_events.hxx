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

// Raw cursor delta in pixels since the previous callback -- not an
// absolute position. Application decides whether it counts as a
// "look" based on whether a drag is currently active.
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
