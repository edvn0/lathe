#pragma once

#include <cstdint>
#include <string_view>

#include "core/event_bus.hxx"

struct KeyPressedEvent : Event<KeyPressedEvent> {
    static constexpr std::string_view name = "KeyPressed";

    std::int32_t key{};
    std::int32_t modifiers{};
};

struct KeyReleasedEvent : Event<KeyReleasedEvent> {
    static constexpr std::string_view name = "KeyReleased";

    std::int32_t key{};
    std::int32_t modifiers{};
};

struct MouseMovedEvent : Event<MouseMovedEvent> {
    static constexpr std::string_view name = "MouseMoved";

    double delta_x{};
    double delta_y{};
};

struct MouseScrolledEvent : Event<MouseScrolledEvent> {
    static constexpr std::string_view name = "MouseScrolled";

    double delta_x{};
    double delta_y{};
};

struct MouseButtonPressedEvent : Event<MouseButtonPressedEvent> {
    static constexpr std::string_view name = "MouseButtonPressed";

    std::int32_t button{};
    std::int32_t modifiers{};
};

struct MouseButtonReleasedEvent : Event<MouseButtonReleasedEvent> {
    static constexpr std::string_view name = "MouseButtonReleased";

    std::int32_t button{};
    std::int32_t modifiers{};
};

struct CursorPositionEvent {
    double ndc_x{};
    double ndc_y{};

    bool inside = false;
};
