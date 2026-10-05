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

// Where the cursor is over the game's image, in normalised device coordinates: x right and y up, both in [-1, 1].
// Only games that return true from IGame::wants_cursor() get these.
struct CursorPositionEvent {
    double ndc_x{};
    double ndc_y{};

    // False when the cursor is outside the image the game is drawn into (over another editor panel, say).
    bool inside = false;
};
