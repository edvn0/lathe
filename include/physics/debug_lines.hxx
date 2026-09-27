#pragma once

class btIDebugDraw;

// The part of DebugRenderer PhysicsWorld needs for collider wireframes. DebugRenderer sits above physics, so
// depending on it directly would be a cycle.
struct IDebugLines {
    [[nodiscard]]
    virtual auto bullet_debug_draw() noexcept -> btIDebugDraw * = 0;

    virtual auto begin_frame() -> void = 0;
    virtual auto clear_lines() -> void = 0;

protected:
    ~IDebugLines() = default;
};
