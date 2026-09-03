#pragma once

class btIDebugDraw;

// Narrow surface of debug_draw::DebugRenderer that PhysicsWorld needs to
// feed it Bullet's collider wireframes. physics_world.hxx/.cxx depend on
// this instead of the full DebugRenderer -- DebugRenderer sits above
// physics in the module layering (it draws physics's colliders), so physics
// calling back into it directly would be a cycle.
struct IDebugLines {
    [[nodiscard]]
    virtual auto bullet_debug_draw() noexcept -> btIDebugDraw * = 0;

    virtual auto begin_frame() -> void = 0;
    virtual auto clear_lines() -> void = 0;

protected:
    ~IDebugLines() = default;
};
