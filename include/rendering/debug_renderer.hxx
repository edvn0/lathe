#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include <glm/fwd.hpp>
#include <glm/vec3.hpp>

#include "physics/debug_lines.hxx"
#include "rendering/overlay.hxx"

class Renderer;

namespace debug_draw {

    class DebugRenderer final : public IDebugLines {
    public:
        explicit DebugRenderer(Renderer &renderer);
        ~DebugRenderer();

        DebugRenderer(DebugRenderer const &) = delete;
        auto operator=(DebugRenderer const &) -> DebugRenderer & = delete;

        DebugRenderer(DebugRenderer &&) noexcept;
        auto operator=(DebugRenderer &&) noexcept -> DebugRenderer &;

        auto begin_frame() -> void override;

        // Scene overlay: uploads this frame's lines and draws them depth-tested. Clears the add_line()/add_aabb()
        // list.
        auto record(OverlayRecordContext const &context) -> void;

        auto clear_lines() -> void override;

        // Queues a line for the next record(). Independent of the physics wireframes.
        auto add_line(glm::vec3 const &from, glm::vec3 const &to, glm::vec3 const &colour) -> void;

        // Draws the 12 edges of a world-space axis-aligned box.
        auto add_aabb(glm::vec3 const &min, glm::vec3 const &max, glm::vec3 const &colour) -> void;

        // Bullet collider wireframes. Off by default.
        auto set_physics_debug_enabled(bool enabled) noexcept -> void;
        [[nodiscard]]
        auto physics_debug_enabled() const noexcept -> bool;

        // One box per (mesh, submesh) drawn this frame, from Renderer::model_submesh_bounds(). Off by default.
        auto set_model_bounds_debug_enabled(bool enabled) noexcept -> void;
        [[nodiscard]]
        auto model_bounds_debug_enabled() const noexcept -> bool;

        [[nodiscard]]
        auto bullet_debug_draw() noexcept -> btIDebugDraw * override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace debug_draw
