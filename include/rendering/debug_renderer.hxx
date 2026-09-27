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

        // OverlayStage::scene record() callback: uploads this frame's lines
        // into frame_index's host-visible buffer and draws them depth-tested
        // against the scene. Clears the add_line()/add_aabb() list.
        auto record(OverlayRecordContext const &context) -> void;

        auto clear_lines() -> void override;

        // Appends one line/box to record(), independent of the Bullet-driven
        // physics wireframes above -- its own line list, consumed and
        // cleared by record() every call, so callers just add lines
        // sometime between one record() and the next (e.g. submit_scene()
        // in main.cxx, which runs once per rendered frame regardless of
        // Application::is_playing/PhysicsWorld::step's own cadence).
        auto add_line(glm::vec3 const &from, glm::vec3 const &to, glm::vec3 const &colour) -> void;

        // Draws the 12 edges of a world-space axis-aligned box.
        auto add_aabb(glm::vec3 const &min, glm::vec3 const &max, glm::vec3 const &colour) -> void;

        // Physics collider wireframes drawn via PhysicsWorld's Bullet debug
        // drawer -- disabled by default, opt in via Application::on_ui().
        auto set_physics_debug_enabled(bool enabled) noexcept -> void;
        [[nodiscard]]
        auto physics_debug_enabled() const noexcept -> bool;

        // Render-cull AABB wireframes: one box per (mesh, submesh) drawn by
        // every Components::Model this frame, sourced from
        // Renderer::model_submesh_bounds() -- see submit_scene() in
        // main.cxx. Disabled by default, opt in via Application::on_ui(),
        // same pattern as set_physics_debug_enabled above.
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
