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

        auto record(OverlayRecordContext const &context) -> void;

        auto clear_lines() -> void override;

        auto add_line(glm::vec3 const &from, glm::vec3 const &to, glm::vec3 const &colour) -> void;

        auto add_aabb(glm::vec3 const &min, glm::vec3 const &max, glm::vec3 const &colour) -> void;

        auto set_physics_debug_enabled(bool enabled) noexcept -> void;
        [[nodiscard]]
        auto physics_debug_enabled() const noexcept -> bool;

        auto set_model_bounds_debug_enabled(bool enabled) noexcept -> void;
        [[nodiscard]]
        auto model_bounds_debug_enabled() const noexcept -> bool;

        [[nodiscard]]
        auto bullet_debug_draw() noexcept -> btIDebugDraw * override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}
