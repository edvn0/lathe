#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "core/transform.hxx"

class Scene;
class Renderer;
namespace debug_draw {
    class DebugRenderer;
}

namespace text_placement {

    // Where world text sits on a surface: lifted off it along the normal, its back to the surface, reading along the
    // surface with its up as close to `up` as the surface allows. On a surface facing along `up` (a floor or ceiling)
    // `fallback_up` decides, so the text does not spin with the last bit of noise in the normal.
    struct Frame {
        glm::vec3 position{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    };

    // `roll_radians` turns the text about the surface normal, its right toward its up for a positive angle.
    [[nodiscard]] auto surface_frame(glm::vec3 const &position, glm::vec3 const &normal, glm::vec3 const &up,
                                     glm::vec3 const &fallback_up, float lift, float roll_radians = 0.0F) -> Frame;

    // The point on the surface (the plane through `on_surface` with `normal`) nearest the world-grid point nearest
    // `on_surface`: on a wall facing along an axis, the position rounded in the wall's own two directions.
    [[nodiscard]] auto snap_to_grid(glm::vec3 const &on_surface, glm::vec3 const &normal, float step) -> glm::vec3;

    // `degrees` rounded to a multiple of `step`, in (-180, 180].
    [[nodiscard]] auto snap_angle(float degrees, float step) -> float;

    // The local transform that puts an entity at `frame` in the world when it is a child of something at
    // `parent_world` (the identity for an entity with no parent).
    [[nodiscard]] auto local_transform(glm::mat4 const &parent_world, Frame const &frame) -> Components::Transform;

    enum class SnapKind : std::uint8_t { vertex, edge_midpoint, face_centre };

    struct Snap {
        glm::vec3 position{0.0F};
        SnapKind kind = SnapKind::face_centre;
    };

    // Maps a world point to a pixel, as the editor camera draws it (left-handed, +z forward, y up on screen).
    struct ScreenProjector {
        glm::mat4 view{1.0F};
        // The projection's focal terms: projection[0][0] and projection[1][1].
        glm::vec2 focal{1.0F};
        glm::vec2 origin{0.0F};
        glm::vec2 size{1.0F};

        // Nothing for a point behind the camera.
        [[nodiscard]] auto project(glm::vec3 const &world) const -> std::optional<glm::vec2>;
    };

    // The vertex, edge midpoint or face centre of `triangle` nearest `cursor` in pixels, when one is within `radius`
    // pixels. A vertex beats an edge midpoint and an edge midpoint beats the centre when they are both in reach, so the
    // finer targets stay reachable on a coarse triangle.
    [[nodiscard]] auto snap_to_triangle(std::array<glm::vec3, 3> const &triangle, ScreenProjector const &projector,
                                        glm::vec2 const &cursor, float radius) -> std::optional<Snap>;

}

// The editor's tool for placing text on the surface under the mouse. While it is active, hovering a model previews the
// text there and a click adds a Text entity as a child of the model's entity, so it follows the model.
class TextPlacement {
public:
    struct Settings {
        std::string text = "Text";
        float size = 0.5F;
        float lift = 0.01F;
        // Snap to the hit triangle's vertices, edge midpoints and centre when the cursor is this close, in pixels.
        bool snap_to_geometry = true;
        float snap_radius = 16.0F;
        // The grid for snapping to it, in world units, and the step of the roll and of angle snapping, in degrees.
        float grid_step = 0.25F;
        float angle_step = 15.0F;
        // The text's turn about the surface normal, in degrees.
        float roll_degrees = 0.0F;
    };

    // What the tool needs to know about the mouse and camera this frame.
    struct Pointer {
        // The ray under the mouse; the direction is unit.
        glm::vec3 origin{0.0F};
        glm::vec3 direction{0.0F, 0.0F, 1.0F};
        glm::vec3 camera_forward{0.0F, 0.0F, 1.0F};
        text_placement::ScreenProjector projector;
        glm::vec2 cursor{0.0F};
        bool click = false;
        // Held to snap to the grid and the roll to the angle step, instead of to the triangle's points.
        bool snap_to_grid = false;
        // Held to place without snapping.
        bool bypass_snap = false;
    };

    Settings settings;

    // Turns the roll by one angle step.
    auto step_roll(bool reverse) noexcept -> void;

    [[nodiscard]] auto active() const noexcept -> bool { return active_; }
    auto set_active(bool active) noexcept -> void { active_ = active; }

    // Previews the placement under the pointer and, when it clicked, places the text. Returns the new entity, if one.
    auto update(Scene &scene, Renderer &renderer, debug_draw::DebugRenderer &lines, Pointer const &pointer)
            -> entt::entity;

private:
    bool active_ = false;
};
