#pragma once

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

    [[nodiscard]] auto surface_frame(glm::vec3 const &position, glm::vec3 const &normal, glm::vec3 const &up,
                                     glm::vec3 const &fallback_up, float lift) -> Frame;

    // The local transform that puts an entity at `frame` in the world when it is a child of something at
    // `parent_world` (the identity for an entity with no parent).
    [[nodiscard]] auto local_transform(glm::mat4 const &parent_world, Frame const &frame) -> Components::Transform;

}

// The editor's tool for placing text on the surface under the mouse. While it is active, hovering a model previews the
// text there and a click adds a Text entity as a child of the model's entity, so it follows the model.
class TextPlacement {
public:
    struct Settings {
        std::string text = "Text";
        float size = 0.5F;
        float lift = 0.01F;
    };

    struct Ray {
        glm::vec3 origin{0.0F};
        // Unit.
        glm::vec3 direction{0.0F, 0.0F, 1.0F};
    };

    Settings settings;

    [[nodiscard]] auto active() const noexcept -> bool { return active_; }
    auto set_active(bool active) noexcept -> void { active_ = active; }

    // Previews the placement under `ray` and, when `click` is set, places the text. Returns the new entity, if one.
    // `camera_forward` orients text on floors and ceilings.
    auto update(Scene &scene, Renderer &renderer, debug_draw::DebugRenderer &lines, Ray const &ray,
                glm::vec3 const &camera_forward, bool click) -> entt::entity;

private:
    bool active_ = false;
};
