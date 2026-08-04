#pragma once

#include <cstdint>
#include <glm/glm.hpp>

// Free-look editor camera: WASD(+QE) movement, right-mouse-drag look,
// scroll to adjust move speed. Pure value type -- no GPU/window
// resources -- so it can be owned directly by Application and fed
// from GLFW callbacks / your event structs.
struct EditorCameraCreateInfo {
    glm::vec3 position{0.0F, 2.0F, 6.0F};
    float yaw_degrees = -90.0F; // -90 faces -Z with the basis below; see rebuild_basis().
    float pitch_degrees = 0.0F;

    float field_of_view_degrees = 60.0F;
    float near_clip = 0.1F;
    float far_clip = 10000.0F;

    float move_speed = 5.0F;
    float sprint_multiplier = 4.0F;
    float look_sensitivity = 0.12F; // degrees per pixel of mouse delta
    float min_move_speed = 0.5F;
    float max_move_speed = 100.0F;
};

class EditorCamera {
public:
    EditorCamera() noexcept = default;

    explicit EditorCamera(EditorCameraCreateInfo const &create_info) noexcept;

    auto on_key_pressed(std::int32_t key) noexcept -> void;
    auto on_key_released(std::int32_t key) noexcept -> void;

    // delta_x/delta_y are raw cursor deltas in pixels since the previous
    // call. Rotation is only integrated while `dragging` is true (e.g.
    // right mouse button held) -- callers still call this every frame so
    // the internal "last cursor position" bookkeeping stays correct, but
    // pass dragging=false to just track position without turning.
    auto on_mouse_moved(float delta_x, float delta_y, bool dragging) noexcept -> void;

    // Adjusts move_speed_ rather than FOV -- scroll-to-zoom on an editor
    // camera usually means "walk faster/slower", not a lens change.
    auto on_mouse_scrolled(float delta_y) noexcept -> void;

    auto set_sprinting(bool sprinting) noexcept -> void;

    auto update(float delta_time_seconds) noexcept -> void;

    [[nodiscard]] auto view() const noexcept -> glm::mat4;
    [[nodiscard]] auto projection(float aspect_ratio) const noexcept -> glm::mat4;
    [[nodiscard]] auto view_projection(float aspect_ratio) const noexcept -> glm::mat4;

    [[nodiscard]] auto position() const noexcept -> glm::vec3 { return position_; }
    [[nodiscard]] auto forward() const noexcept -> glm::vec3 { return forward_; }
    [[nodiscard]] auto move_speed() const noexcept -> float { return move_speed_; }

    [[nodiscard]] auto near_clip() const noexcept -> float { return near_clip_; }
    [[nodiscard]] auto far_clip() const noexcept -> float { return far_clip_; }
    [[nodiscard]] auto field_of_view_degrees() const noexcept -> float { return field_of_view_degrees_; }

    auto set_position(glm::vec3 const &position) noexcept -> void { position_ = position; }

private:
    auto rebuild_basis() noexcept -> void;

    glm::vec3 position_{0.0F, 2.0F, 6.0F};
    glm::vec3 forward_{0.0F, 0.0F, -1.0F};
    glm::vec3 right_{1.0F, 0.0F, 0.0F};
    glm::vec3 up_{0.0F, 1.0F, 0.0F};

    float yaw_degrees_ = -90.0F;
    float pitch_degrees_ = 0.0F;

    float field_of_view_degrees_ = 60.0F;
    float near_clip_ = 0.1F;
    float far_clip_ = 10000.0F;

    float move_speed_ = 5.0F;
    float sprint_multiplier_ = 4.0F;
    float look_sensitivity_ = 0.12F;
    float min_move_speed_ = 0.5F;
    float max_move_speed_ = 100.0F;

    bool moving_forward_ = false;
    bool moving_backward_ = false;
    bool moving_left_ = false;
    bool moving_right_ = false;
    bool moving_up_ = false;
    bool moving_down_ = false;
    bool sprinting_ = false;
};