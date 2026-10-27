#include "empty_game.hxx"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "rendering/scene.hxx"

auto EmptyGame::on_populate(Scene &scene, Renderer &, EngineModels const &) -> void {
    // Ctrl-R populates the same scene again: an empty game leaves nothing behind.
    scene.get_registry().clear();
}

auto EmptyGame::on_update(Scene &, float) -> void {}

auto EmptyGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    constexpr float near_clip = 0.1F;
    constexpr float far_clip = 1000.0F;

    return CameraParams{
            .view = glm::lookAtLH(glm::vec3{6.0F, 5.0F, -8.0F}, glm::vec3{0.0F}, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(60.0F), aspect_ratio, near_clip, far_clip),
            .near_clip = near_clip,
            .far_clip = far_clip,
            .vertical_fov_radians = glm::radians(60.0F),
    };
}
