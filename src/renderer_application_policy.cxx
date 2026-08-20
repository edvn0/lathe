#include "renderer_application_policy.hxx"

#include "application.hxx"

void ApplicationOverlayPolicy::render_debug(Application const &app, VkCommandBuffer cmd, glm::mat4 const &vp,
                                            std::uint32_t frame_index) {
    app.debug_renderer->render(cmd, vp, frame_index);
}

void ApplicationOverlayPolicy::render_ui(Application const &app, VkCommandBuffer cmd, std::uint32_t frame_index) {
    app.imgui_renderer->render(cmd, frame_index);
}
