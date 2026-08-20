#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <volk.h>

#include "forward.hxx"

struct ApplicationOverlayPolicy {
    static void render_debug(Application const &app, VkCommandBuffer cmd, glm::mat4 const &vp,
                             std::uint32_t frame_index);
    static void render_ui(Application const &app, VkCommandBuffer cmd, std::uint32_t frame_index);
};
