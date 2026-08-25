#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include <glm/fwd.hpp>

class Renderer;
class btIDebugDraw;

namespace debug_draw {

class DebugRenderer final {
public:
    explicit DebugRenderer(Renderer &renderer);
    ~DebugRenderer();

    DebugRenderer(DebugRenderer const &) = delete;
    auto operator=(DebugRenderer const &) -> DebugRenderer & = delete;

    DebugRenderer(DebugRenderer &&) noexcept;
    auto operator=(DebugRenderer &&) noexcept -> DebugRenderer &;

    auto begin_frame() -> void;

    auto render(
        VkCommandBuffer cmd,
        glm::mat4 const &view_projection,
        std::uint32_t frame_index
    ) -> void;

    auto render(
        VkCommandBuffer cmd,
        std::span<const float, 16> view_projection,
        std::uint32_t frame_index
    ) -> void;

    auto clear_lines() -> void;

    [[nodiscard]]
    auto bullet_debug_draw() noexcept -> btIDebugDraw *;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace debug_draw