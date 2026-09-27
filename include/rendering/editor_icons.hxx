#pragma once

#include <array>
#include <cstdint>

#include <imgui.h>

#include "gpu/image.hxx"

struct Renderer;

namespace gui {

    // Each entry needs a same-named PNG in assets/editor/icons/.
    enum class EditorIcon : std::uint8_t {
        mesh,
        point_light,
        spot_light,
        script,
        player,
        bullet,
        empty,
        folder,
        search,
        move,
        rotate,
        scale,
        local,
        world,
        count,
    };

    // White editor icons with alpha, loaded once. Callers tint them per use.
    class EditorIcons {
    public:
        explicit EditorIcons(Renderer &renderer);

        [[nodiscard]] auto texture(EditorIcon icon) const noexcept -> ImTextureID;

    private:
        std::array<ImageHandle, static_cast<std::size_t>(EditorIcon::count)> textures_{};
    };

} // namespace gui
