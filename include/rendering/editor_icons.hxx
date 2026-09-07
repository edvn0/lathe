#pragma once

#include <array>
#include <cstdint>

#include <imgui.h>

#include "gpu/image.hxx"

struct Renderer;

namespace gui {

    // Every entry here needs a same-named PNG under assets/editor/icons/
    // (see EditorIcons::EditorIcons) -- keep this list and that directory in
    // sync.
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

    // Loads the small white-silhouette icons used by editor panels (e.g. the
    // Hierarchy widget) once, via the same DecodedImage -> ImageStorage
    // pipeline the light-bulb debug icon uses (see renderer.cxx). Icons are
    // plain white with alpha rather than pre-coloured, so callers recolour
    // them per entity type with ImGui::ImageWithBg's tint_col instead of
    // needing a separate texture per colour.
    class EditorIcons {
    public:
        explicit EditorIcons(Renderer &renderer);

        [[nodiscard]] auto texture(EditorIcon icon) const noexcept -> ImTextureID;

    private:
        std::array<ImageHandle, static_cast<std::size_t>(EditorIcon::count)> textures_{};
    };

} // namespace gui
