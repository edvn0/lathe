#pragma once

#include <array>
#include <cstdint>

#include <imgui.h>

#include "gpu/image.hxx"

struct Renderer;

namespace gui {

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

    class EditorIcons {
    public:
        explicit EditorIcons(Renderer &renderer);

        [[nodiscard]] auto texture(EditorIcon icon) const noexcept -> ImTextureID;

        // Icons are only crisp when drawn at a whole number of pixels, on a whole pixel: round the size, and floor
        // the offset that centres it in a row.
        [[nodiscard]] static auto snapped_size(float size) noexcept -> float;
        [[nodiscard]] static auto centred_offset(float available, float size) noexcept -> float;

    private:
        std::array<ImageHandle, static_cast<std::size_t>(EditorIcon::count)> textures_{};
    };

}
