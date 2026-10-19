#pragma once

struct ImFontLoader;

namespace gui {

    // An ImGui font loader that bakes every glyph as a signed distance field (FreeType's SDF renderer) instead of
    // coverage. The field is rasterised at a fixed `sdf_pixels_per_em` whatever size ImGui asks for, so a glyph looks
    // sharp at any scale, including the zoomed canvas of the frame graph editor, and the shader (gui.slang) turns the
    // distance into coverage per pixel. The field is in the alpha channel; the colour channels are white.
    inline constexpr float sdf_pixels_per_em = 48.0F;

    [[nodiscard]] auto sdf_font_loader() -> ImFontLoader const *;

}
