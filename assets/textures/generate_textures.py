#!/usr/bin/env python3
"""Generate the light-bulb icon. Run from anywhere:

    uv run --with pillow python assets/textures/generate_textures.py

Writes next to this script:
    light_bulb.png                 32x32 black outline, alpha-shaped

The terrain textures in terrain/ are not generated: they are Poly Haven's "Grass Ground" (CC0), 2k. terrain_albedo.jpg
and terrain_normal.jpg (OpenGL +Y) are the diffuse and nor_gl maps as downloaded; terrain_roughness.png puts the
roughness map in R and G (glTF metallic-roughness layout, metallic 0).
"""
import os

from PIL import Image, ImageDraw

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_light_bulb():
    ss, size = 16, 32
    canvas = size * ss
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def px(v):
        return v * canvas / 24.0

    ink = (0, 0, 0, 255)
    width = round(1.6 * ss * size / 24)
    # Glass: a circle that runs into a short neck, then the base lines.
    draw.arc([px(5.5), px(2.5), px(18.5), px(15.5)], start=-215, end=35, fill=ink, width=width)
    draw.line([(px(8.9), px(13.2)), (px(9.4), px(16.5)), (px(14.6), px(16.5)), (px(15.1), px(13.2))], fill=ink, width=width, joint="curve")
    draw.line([(px(9.6), px(19.2)), (px(14.4), px(19.2))], fill=ink, width=width)
    draw.line([(px(10.6), px(21.8)), (px(13.4), px(21.8))], fill=ink, width=width)
    img.resize((size, size), Image.LANCZOS).save(os.path.join(OUT_DIR, "light_bulb.png"))


def main():
    write_light_bulb()


if __name__ == "__main__":
    main()
