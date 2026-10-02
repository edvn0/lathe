#!/usr/bin/env python3
"""Generate the procedural terrain textures and the light-bulb icon.

Everything here is made from seeded noise and PIL drawing, so the output is reproducible and nothing third-party is
checked in. Run from anywhere:

    uv run --with pillow --with numpy --with OpenEXR python assets/textures/generate_textures.py

Writes next to this script:
    terrain/terrain_albedo.png     sRGB colour
    terrain/terrain_normal.exr     tangent-space normal, OpenGL (+Y up) convention, half float, ZIP
    terrain/terrain_roughness.png  roughness in G (glTF metallic-roughness layout), metallic 0
    light_bulb.png                 32x32 black outline, alpha-shaped
"""
import os

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 512
SEED = 20251002


def tileable_noise(rng, size, falloff):
    """Band-limited noise that wraps on both axes, from a power-law filtered FFT of white noise. Normalised to [0, 1]."""
    freq = np.fft.fftfreq(size)
    radius = np.hypot(*np.meshgrid(freq, freq))
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(rng.standard_normal((size, size))) / radius**falloff
    spectrum[0, 0] = 0.0
    field = np.fft.ifft2(spectrum).real
    return (field - field.min()) / (field.max() - field.min())


def lerp(a, b, t):
    return a + (b - a) * t[..., None]


def make_height(rng):
    broad = tileable_noise(rng, SIZE, 1.6)
    medium = tileable_noise(rng, SIZE, 1.1)
    grain = tileable_noise(rng, SIZE, 0.35)
    height = 0.55 * broad + 0.3 * medium + 0.15 * grain
    # Scattered pebbles: bumps where a noise field is in its top few percent.
    pebbles = tileable_noise(rng, SIZE, 0.6)
    height += 0.25 * np.clip((pebbles - 0.93) / 0.07, 0.0, 1.0)
    return height, broad, medium, grain


def write_albedo(rng, height, broad, medium, grain):
    dark = np.array([0.20, 0.13, 0.08])
    mid = np.array([0.38, 0.27, 0.17])
    light = np.array([0.52, 0.41, 0.28])
    base = lerp(np.broadcast_to(dark, (SIZE, SIZE, 3)), np.broadcast_to(mid, (SIZE, SIZE, 3)), broad)
    base = lerp(base, np.broadcast_to(light, (SIZE, SIZE, 3)), np.clip((medium - 0.4) * 1.6, 0.0, 1.0))
    base *= (0.85 + 0.3 * grain)[..., None]
    base *= (0.8 + 0.4 * height)[..., None]
    linear = np.clip(base, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    Image.fromarray((srgb * 255 + 0.5).astype(np.uint8), "RGB").save(os.path.join(OUT_DIR, "terrain", "terrain_albedo.png"))


def write_normal(height):
    strength = 6.0
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * strength
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 * strength
    # Image rows run top to bottom (-V), so +Y up (OpenGL) negates the row derivative.
    normal = np.stack([-dx, dy, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    encoded = (normal * 0.5 + 0.5).astype(np.float16)
    channels = {name: encoded[..., i] for i, name in enumerate("RGB")}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, channels) as exr:
        exr.write(os.path.join(OUT_DIR, "terrain", "terrain_normal.exr"))


def write_roughness(medium, grain):
    roughness = np.clip(0.78 + 0.2 * (medium - 0.5) + 0.08 * (grain - 0.5), 0.0, 1.0)
    value = (roughness * 255 + 0.5).astype(np.uint8)
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[..., 0] = value
    image[..., 1] = value
    Image.fromarray(image, "RGB").save(os.path.join(OUT_DIR, "terrain", "terrain_roughness.png"))


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
    os.makedirs(os.path.join(OUT_DIR, "terrain"), exist_ok=True)
    rng = np.random.default_rng(SEED)
    height, broad, medium, grain = make_height(rng)
    write_albedo(rng, height, broad, medium, grain)
    write_normal(height)
    write_roughness(medium, grain)
    write_light_bulb()


if __name__ == "__main__":
    main()
