# Colour grading

Composition can run the tone-mapped image through a 3D colour lookup table (LUT) before the outline and dither steps.
The LUT maps display-referred colour (sRGB-encoded, 0 .. 1) to graded colour, so grading sits after ACES, not before it.

## Built-in LUTs

The renderer generates these at startup (`make_default_colour_lut` in `rendering/colour_lut.hxx`), 32^3 each, so no
files ship with the engine:

| name        | look                                                                 |
|-------------|----------------------------------------------------------------------|
| `cinematic` | teal shadows, orange highlights, a little more contrast and colour   |
| `vintage`   | lifted faded blacks, muted colour, warm cast                         |
| `noir`      | near-monochrome, strong contrast, slightly cool shadows              |

## Using them

- Editor: the **Colour grading** stage has a LUT picker and an intensity slider.
- Lua: `render.set_grade{ lut = "cinematic", intensity = 0.8 }`, `render.set_grade{ lut = false }` to turn it off, and
  `render.grade()` to read it back.
- C++: `Renderer::set_colour_grade_settings({.lut = *renderer.find_colour_lut("noir"), .intensity = 1.0F})`.

## Your own LUTs

`render.set_grade{ lut = "assets/luts/mine.cube" }` (or `Renderer::load_colour_lut`) loads a 3D `.cube` file from the
data directory and registers it under that path. Supported: `LUT_3D_SIZE` 2 .. 64 and the default 0 .. 1 domain.
1D tables and other domains are rejected.

## How it is stored

Vulkan 3D textures are not part of the bindless set, so a LUT is a 2D strip, `size * size` texels wide and `size`
tall (`ColourLut::to_strip_rgba8`): blue slices run along x, green down y, red along x inside a slice. `composite.slang`
filters inside a slice with the sampler (texel centres keep it from touching the next slice) and blends the two
neighbouring blue slices itself, which makes it a full trilinear lookup. Texture index 0 (the white default image)
means "no LUT".
