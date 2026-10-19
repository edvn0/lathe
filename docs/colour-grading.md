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

A LUT is a real 3D texture: a `size`^3 `R8G8B8A8_UNORM` volume with red along x, green along y and blue along z
(`ColourLut::to_volume_rgba8`), bound in the bindless set as `sampled_3d` (binding 7, `Texture3D<float4>` in
`bindless.slang`). `composite.slang` maps colour onto texel centres and takes one `SampleLevel`, so the hardware does the
trilinear blend. Texture index 0 (the white default image) means "no LUT".

Any other system can use the same binding: create an `Image` with `image_type = VK_IMAGE_TYPE_3D`, `view_type =
VK_IMAGE_VIEW_TYPE_3D` and the `ImageDescriptorView::sampled_3d` descriptor view. Volumes are single-level for now (the
upload path does not generate mips for 3D images), and slots with no volume read a 1x1x1 black fallback.
