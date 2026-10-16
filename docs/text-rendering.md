# Text rendering

Text is drawn entirely on the GPU from fonts read with FreeType. `TextRenderer` (`rendering/text_renderer.hxx`) is an
immediate-mode renderer like the debug and ImGui ones: `draw()`, `draw_world()` and every `Components::Text` in the scene
say what to show this frame, and the frame graph hook (`declare`) turns that into passes.

## From outline to pixels

1. **Outlines (CPU, `rendering/font.hxx`).** FreeType loads a glyph unscaled and unhinted. Its curves are flattened to
   line segments in em units, so the rest of the pipeline only ever sees straight segments. Nothing is rasterised on the
   CPU.
2. **Cache and batch (CPU, `rendering/text_cache.hxx`).** `text::Cache::build` lays out the frame's text
   (`rendering/text_layout.hxx`: UTF-8, kerning, newlines), notes which glyphs the atlas lacks and gives each one a *job*
   (its segments, tile size and atlas slot), and builds a quad per drawn glyph. It mirrors the GPU's atlas placement
   (`text::ShelfPlanner`) to know when the atlas is full; then it clears everything and the GPU resets the atlas before
   this frame's jobs. At most 128 new glyphs are rasterised per frame; the rest follow on the next.
3. **Rasterise (`text_raster.slang`).** One thread per texel of every new tile: the signed distance, in texels, to the
   outline (nearest segment; inside is a non-zero winding number). Glyphs are rasterised once, at 64 px per em with 8
   texels of reach, and scaled to any size at draw time.
4. **Pack (`text_place.slang`, `text_blit.slang`).** One thread shelf-packs the new tiles into a persistent 2048x2048
   atlas and writes a rectangle table (slot -> x, y, w, h). A second pass copies the tiles in, four texels per thread,
   as bytes with 128 on the outline. The atlas, the rectangle table and the shelf state are persistent buffers, so a
   glyph is rasterised and packed once for as long as it stays.
5. **Draw.** The same atlas is drawn two ways:
   - **Screen space (`text_bin.slang`, `text_composite.slang`)**, in the `before_composite` slot. Glyph quads are binned
     into 16x16 screen tiles; a 16x16 compute group per tile sorts its list into draw order and blends the glyphs over the
     scene colour, anti-aliased over one pixel from the distance field.
   - **World space (`text_world.slang`)**, a scene draw in the forward pass: depth-tested, alpha-blended quads placed by a
     transform, with the edge width taken from `fwidth` so text stays sharp under perspective.

## Entities, emission and glow

`Components::Text` is drawn at its entity's world transform. `emissive` multiplies the colour (per glyph, as a float,
since the packed 8-bit colour cannot hold HDR), so values above 1 pass the bloom threshold. A `ParticleEmitter` with
`ParticleShape::text` spawns on the outline of the entity's text: `TextRenderer::emission_points` spaces points evenly
along the outline and uploads them for the frame, and `particles_simulate.slang` picks one per spawned particle.

## Shared plumbing

The renderers share two helpers: `UploadRing` (`gpu/upload_ring.hxx`), per-frame host-visible buffers that grow by
adding blocks rather than replacing the one recorded commands still read, and `OverlayShader`
(`rendering/overlay_shader.hxx`), which registers an overlay's pipeline for the scope it records into and hides the
raw command-buffer calls behind typed push constants. The debug and ImGui renderers use them too; the text renderer's
upload buffers are the same ring.

## Limits

Single-channel distance fields round very sharp corners at large sizes; glyphs whose tile would exceed 160 texels per
side are not drawn; there is no shaping beyond kerning (ligatures, right-to-left and complex scripts need a shaper such
as HarfBuzz); and the atlas holds about 4096 glyphs at once before it is cleared.
