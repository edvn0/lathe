# Scenes

Saved `.lbf` scenes. Open one with **Ctrl+O** in the editor, by dropping it
on the window, or at startup with `--scene=assets/scenes/<name>.lbf`.

| File | Contents |
| --- | --- |
| `light_field.lbf` | `BasicGame`'s village plus 4,900 lanterns, about 5,000 point lights in all, for exercising clustered lighting. |

## `light_field.lbf`

- **Lanterns:** each is a small emissive sphere with a point light, 0.6-2 m above the terrain, ranges 3.5-6.5 m,
  in six colours. They sit on a jittered 2.4 m grid over 190 x 190 m around the village, keeping clear of the
  village centre and the roads.
- **Hierarchy:** the lanterns are grouped under `light_field`, one `light_field_row_N` child per grid row, so the
  Hierarchy stays collapsible.
- **Shadows:** the lantern materials don't cast sun shadows.
- **Assets not embedded:** the file was saved with `SceneSaveOptions::embed_assets = false`. It only references
  `assets/models/scattering_skull.glb` and `assets/models/test_cube.glb`, which are imported from source on
  load. Saving it again from the editor embeds them, which makes the file about 5 MiB bigger.
- **Not saved:** what the format can't hold. The road ribbons are generated at runtime, so the lamp posts stand
  on bare terrain. The enemies' unregistered AI script is dropped too.
