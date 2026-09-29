# Models

`assets/models/*` is gitignored; add committed models with `git add -f`.

| File | Source | License |
| --- | --- | --- |
| `scattering_skull.glb` | [ScatteringSkull](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull) from the Khronos glTF Sample Assets. Model by Vladimir Petkovic, (c) 2025 Adobe Inc. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

`scattering_skull.glb` is ~189K triangles (~98K vertices) in a single primitive, used as a dense mesh for the
meshlet debug colours (`MaterialCreateInfo::debug_meshlet_colours`). The engine ignores its subsurface/volume
material extensions and renders it as a plain PBR material with its baked occlusion.
