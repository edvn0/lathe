# Models

`assets/models/*` is gitignored; add committed models with `git add -f`.

| File | Source | License |
| --- | --- | --- |
| `scattering_skull.glb` | [ScatteringSkull](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull) from the Khronos glTF Sample Assets. Model by Vladimir Petkovic, (c) 2025 Adobe Inc. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

`scattering_skull.glb` is ~189K triangles (~98K vertices) in a single primitive, used as a dense mesh for the
meshlet debug colours (`MaterialCreateInfo::debug_meshlet_colours`). The engine ignores its subsurface/volume
material extensions and renders it as a plain PBR material with its baked occlusion.

## Downloaded at runtime

These are not in the repository. The game fetches them on first run into `assets/models/` (gitignored), straight from
the upstream repository, pinned to a commit and verified against a SHA-256 before it is kept.

| File | Source | License |
| --- | --- | --- |
| `damaged_helmet.glb` | [DamagedHelmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/5bad5aaa0bbb5d0f9cdc934e626f27d0df1e79b8/Models/DamagedHelmet) from the Khronos glTF Sample Assets. Converted to glTF by ctxwing; earlier version by theblueturtle\_. | Conversion: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original model: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

The helmet is for **non-commercial use only** (the CC BY-NC term follows the file, not the engine's MIT code). Credit
both authors if you show it. To change the pinned commit or hash, edit `game/src/basic_game.cxx`.

## Chess pack (local only)

`chess/*.glb` is the "3D voxel Chess pack" (glb folder), used by `--game=chess`. It is gitignored and not committed; the
pack shipped without a license file, so check its terms before adding it with `git add -f`. Copy the `glb/` files into
`assets/models/chess/`.

## Animated human (local only)

| File | Source | License |
| --- | --- | --- |
| `animated_human.glb` | [Animated Human - Low Poly](https://opengameart.org/content/animated-human-low-poly) by Quaternius | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

A rigged low-poly character (48 joints; Idle, Walk, Run, Jump, Death, Punch, Working and more). `--game=moving` uses it
in "Skinned model" mode. A copy is in `test/data/animated_human.glb` for the
skin importer tests; this one is the gitignored runtime copy.
