# Lathe Binary Format (.lbf)

One chunked container for scene files and asset packs. A scene saved from
the editor is self-contained: the entities plus every model and texture
they use, cooked into GPU-ready form, so opening it skips glTF parsing,
tangent generation, LOD simplification, meshlet building and texture
transcoding.

Code: `include/serialisation/`, `src/serialisation/` (the
`engine_serialisation` module, above `engine_rendering`). Editor glue:
`src/app/scene_files.cxx`.

## Why cooked assets

Engines like Unreal don't load source assets at runtime; they *cook* them
offline into platform-ready blobs (Unreal's `.uasset`/`.pak`, compressed
with Oodle) and load those. LBF does the same thing at a smaller scale. The
runtime import path here is already split into a CPU stage (`ModelCpuData`,
`CompressedTexture`) and a render-thread upload stage
(`start_model_gpu_upload` / `ImageStorage::upgrade_pending_image`); a cooked
asset is simply the CPU stage's *output* written to disk, so loading one
plugs straight into the existing upload path.

| Work at load        | glTF / PNG source | cooked `.lbf` |
| ------------------- | ----------------- | ------------- |
| parse               | fastgltf          | flat binary   |
| tangents (MikkTSpace) | yes             | no            |
| LOD simplification  | yes               | no            |
| vertex packing      | yes               | no            |
| meshlet build       | yes               | no            |
| texture decode + UASTC encode + BC transcode | yes (cached in `~/.cache/ktx2`) | no |
| what's left         |                   | zstd + meshopt decode, GPU upload |

## Container

```
[LbfFileHeader]            64 bytes at offset 0
[chunk payload] ...        each starts on a 64-byte boundary
[LbfChunkEntry x N]        table of contents (TOC), located by the header
```

- **Header**: magic `LBF\x1A`, version major/minor, file kind (`scene` or
  `asset_pack`), TOC offset/size/xxh64, total file size (a truncated file is
  rejected up front).
- **TOC entry** (48 bytes): fourcc type, compression, payload version, id
  (an `AssetId`, or 0 for singletons), offset, stored size, raw size, xxh64
  of the stored bytes. Sorted by (type, id), so lookups are binary searches.
- **Chunk types**: `SCEN` (the scene), `MODL` (cooked model), `TEXR`
  (cooked texture), `ENVM` (cooked HDR environment), `META` (which
  engine/format versions wrote the file; informational).
- **Compression**: per chunk, zstd (default level 6), or stored raw when it
  saves less than 3% (BC7 blocks barely compress, and skipping
  decompression is worth more). zstd is the copy libktx already links in.

### Streaming, not mapping

Nothing is memory-mapped or read whole. `LbfReader::open` reads the header
and the TOC; every `read_chunk` reads only that chunk, through a fixed
1 MiB buffer, decompressing (`ZSTD_decompressStream`) and checksumming
(`Xxh64Stream`) as it goes. Peak memory is the chunk's decompressed size
plus the buffer, whatever the file's size. `read_chunk` is thread-safe
(each call opens its own handle), so models decode in parallel on the
thread pool.

Writing streams too: `LbfWriter::write_file` writes a placeholder header,
then each chunk as soon as it's compressed (compression runs a bounded
window ahead of the write position), then the TOC, then patches the header.
That's why the TOC sits at the end. Each chunk's buffer is freed once
written. The file goes to `<path>.tmp` and is renamed over `<path>`, so a
failed save never leaves a half-written scene.

## Asset identity

`AssetId` = xxh64 of a canonical key (`asset_id.hxx`):

- `model:<path>`
- `texture:<path>|<role>`: the same image cooked as colour and as data is
  two assets
- `texture:<cache key>|<role>`: images embedded in a glTF
- `engine://cube` etc.: built-in procedural models, never cooked

Paths are normalised and made relative to the working directory, so ids
match across machines. Scenes store both the id and the source path: a scene
loads its assets from any pack holding the id and falls back to the source
file when none does.

## Chunks

### `MODL` (`cooked_model.hxx`)

A finalized `ModelCpuData`: node hierarchy, lights, materials (factors,
alpha mode, default sampler, image indices) and per primitive:

- packed `CompressedModelVertex` array through `meshopt_encodeVertexBuffer`
- every LOD's index buffer through `meshopt_encodeIndexBuffer` (a level
  mask marks which LODs alias the previous one)
- every LOD's meshlets: `GpuMeshlet` descriptors + topology data
- local AABB (the full-precision `ModelVertex` array isn't stored)

Images are references to `TEXR` chunks by id, so models sharing a texture
share the chunk.

### `TEXR` (`cooked_texture.hxx`)

A `CompressedTexture` exactly as `ImageStorage` uploads it: VkFormat
(BC7 sRGB / BC7 UNORM / BC5), every mip's extent and byte range, block data
16-aligned.

### `ENVM` (`cooked_environment.hxx`)

An HDR environment as half-float RGBA, so loading one is a zstd decode and a
byte reshuffle, with no `.hdr`/`.exr`/`.ktx2` parsing:

```
u32 width, u32 height, u32 vk_format (R16G16B16A16_SFLOAT only), u32 layers (1 = equirect, 6 = cubemap)
pixels: every layer's half-float RGBA, byte-shuffled (all low bytes, then all high bytes)
```

The id is `asset_id_from_key(environment_asset_key(path))`, i.e. the xxh64 of
`environment:<normalised path>`. The decoder refuses layers other than 1 or 6,
equirects that aren't 2:1 or exceed 16384x8192, cube faces that aren't square
powers of two up to 2048, other formats, and payloads whose size doesn't match
exactly. A scene whose environment is an image gets one `ENVM` chunk when it is
saved with assets embedded; loading prefers it over reading the source file.

### `SCEN` (`scene_codec.hxx`)

A `SceneDescription`, plain data with no handles: model/texture reference
tables, materials, entities (name, transform, parent, tags) and one list
per component type. Stored as **sections**,
`[u32 type][u16 version][u16 reserved][u64 size][payload]`, one per table,
component-wise (all point lights together, ...), so loading is a few tight
loops rather than a per-entity switch.

Instanced models' per-instance transforms (`instanced_models` v2) are
stored column-wise, one float column per field, each column byte-shuffled
(every value's first byte, then every second byte, ...) so zstd sees long
runs of shared sign/exponent bytes. When every instance in a component is a
plain translation-rotation-scale they are stored as those 10 floats,
recomposing to the original matrix within float rounding; otherwise (shear,
projection) as the 16 matrix floats, exactly. On a 14,892-blade grass field
that is 288 KiB compressed, against 464 KiB for v1's interleaved matrices.

The `environment` section (type 14) holds the scene's sky and image-based
lighting: source (flat ambient, procedural Preetham sky or HDR image), the image
path and `ENVM` asset id, rotation, exposure, diffuse/specular/occlusion
intensities, multi-scatter, the sun (azimuth, elevation, turbidity, ground
albedo, disc radius, colour, intensity) and the fog. A scene without it loads
as flat ambient with the sun at 30/55 degrees, exactly as scenes looked before
it existed; an engine from before it skips the section.

Saved components: Transform, Parent, Model, MaterialOverride (whole-model
and per-slot; slots name the model's material by index), InstancedModel,
PointLight, SpotLight, RigidBody (except heightfields, which terrain
regenerates), Script (by registered name), Lifetime, PlayerTag/BulletTag/
StreamedModelTag, plus the scene's physics settings and environment.

Not saved (warned about at save time): models generated at runtime with no
file behind them (e.g. procedural ribbons), scripts without a registered
name, and material textures with no source file.

## Versioning

Three levels, so builds with different serialisation versions can coexist:

1. **Container** (`lbf_version_major.minor`): a reader refuses another
   major; minor bumps are additive and ignored by older readers.
2. **Chunk payload** (`LbfChunkEntry::version`, per type):
   `cooked_model_version`, `cooked_texture_version`,
   `cooked_environment_version`, `scene_chunk_version`.
   Encoders always write the current version; decoders take the version
   from the TOC and accept `[oldest_readable, current]`. To change a
   layout, bump the constant, branch on the version in the decoder and keep
   the old branch. `MODL` embeds GPU-facing layouts (`CompressedModelVertex`,
   `GpuMeshlet`, meshlet limits); `static_assert`s in `cooked_model.cxx` trip
   when one changes, as a reminder to bump.
3. **Scene section** (per section type): a reader skips section types it
   doesn't know and versions newer than it understands (a newer engine's
   component) and keeps the rest. `SceneDecodeReport::skipped_sections`
   counts them, and the load logs a warning.

On re-save, chunks copied from the previously opened file are only reused
if they're at the current version; older ones are re-cooked from source,
so files converge on the newest layout.

| Payload | Current | Oldest readable |
| ------- | ------- | --------------- |
| container | 1.0 | 1.x |
| `MODL` | 1 | 1 |
| `TEXR` | 1 | 1 |
| `ENVM` | 1 | 1 |
| `SCEN` framing | 1 | 1 |
| `SCEN` sections | 1 | 1 |
| `SCEN` `instanced_models` section | 2 | 1 |
| `SCEN` `environment` section | 1 | 1 |

## APIs

| | blocking | non-blocking |
| - | - | - |
| save | `save_scene()` | `SceneSaveJob::start()` → `ready()` → `take()` |
| load | `load_scene()` | `SceneLoadJob::start()` → `step()` once per frame |

- **`SceneSaveJob`**: `capture_scene()` runs on the render thread (it reads
  live handles; just a registry walk). Cooking (models parsed in parallel,
  each texture cooked once), compression and the streamed write run on a
  background thread fanning out to the thread pool. Unchanged cooked assets
  are copied verbatim from `SceneSaveOptions::source_packs` (the file the
  scene was opened from) instead of re-cooked.
- **`SceneLoadJob`**: the file is opened and `SCEN` decoded in the
  background; `step()` then instantiates the entities at once, with models
  going through `ModelStreamer::request_prepared()`. Each cooked model
  decodes on the thread pool and uploads a few primitives per frame,
  rendering as the engine cube until it installs. Textures stream as usual.
  Per-slot material overrides wait for their model to install
  (`apply_deferred_slot_overrides`).
- **`load_scene()`**: blocking. Cooked models decode in parallel and upload
  before it returns. `SceneLoadOptions::wait_for_textures` also blocks until
  every texture is uploaded (`TextureStreamer::flush`), for loads that must
  be fully resident (tools, tests, game startup).
- **`scene_fingerprint()`**: an order-independent hash of a
  `SceneDescription`, which the editor uses to tell whether the scene changed
  since it was opened or saved.

Lower level: `LbfWriter`/`LbfReader` (container), `cook_assets()` +
`AssetPack` (asset packs; an `.lbf` of kind `asset_pack` holds only
`MODL`/`TEXR`/`ENVM`), `encode_scene`/`decode_scene`.

## Editor

- **Scene** panel: Open..., Save, Save As..., status and progress.
- **Ctrl+S** save, **Ctrl+Shift+S** save as, **Ctrl+O** open.
- **Drag and drop** onto the window: a `.lbf` opens the scene, `.gltf`/`.glb`
  spawns the model. If the scene has unsaved changes, a prompt offers
  *Save (overwrite) and open*, *Discard changes and open* or *Cancel*.
- Command line: `--scene=<path>` opens a scene at startup;
  `--save-scene=<path>` cooks the game's populated scene into a file.
