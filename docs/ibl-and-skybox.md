# Image-based lighting and dynamic skyboxes

The environment lights the scene and fills the background. It has three sources:

- **Flat ambient**: today's look, and the fallback.
- **HDR image**: an equirectangular `.hdr` or `.exr`.
- **Procedural sky**: Preetham, driven by the sun's azimuth, elevation and turbidity. The sun direction also drives the directional light, so the sky and the shadows always agree.

From the environment, compute passes build three things:

- an L2 spherical-harmonics irradiance (9 RGB coefficients);
- a GGX-prefiltered specular cube, one roughness per mip;
- a split-sum BRDF LUT, built once.

The forward pass replaces the flat ambient term with SH diffuse plus prefiltered specular × (F0·A + B), times AO, with a specular occlusion term derived from AO. A fullscreen skybox draws where no opaque geometry landed.

**Status: implemented** (PR #46), with the deviations and the not-yet-done items listed under [As built](#as-built). The milestones at the end are the order the work landed in; each one built, passed `cargo xtask test` and had a GPU check.

## Goals

- HDR equirect environments (`.hdr`, `.exr`) projected to a cube on the GPU.
- A procedural sky (Preetham, analytic) that renders into the same cube. The sun's direction and colour drive `Renderer::DirectionalLight`, and so the cascades.
- Precomputation in compute shaders: L2 SH irradiance, an importance-sampled GGX prefilter, and the BRDF LUT (once at startup, and again on shader hot reload).
- Rebuilds only when inputs change. Procedural-sky rebuilds are spread over frames and double-buffered, so a half-built cube is never sampled.
- A skybox drawn after opaque/mask geometry and before blend, in the forward pass, so it lands before bloom and composite.
- Forward shading with SH diffuse, split-sum specular, AO, and specular occlusion from AO. The flat ambient stays as the fallback when no environment is ready.
- The environment is part of the scene: saved and loaded, versioned, readable from old scenes, and readable by old engines.
- An editor Environment panel with source, asset picker, rotation, exposure, sky parameters, IBL intensities and debug views.
- KTX2 float equirect and cubemap sources, in addition to `.hdr` and `.exr`.
- Multi-scatter energy compensation on IBL specular, from the existing BRDF LUT.
- Fog that takes its colour from the environment (sun-tinted at sunset), and fog settings saved with the scene.

## Non-goals

- Local reflection probes, parallax-corrected cubes, screen-space reflections.
- Atmospheric scattering applied to geometry (aerial perspective). The existing exponential fog stays as it is, apart from the optional environment-derived colour.
- Time-of-day animation, night sky, moon, clouds.
- BC6H or other compressed HDR formats. KTX2 sources are uncompressed float only (no transcoding).
- Removing the sun from HDRIs. An HDRI with a baked sun double-counts with the directional light, and the user decides which one to keep (see Risks).
- Multi-scatter compensation for punctual and directional lights. v1 applies it to IBL specular only, so existing light results don't move.

## Current state (verified)

- **Flat ambient.** `UBO::ambient_intensity` (`assets/shaders/scene_types.slang:186-187`, mirror in `include/rendering/renderer.hxx:246-247`) is "a flat multiplier on the ambient term, standing in for IBL". The forward fragment shader uses it as `ambient = base_colour * ambient_occlusion * ubo.ambient_intensity` (`assets/shaders/forward_geom.slang:346`). It is set by `Renderer::set_ambient_intensity` (`renderer.hxx:477-479`, member `ambient_intensity_` at `:1297`, default 0.15), written into the UBO at `src/rendering/renderer.cxx:2915`, and edited by the Lighting panel slider (`src/app/application.cxx:2239-2242`).
- **AO.** `ambient_occlusion` is the baked AO times GTAO (`forward_geom.slang:265-273`). GTAO outputs a scalar visibility only, with no bent normals (`gtao.slang`).
- **BRDF.** `assets/shaders/pbr.slang` has GGX D, height-correlated Smith V (`visibility_smith_ggx_correlated`) and Schlick F, with F0 = lerp(0.04, albedo, metallic). The BRDF LUT must integrate the same D and V.
- **Background.** The forward pass clears HDR to `(0.015, 0.025, 0.050)` (`src/rendering/render_passes.cxx:1050`). Under MSAA it resolves with `AVERAGE` and stores `DONT_CARE` (`:1053-1058`), so nothing can draw into the MSAA colour target after `vkCmdEndRendering` (`:1144`). Draw order inside the scope is opaque (depth `EQUAL`), mask, blend (`GREATER_OR_EQUAL`, no depth write), then scene overlays (`:1118-1142`). Pipeline-statistics queries span the draws (`:1087`, `:1140`).
- **Depth convention.** `perspectiveLH_ZO` with the viewport `minDepth = 1, maxDepth = 0` and flipped Y (`set_forward_dynamic_state`, `render_passes.cxx:316-343`). Stored depth is `1 - ndc.z` and the clear (0) is the far plane (`docs/occlusion-culling.md`, Design). The UBO already holds `view`, `projection` and `inverse_projection` (`scene_types.slang:145-151`).
- **Pass graph.** `Renderer::record_frame` (`renderer.cxx` ~4590-4650) runs shadow → depth prepass (early) → [Hi-Z build → occlusion cull → late prepass] → occlusion stats readback → AO → forward (+ scene overlays) → bloom → composite (+ UI). Each stage writes two timestamps (`RenderStage`, `include/rendering/render_stage.hxx`) and a `TracyVkZoneC`. Bindless descriptors are refreshed in `prepare_frame` before recording (`gpu_resource_table_.prepare_frame`, `renderer.cxx:2331`). An image created during `record_frame` is therefore not visible to shaders until the next frame's `prepare_frame`.
- **Bindless table** (`include/gpu/gpu_resource_table.hxx:19-25`, `src/gpu/gpu_resource_table.cxx:47-86`) has these bindings: 0 `sampled_2d`, 1 `samplers`, 2 `comparison_samplers`, 3 `sampled_2d_depth`, 4 `storage_2d`. **There is no cube binding.** `assets/shaders/bindless.slang` declares bindings 0-3; compute shaders declare binding 4 locally (`RWTexture2D<float4> storage_2d[]` in `bloom_upsample.slang`, and `[format("r32f")]` in `hiz_build.slang:25`). Cube, array and storage views are only written when present: "Cube/array/storage bindings have no fallback images yet" (`gpu_resource_table.cxx:230`). Pool sizes are 3× (sampled) and 2× (storage) `image_capacity` per frame (`:107-120`).
- **Images.** `ImageDescriptorView` already has `sampled_cube` and `storage_2d_array` (`include/gpu/image.hxx:29-36`). `Image::create` builds a `VK_IMAGE_VIEW_TYPE_CUBE` view when the image is `CUBE_COMPATIBLE` with ≥ 6 layers (`src/gpu/image.cxx:49-52`, `:253-255`). `create_mip_layer_views` builds one 2D view per (mip, layer) (`image.hxx:149-155`). `ImageDescriptorRecord` carries `sampled_cube` (`image_storage.hxx:31-40`, filled at `image_storage.cxx:930`). `ImageStorage::register_view` aliases only `sampled_2d`/`storage_2d` (`image_storage.hxx:107-110`, `:139-142`). That is enough for per-(mip, face) slots, the pattern used by `Renderer::create_hiz_pyramid` (`renderer.cxx:3542-3594`) and the bloom target (`:4727-4741`). Uploads with pixels end in `SHADER_READ_ONLY` visible to `VERTEX|FRAGMENT|COMPUTE` (`image_storage.cxx:612-620`).
- **HDR loading.** `DecodedImage::load_from_file` (`image.cxx` ~840-855) sends `.exr` to `decode_exr` (tinyexr, `CMakeLists.txt:438`), which returns `VK_FORMAT_R16G16B16A16_SFLOAT` (`image.cxx:759`). Everything else, `.hdr` included, goes through `stbi_load` (`image.cxx:774`), i.e. **8-bit LDR**. stb is built without `STBI_NO_HDR` (`src/assets/stb.cxx`), so `stbi_loadf` is available, and so is `stb_image_resize2`. libktx is linked (`CMakeLists.txt:479`), but `texture_pipeline.hxx` only produces BC5/BC7 through UASTC (`TextureRole`: colour, generic, normal_map; `include/gpu/compressed_texture.hxx:16-20`).
- **Samplers.** `SamplerStorage::linear_clamp()` (trilinear, `max_lod = VK_LOD_CLAMP_NONE`) and `linear_repeat()` (`include/gpu/sampler_storage.hxx:136-150`).
- **Pipelines** are `VK_EXT_shader_object` sets behind `PipelineGraphRepository` (`docs/pipeline_to_shader_objects.md`, `include/rendering/pipeline_graph_repository.hxx`). They are registered in one parallel batch in `Renderer::initialize` (`renderer.cxx:368-856`, indexed handles at `:841-863`). All share `global_push_constant_range` (256 B, `include/gpu/shader_stage.hxx:3-7`). Hot reload goes `ShaderChangeQueue` → `on_files_changed` → `process_dirty` at the top of `prepare_frame` (`renderer.cxx:2314-2318`). A rebuilt pipeline gets a new `ShaderObjectHandle` (`PipelineGraphRepository::shader_object_handle`). Push-constant structs are reflected into `shader_push_constants.hxx` by `add_shader_push_constant(...)` in `cmake/shader_push_constant_reflection.cmake:209-286`. `tools/check_shaders.py` discovers every `[shader(...)]` entry point automatically.
- **Directional light.** `Renderer::DirectionalLight` (`renderer.hxx:440-444`) is set only from the Lighting panel. That panel converts `light_azimuth_degrees`/`light_elevation_degrees` (`include/app/application.hxx:168-169`, defaults 30°/55°) with `dir = (cos e·cos a, sin e, cos e·sin a)`, clamps elevation to 5°-89° because "a horizontal light degenerates the cascade depth range" (`application.cxx:2229-2264`), and calls `set_directional_light`. A change of light direction invalidates every cached cascade (`renderer.cxx:2789`). **The directional light, fog and ambient are not saved in scenes.**
- **Scene serde.** `SceneDescription` (`include/serialisation/scene_codec.hxx`) is stored as SCEN sections `[u32 type][u16 version][u16 reserved][u64 size][payload]`. Section ids 1-13 are taken (`settings` = 1 holds only physics). A reader skips unknown types and newer versions and counts them in `SceneDecodeReport::skipped_sections` (`src/serialisation/scene_codec.cxx:883-889`). The codec table is at `:692-765`. `read_textures` **fails the section** if `role > TextureRole::normal_map` (`:162`), so a new `TextureRole` would make older engines reject the whole scene. Settings are copied on capture and instantiate (`scene_serialisation.cxx:258`, `:658`). `scene_fingerprint` hashes a `settings_only` description (`:1256`). `play()` copies `physics_settings` into the runtime scene (`application.cxx:2490`). `LbfReader` ignores unknown chunk fourccs (lookups are by type + id). Asset keys are in `include/serialisation/asset_id.hxx`. The fixture `assets/scenes/light_field.lbf` predates this work.
- **Tests.** doctest, CPU only, `test/*_test.cxx` globbed into `lathe-tests` against `lathe::core` (`test/CMakeLists.txt`). Mirrored CPU/GPU maths has precedent: `include/rendering/hiz_occlusion.hxx` ↔ `hiz_occlusion.slang`, tested in `test/hiz_occlusion_test.cxx`. Serde tests are in `test/lbf_test.cxx`.
- **Tracy.** CPU zones use `ZoneScopedNC`, GPU zones use `TracyVkZoneC(context_.host_query_context.context, cmd, "Name", tracy::Color::X)` (`renderer.cxx:3752`, `:4175`, ...).
- **Assets.** Third-party assets were removed in favour of generated ones (`assets/LICENSES.md`, `assets/textures/generate_textures.py`, run with `uv run --with ...`). There is no HDRI in the repository.

## Architecture

```
Scene::environment (SceneEnvironment, plain data, saved)
        |  submit_scene() each frame (src/main.cxx)
        v
Renderer::set_environment(EnvironmentDesc)  -- hashes inputs, derives DirectionalLight from the sun
        |
        v
EnvironmentSystem  (include/rendering/environment.hxx, src/rendering/environment.cxx)
  prepare(frame)   [in prepare_frame, before gpu_resource_table_.prepare_frame]
      - finishes async HDR decode -> uploads equirect (RGBA16F 2D)
      - (re)creates cubes when source/size changes; retires old ones after frames_in_flight
      - plans this frame's work units; decides which prefilter/SH set is live; fills UBO env block
  record(ctx)      [record_frame, new stage RenderStage::Environment, first pass]
      - BRDF LUT (once / on hot reload)
      - capture: equirect->cube or sky->cube, mip chain, SH projection, prefilter mip 0
      - prefilter units (one face per frame when amortized)
render_pass::forward_geometry
      opaque -> mask -> skybox (new) -> blend -> scene overlays
```

- **`EnvironmentSystem`** is owned by `Renderer` (member `environment_`), in its own files, to keep `renderer.cxx` (5.1K lines) from growing. It holds the GPU resources, the input hash, the build state machine, and the pipeline handles of the six new shader sets.
- **Inputs vs. lookups.** Rotation (yaw), exposure, the diffuse and specular intensities, and the debug view are applied at lookup time (skybox, SH eval, prefilter sample). Changing them never triggers a rebuild. A rebuild is needed only when the **radiance changes**: source, HDR file, sky turbidity, ground albedo or sun elevation/azimuth (procedural only), cube size, or a hot-reloaded environment shader.
- **Sun.** In procedural mode the sun direction comes from azimuth/elevation (the editor's existing convention: azimuth 0 = +X, 90° = +Z, Y up). In world space this is the same vector as `DirectionalLight::direction`. The sun's colour is the sky model's transmittance at that elevation (CPU, `sky_model.cxx`), times a user intensity. The sun disc is drawn in the skybox but **left out of the radiance capture**: the directional light already lights the scene with it, and including it would count it twice. In HDR mode the directional light is manual (direction from the same azimuth/elevation, manual colour and intensity).
- **Double buffering.** Only the prefilter cube and the SH slot are double-buffered (`set[0]`, `set[1]`). Forward reads `live` while a build writes `!live`. The flip happens in `prepare`, in the frame whose recorded work completes the build. The schedule is fixed on the CPU, so the UBO for that frame already points at the new set. The radiance cube is single-buffered: it is only read by builds (which capture it at their start) and by the HDR skybox (HDR builds are immediate, see below).
- **Build policy.** `immediate` does the whole build in one frame. It is used for HDR source loads, cube-size changes, the first build, "Rebuild now", and when `amortize_rebuilds` is off. `amortized` does capture in frame 0 and one prefilter face per frame in frames 1-6. It is used for procedural sky parameter changes. If inputs change during a build, the build **finishes** and one more starts with the latest inputs. Lighting is therefore at most two builds (about 14 frames) behind a slider drag, and it always converges. Restarting on every change would starve during continuous drags.
- **Hot reload.** `EnvironmentSystem::prepare` compares each of its pipelines' current `ShaderObjectHandle` with the one it last built with. A change to `env_brdf_lut.slang` re-runs the LUT. A change to any capture, projection or prefilter shader marks the environment dirty (immediate rebuild). The skybox needs nothing.

## Resources and formats

| Resource | Format | Size / mips | Views and bindless slots | Lifetime |
|---|---|---|---|---|
| Equirect source | `R16G16B16A16_SFLOAT`, 2D | source size, capped at 8192×4096 (downscaled on CPU with `stb_image_resize2`); 1 mip | `sampled_2d` | Upload → projection; retired `frames_in_flight` frames after projection |
| Radiance cube | `R16G16B16A16_SFLOAT`, `CUBE_COMPATIBLE`, 6 layers | HDR equirect: 512² (option 1024), 10 mips; KTX2 cube: its own face size (power of two, 64…1024); procedural: 256², 9 mips | primary `sampled_cube` (binding 5); per (mip, face) `mip_layer_view` registered via `register_view{sampled_2d, storage_2d}` = 60 / 54 slots | Recreated on source or size change |
| Prefilter cube ×2 | `R16G16B16A16_SFLOAT`, cube | 256², **6 mips** (256…8), roughness `r_m = m / 5` | `sampled_cube` each; 36 (mip, face) slots each = 72 | Created once |
| BRDF LUT | `R16G16B16A16_SFLOAT`, 2D (the engine's compute shaders write RGBA16F through format-less storage, so the LUT does too; only R and G are used) | 128², 1 mip; x = N·V, y = perceptual roughness | `sampled_2d` + `storage_2d` | Created once |
| SH buffer | storage buffer, device local, `SHADER_DEVICE_ADDRESS` | 2 slots × `GpuEnvironmentSh` (9 × float4 = 144 B) | device address in `UBO::environment_sh_address` | Created once |
| Black cube fallback | `R8G8B8A8_UNORM` cube, 1² | 1 mip | written to every binding-5 slot without a cube view | `ImageStorage` default |

Memory: radiance 16.8 MB (HDR 512) / 4.2 MB (procedural), prefilter 2 × 4.2 MB, LUT 64 KB. A 4K equirect upload is a transient 64 MB, plus the same again for staging. Bindless slots: about 135 of the default 4096.

Coordinate conventions, implemented once in `include/rendering/cube_map.hxx` and mirrored in `assets/shaders/environment.slang`:

- **Cube faces** follow the Vulkan order and per-face (s, t) table: +X, −X, +Y, −Y, +Z, −Z. `cube_texel_direction(face, uv)` and `direction_to_cube_texel(dir)` invert each other (unit-tested against the spec table).
- **Equirect.** `φ = atan2(d.x, d.z)`, `u = 0.5 + φ / 2π`, `v = acos(d.y) / π`. The image's centre column faces +Z and its top row is the zenith.
- **Rotation.** Yaw θ about +Y. Lookups use `R_y(−θ)·d`. Shaders take `(cos θ, sin θ)` from the UBO.

## Bindless changes

- `GpuResourceBinding::sampled_cube = 5` (`gpu_resource_table.hxx`). Add a 6th `VkDescriptorSetLayoutBinding` (`SAMPLED_IMAGE`, `image_capacity`, `STAGE_ALL`) and raise the sampled-image pool size to 4 × `image_capacity` × frames. `prepare_frame` writes `record.sampled_cube`, or `images.black_cube()`'s view for every slot without one, which retires the "no fallback" comment for cubes.
- `ImageStorage::black_cube()`: a separate protected default (not in `DefaultImage`, so the `default_image_count` loops in `image_storage.cxx` stay untouched). It is created with `CUBE_COMPATIBLE`, 6 layers and `descriptor_views = sampled_cube`, and cleared to 0 with `vkCmdClearColorImage` in `ImageStorage::prepare_frame` next to the default uploads.
- `bindless.slang`: `public static const uint sampled_cube_binding = 5;` and `public [[vk::binding(sampled_cube_binding, 0)]] TextureCube<float4> sampled_cube[];`.
- Storage writes go per face through the existing binding 4 (`[format("rgba16f")] RWTexture2D<float4>`). Each dispatch receives the six face slots of its mip in push constants, so neither `storage_2d_array` nor per-mip array views are needed.

## Compute passes

All of them are `[numthreads(8, 8, 1)]` unless stated otherwise. Each binds through `detail::bind_compute_node` and `resource_table.bind`, and is registered in `Renderer::initialize` after `hiz_build` (indices 23+). Push constants are reflected by new `add_shader_push_constant` lines.

| Shader (`assets/shaders/`) | Entry | Dispatch | Work |
|---|---|---|---|
| `env_brdf_lut.slang` | `main_cs` | (16, 16, 1) for 128² | Split-sum LUT: Hammersley, **1024** GGX importance samples per texel. V = `visibility_smith_ggx_correlated` (same as `pbr.slang`). Outputs (A, B) with F = F0·A + B. Rows run from roughness 0.045 (the forward clamp) to 1. |
| `env_equirect_to_cube.slang` | `main_cs` | (s/8, s/8, 6), z = face | 4 samples per texel (2×2 rotated grid) of the equirect with a dedicated sampler (U repeat, V clamp, linear, no mips). Clamps to 65000 (below half-float max) and replaces NaN/inf with 0. Writes radiance mip 0. |
| `env_sky_to_cube.slang` | `main_cs` | (s/8, s/8, 6) | Evaluates Preetham from the UBO env block (`sky_perez`, `sky_zenith`), converting xyY → linear sRGB. Below the horizon: ground albedo × the horizon value, blended over −2°…0°. **No sun disc.** Writes radiance mip 0. |
| `env_downsample.slang` | `main_cs` | (s_m/8, s_m/8, 6) per level | 2×2 box from level m−1 through that level's per-face `sampled_2d` slots (`Load`), written to level m's `storage_2d` slots. Within-face box; the source mips only feed filtered importance sampling. |
| `env_sh_project.slang` | `main_cs` | (1, 1, 1), `[numthreads(256,1,1)]` | Reads the radiance level where faces are 32² (6144 texels, 24 per thread), weighted by texel solid angle (`atan2(xy, sqrt(x²+y²+1))` corner form). Accumulates 9 × RGB with `WaveActiveSum`, then a `groupshared float[8][27]` cross-wave sum. Multiplies by the cosine-lobe factors (π, 2π/3, π/4) and by 1/π. Writes `GpuEnvironmentSh` to the building slot. |
| `env_prefilter.slang` | `main_cs` | (⌈s_m/8⌉, ⌈s_m/8⌉, 1) per (face, mip) | Mip 0: mirror copy via `SampleLevel(dir, log2(radiance_size / 256))`. Mips 1-5: GGX importance sampling with N = V = R (Karis), samples with N·L ≤ 0 skipped, weighted by N·L. **Filtered importance sampling**: `lod = 0.5·log2(Ω_s/Ω_p) + 1`, with `Ω_s = 1/(N·pdf)`, `pdf = D/4`, `Ω_p = 4π/(6·size²)`, sampling the radiance cube (binding 5, `linear_clamp`). Samples per mip: 32 / 48 / 64 / 64 / 64 for mips 1-5. |

The CPU computes the sample sets and Hammersley sequence identically in the shaders. Nothing is uploaded.

Cost per prefilter build (base 256): mip 1 is 6 × 128² × 32 ≈ 3.1 M samples; the whole cube is about 5 M trilinear cube fetches. A face per frame is about 0.85 M.

**Schedule** (`EnvironmentSystem::plan_frame`):

| Build frame | Immediate | Amortized (procedural) |
|---|---|---|
| 0 | capture, downsample chain, SH, prefilter mip 0 + all faces of mips 1-5, flip | capture, downsample chain, SH (into the building slot), prefilter mip 0 (all faces) |
| 1-6 | | prefilter face `f = frame − 1`, mips 1-5; frame 6 flips |

The SH flips together with the prefilter set, so diffuse and specular always come from the same radiance.

## Barriers

These follow the bloom and Hi-Z conventions (`render_passes.cxx:1249-1271`, `:880-901`). Each target level is `GENERAL` while written and `SHADER_READ_ONLY_OPTIMAL` while read, with one barrier per level.

- **Capture start**: all radiance subresources `UNDEFINED → GENERAL` (mip 0) / `UNDEFINED → GENERAL` per level as reached. src = `COMPUTE | FRAGMENT`, access none: a write-after-read against earlier frames' skybox and prefilter reads, ordered because there is one queue with in-order submission. dst = compute storage write.
- **Each level after its dispatch**: `GENERAL → SHADER_READ_ONLY` (src compute storage write → dst `COMPUTE | FRAGMENT` sampled read).
- **SH**: buffer barrier before the write (src `FRAGMENT` storage read → dst `COMPUTE` storage write, execution and WAR on the building slot); after the write, `COMPUTE` write → `FRAGMENT | COMPUTE` read.
- **Prefilter unit**: that face's mips in the building set `UNDEFINED → GENERAL` (src `FRAGMENT | COMPUTE`, none), dispatches, then `GENERAL → SHADER_READ_ONLY` (dst `FRAGMENT | COMPUTE` sampled read). The building set is not referenced by this frame's UBO until the flip frame, and the flip frame's last unit barrier precedes the forward pass in the same command buffer.
- **BRDF LUT**: `UNDEFINED → GENERAL` → dispatch → `SHADER_READ_ONLY` (dst `FRAGMENT`).
- **Equirect upload**: already ends in `SHADER_READ_ONLY` visible to `COMPUTE` (`image_storage.cxx:612-620`). It is recorded in `prepare_frame`, before `record_frame`'s projection in the same command buffer.
- **Resource creation and retirement**: images are created in `EnvironmentSystem::prepare`, before `gpu_resource_table_.prepare_frame` (`renderer.cxx:2331`), so their slots exist in the frame that first uses them. Replaced cubes go on a retire list keyed by frame number and are released `frames_in_flight` frames later. Source changes never wait for idle.

The Environment stage writes its two timestamps every frame, empty when idle (`Renderer::write_empty_stage`).

## Pass graph placement

- **`RenderStage::Environment`** is appended **after `BloomPass`** (before `Count`) so existing stage indices and benchmark JSON keys keep their values. Add `to_string` "Environment" and the benchmark name `environment` (`src/app/benchmark.cxx`). It is recorded by the new `Renderer::record_environment_pass(pass_context, frame)` directly after `record_overlay_prepares` and before `record_shadow_pass`, with `TracyVkZoneC(..., "Environment", tracy::Color::SkyBlue)`. The CPU side of `EnvironmentSystem::prepare` uses `ZoneScopedNC("EnvironmentPrepare", tracy::Color::SkyBlue)`, and the HDR decode job uses `ZoneScopedNC("DecodeEnvironment", tracy::Color::Goldenrod)`.
- **Skybox** goes in `render_pass::forward_geometry`, after the mask draws and before blend (`render_passes.cxx` between `:1130` and `:1132`). Blended surfaces then composite over the sky, MSAA edges against the sky resolve correctly, and no extra resolve or load is needed. It has no GPU zone of its own, because `render_pass::Context` carries no Tracy GPU context. Its time is part of `ForwardPass`. It lies inside the pipeline-statistics query, so Scene stats' fragment invocations include sky pixels (note this in the stats tooltip).
- Without a skybox (flat ambient, or `draw_skybox` off) the clear colour shows, exactly as today.

## Skybox

`assets/shaders/skybox.slang`, entries `main_vs` and `main_fs`, registered as a graphics shader set with `colour_formats = {hdr_format}`, `depth_format`, `samples = create_info.samples` (`renderer.skybox_pipeline`).

- `ForwardGeometryInfo` gains `std::optional<SkyboxDrawInfo> skybox` (`PipelineNodeHandle pipeline; VkDeviceAddress ubo_address;`).
- New `ForwardDynamicStateMode::sky`: cull none, depth test on, **depth write off, `GREATER_OR_EQUAL`**, empty vertex input, blending off. Draw with `vkCmdDraw(cmd, 3, 1, 0, 0)`.
- **VS**: fullscreen triangle (as in `composite.slang`) with `position = float4(xy, 1, 1)`, i.e. ndc.z = 1. Through the reversed viewport that stores depth 0 = far, so it passes `GREATER_OR_EQUAL` exactly where the depth is still the clear value.
- **FS**: ray = `normalize(transpose((float3x3)ubo.view) · (inverse_projection · float4(ndc, 1, 1)).xyz)`, using the same clip coordinates as the scene. The viewport flip is shared, so no extra Y flip is needed. Then:
  - procedural: Preetham plus a limb-darkened sun disc (`sun_direction_cos_radius`, `sun_disc_radiance`);
  - HDR: `sampled_cube[radiance].SampleLevel(R_y(−θ)·ray, 0)`;
  - debug views: the prefilter cube at a chosen LOD, or SH irradiance.

  The result is multiplied by `environment_exposure`.
- **Fog**: unfogged by default. With `fog_sky` set, `fogged = sky·T + scattered` at distance `far_clip` (exponential fog is near-total there, so this is effectively "sky = fog colour"). With `fog_from_environment` set, forward shading (`forward_geom.slang:367-370`) multiplies the in-scattered colour `fog_colour · fog_inscattering` by `environment_fog_colour(d)`: the prefilter cube at its blurriest mip, along the view ray with its vertical component scaled by 0.25, times exposure and sky intensity. `fog_colour` then acts as a tint (white by default). Fog follows sunsets and HDRIs for free, because the prefilter set is already kept up to date, and it needs no extra pass. Without a valid environment it falls back to the plain `fog_colour`.
- `SkyboxPushConstants { uint64 ubo_address; }` is reflected from `main_fs`.

## Shader changes

**UBO** (`scene_types.slang` and its mirror in `renderer.hxx`). `ambient_intensity` stays as the flat fallback. A block is appended after `light_lod_fade_radius_pixels`, with scalar-layout offsets pinned by `static_assert`s (UBO 760 → **952** bytes). In C++ the block is `EnvironmentUboBlock` (`environment.hxx`), nested in `UBO` as `environment` so `renderer.hxx` does not need the system's internals; both are scalar layout, so the flat fields in the shader land on the same offsets:

| Offset | Field | Meaning |
|---|---|---|
| 760 | `uint environment_flags` | bit 0 `ibl_valid`, 1 `skybox`, 2 `sky_procedural`, 3 `specular_occlusion`, 4 `fog_sky`, 5 `multi_scatter`, 6 `fog_from_environment`; bits 8-11 debug view |
| 764 | `float environment_exposure` | 2^EV, for both the skybox and IBL |
| 768 | `float4 environment_rotation` | cos θ, sin θ, prefilter max LOD (5), specular occlusion strength |
| 784 | `float4 environment_intensity` | diffuse, specular, sky intensity scale, debug LOD |
| 800 | `uint radiance_cube_texture, prefilter_cube_texture, brdf_lut_texture, environment_sampler` | bindless indices (`linear_clamp`) |
| 816 | `float4 sky_perez[4]` | Perez A-D for Y, x, y in rows 0-2; row 3 = (E_Y, E_x, E_y, 0) |
| 880 | `float4 sky_zenith` | xyz = zenith (Y, x, y) over F(0, theta_sun); Y carries the calibration, the user's sky intensity and the night fade |
| 896 | `float4 sky_ground` | rgb ground albedo |
| 912 | `float4 sun_direction_cos_radius` | xyz toward the sun, w = cos(angular radius) |
| 928 | `float4 sun_disc_radiance` | rgb |
| 944 | `Ptr<GpuEnvironmentSh> environment_sh` | live SH slot (`uint64`, 8-aligned) |

Putting the SH pointer in the UBO keeps the shared `PC`, `ForwardPushConstants` and `ShadowPushConstants` unchanged.

**New `assets/shaders/environment.slang`** (module): cube/equirect conventions, `rotate_y`, `sh9_irradiance(Ptr<GpuEnvironmentSh>, n)`, `preetham_sky(perez, zenith_xyY, cos_theta, gamma)`, `environment_fog_colour(dir)`, `specular_occlusion(ndotv, ao, roughness)`, Hammersley, GGX importance sampling (shared by the LUT and the prefilter).

**`forward_geom.slang` `main_fs`**, replacing line 346:

```slang
float3 ambient;
if ((ubo.environment_flags & env_ibl_valid) != 0) {
    const float3 n_env = rotate_y(shading_normal, ubo.environment_rotation.xy);   // R_y(-θ)
    const float3 r_env = rotate_y(reflect(-view_direction, shading_normal), ubo.environment_rotation.xy);
    const float ndotv  = max(dot(shading_normal, view_direction), 1e-4);
    const float3 f0    = lerp(float3(0.04), base_colour, metallic);
    const float2 ab    = sampled_2d[lut].SampleLevel(clamp_sampler, float2(ndotv, roughness), 0).rg;
    const float3 e_spec = f0 * ab.x + ab.y;
    // Multi-scatter energy compensation (Fdez-Aguera 2019), from the same LUT.
    const float3 energy = (flags & env_multi_scatter) != 0 ? 1.0 + f0 * (1.0 / (ab.x + ab.y) - 1.0) : float3(1.0);
    const float3 irradiance = sh9_irradiance(ubo.environment_sh_address, n_env);        // E/π
    const float3 prefiltered = sampled_cube[prefilter].SampleLevel(s, r_env, roughness * max_lod).rgb;
    const float so = specular_occlusion(ndotv, ambient_occlusion, roughness);            // Lagarde 2014
    const float horizon = saturate(1.0 + dot(reflect(-view_direction, shading_normal), normal));
    ambient = (base_colour * (1.0 - metallic) * (1.0 - e_spec) * irradiance * ambient_occlusion * diffuse_intensity
             + prefiltered * e_spec * energy * so * horizon * horizon * specular_intensity) * ubo.environment_exposure;
} else {
    ambient = base_colour * ambient_occlusion * ubo.ambient_intensity;   // unchanged fallback
}
```

`specular_occlusion = saturate(pow(ndotv + ao, exp2(-16·α − 1)) − 1 + ao)` with α = roughness². When `specular_occlusion` is disabled, `so = 1`. The `(1 − e_spec)` diffuse weight is the Filament-style energy split. The fallback branch is arithmetically identical to today's, so old scenes should render bit-identically. Debug views (bits 8-11) override `shaded_colour` before fog, the same way as the cluster heatmap and cascade tint.

## Procedural sky (Preetham)

- **CPU**: `include/rendering/sky_model.hxx` / `src/rendering/sky_model.cxx` (in `engine_rendering`, so `lathe-tests` can reach it). It is the analytic Preetham et al. (1999) model, so there is no vendored data.
  - `SkyModelParams { float elevation_radians, turbidity; }`. Ground albedo only shapes the below-horizon blend and is not part of the model.
  - `auto make_sky_state(SkyModelParams) -> SkyState` gives the Perez coefficients A-E for Y, x and y (each linear in turbidity) and the zenith (Yz, xz, yz) from the turbidity/sun-angle polynomials. It is packed into `sky_perez[4]` and `sky_zenith` in the UBO.
  - The shader evaluates `F(θ,γ) = (1 + A·e^(B/cosθ))(1 + C·e^(Dγ) + E·cos²γ)` per channel, then `Y·F(θ,γ)/F(0,θs)` and the same for x and y. It converts xyY → XYZ → linear sRGB and multiplies by `sky_intensity_scale` (the model's kcd/m² need calibrating to scene units). `cosθ` is clamped to ≥ 0.01 because the model is undefined below the horizon.
  - `auto sun_transmittance(elevation, turbidity) -> glm::vec3`: Rayleigh + Mie optical depth with Kasten-Young air mass, normalised to 1 at the zenith. It colours the directional light, so the light reddens at sunset with the sky.
- **Domain**: turbidity is clamped to [2, 10] (accuracy drops above about 6), and the model elevation to [0°, 90°]. The UI elevation range is [−10°, 90°]. Below 0° the model is evaluated at 0° and the sky radiance fades with `smoothstep(−10°, 0°)` toward a dim, night-ish floor. Below the horizon the ground colour is `ground_albedo` times the horizon value, blended over −2°…0°.
- **Directional light**:
  - direction = sun direction with elevation clamped to ≥ 5° (the cascade constraint);
  - colour = `sun_transmittance × user colour`;
  - intensity = `user intensity × smoothstep(−2°, 5°, elevation)`.

  These are pushed through `Renderer::set_directional_light` from `Renderer::set_environment` when `sun_drives_directional_light`. Because the cascade cache is keyed on the light direction, every sun move redraws every cascade. That is fine for editing, and it is the dominant cost of any future time-of-day animation.
- **Calibration**: `sky_intensity_scale` defaults so that, with the default sun (30°, 55°, turbidity 2.5), an upward-facing white Lambert surface gets the same ambient as today's flat term (0.15 × albedo). Scenes switched to the sky then keep their exposure.
- **Change detection**: the CPU state is rebuilt only when (elevation, turbidity, albedo) change by more than ε, and so is the radiance rebuild. A change of azimuth alone needs a capture (the sky is not rotation-invariant), so it also rebuilds, amortized.

## HDR images

- `DecodedImage::load_from_file` routes `.hdr` (and anything `stbi_is_hdr` accepts) to a new `decode_stbi_hdr`. It uses `stbi_loadf` with 4 channels and packs to `R16G16B16A16_SFLOAT` with `glm::packHalf1x16`, clamping to [0, 65000] and replacing NaN/inf with 0. `.exr` already works.
- Decoding runs on `thread_pool()` (`EnvironmentSystem::request_hdr`, with a `std::future`). Panoramas wider than 8192 are downscaled with `stb_image_resize2` before upload. The upload is in `prepare`; an immediate build follows in that frame's `record`. The skybox keeps the previous environment, or the clear colour, until then. Load errors are toasted and leave the previous environment live.
- **KTX2** (in v1): `ktxTexture2_CreateFromNamedFile` with uncompressed float formats (`R16G16B16A16_SFLOAT`, `E5B9G9R9_UFLOAT_PACK32`, `B10G11R11_UFLOAT_PACK32`) converted to half-float RGBA on the CPU. Block-compressed and Basis Universal files are refused, so libktx does no transcoding. Zstd or zlib supercompression is fine: libktx inflates it on load. `tools/env_bake` (`lathe-env-bake`) writes exactly this form (E5B9G9R9, Zstd) from an equirect, and the vendored `assets/environments/belfast_sunset_puresky_512.ktx2` is its output. A 2D file is treated as an equirect. A cubemap (`numFaces == 6`) skips projection: the faces are uploaded straight into radiance mip 0, and the radiance size is the face size. Both go through the same `DecodedHdr {width, height, layers, half pixels}` and the same `ENVM` cooked form.

## Scene data and serialisation

**Data** (`include/scene/environment.hxx`, plain, in `engine_scene`). It is a scene-level resource on `Scene` (`Scene::environment`, next to `physics_settings`), not an entity component. There is exactly one per scene, and it is not part of the hierarchy.

```cpp
enum class EnvironmentSource : std::uint8_t { flat_ambient = 0, procedural_sky = 1, hdr_image = 2 };

struct SceneSun {
    float azimuth_degrees = 30.0F, elevation_degrees = 55.0F;   // today's Application defaults
    float turbidity = 2.5F;
    glm::vec3 ground_albedo{0.3F};
    float angular_radius_degrees = 0.27F;
    glm::vec3 colour{1.0F, 0.97F, 0.92F};                       // today's DirectionalLight defaults
    float intensity = 3.0F;
    bool derive_colour_from_sky = true;                          // procedural only
};

struct SceneFog {
    bool enabled = false;
    glm::vec3 colour{1.0F};                     // tint when from_environment
    float extinction = 0.0F, inscattering = 1.0F;   // other defaults mirror Renderer's FogSettings
    bool from_environment = false;
};

struct SceneEnvironment {
    EnvironmentSource source = EnvironmentSource::flat_ambient;
    float ambient_intensity = 0.15F;            // flat fallback (and the value used while IBL is building)
    std::string hdr_source;                     // normalised path, hdr_image only
    float rotation_degrees = 0.0F;              // hdr_image only
    float exposure_ev = 0.0F;
    float diffuse_intensity = 1.0F, specular_intensity = 1.0F;   // "IBL intensity"
    float specular_occlusion = 1.0F;            // 0 disables
    float sky_intensity = 1.0F;                 // procedural only: scales the sky's radiance
    bool draw_skybox = true, fog_sky = false, sun_drives_directional_light = true;
    bool multi_scatter = true;                  // IBL specular energy compensation
    std::uint32_t hdr_cube_size = 512;          // equirect sources: 256, 512 or 1024
    SceneSun sun;
    SceneFog fog;                               // moved here from Renderer-only FogSettings
};
```

`SceneDescription::environment` holds the same struct plus `AssetId hdr_id`. `capture_scene` copies it (`scene_serialisation.cxx:258`) and `instantiate_scene` assigns it (`:658`). `Application::play()` copies it into the runtime scene (`application.cxx:2490`). `submit_scene` (`src/main.cxx`) calls `renderer->set_environment(...)` each frame, which is cheap because it hashes and compares. `Application::light_azimuth_degrees`/`light_elevation_degrees` go away in favour of `active_scene()->environment.sun`, and the fog UI writes `environment.fog`, which `set_environment` forwards to `Renderer::set_fog_settings`. Fog was not saved before; it now is.

**SCEN section** `scene_section::environment = 14`, `environment_section_version = 1`, `oldest_readable = 1`:

```
u8  source            u8 flags (draw_skybox | fog_sky<<1 | sun_drives_light<<2 | derive_sun_colour<<3 | multi_scatter<<4 | fog_enabled<<5 | fog_from_environment<<6)   u16 reserved
f32 ambient_intensity, rotation_degrees, exposure_ev, diffuse_intensity, specular_intensity, specular_occlusion, sky_intensity
u32 hdr_cube_size
u64 hdr_asset_id      string hdr_source
f32 sun.azimuth_degrees, sun.elevation_degrees, sun.turbidity   vec3 sun.ground_albedo
f32 sun.angular_radius_degrees   vec3 sun.colour   f32 sun.intensity
vec3 fog.colour   f32 fog.extinction   f32 fog.inscattering
```

- `validate_scene` refuses:
  - `source > 2`;
  - non-finite values;
  - turbidity outside [2, 10];
  - albedo outside [0, 1];
  - elevation outside [−90, 90];
  - negative intensities or exposure outside [−20, 20];
  - `hdr_image` with an empty source;
  - non-finite or negative fog values.
- **Old scenes, new engine**: `decode_scene` initialises `description.environment` to `SceneEnvironment{}` (flat ambient 0.15, sun at 30°/55°) before reading the sections, so a file without section 14 looks exactly like today. `assets/scenes/light_field.lbf` is the fixture.
- **New scenes, old engine**: section 14 is skipped (`skipped_sections`, with the existing load warning) and `ENVM` chunks are ignored, so the scene loads with today's lighting. **Do not** add a `TextureRole` for environments: older `read_textures` rejects unknown roles (`scene_codec.cxx:162`) and would fail the whole load.
- **New scenes in the editor** default to `procedural_sky`. That is the `Scene` constructor's default, not the decode default.
- **Fingerprint**: add `settings_only.environment = description.environment` (`scene_serialisation.cxx:1256`), so environment edits mark the scene dirty.

**Asset id and chunk.**

- Key: `environment:<path>` (`environment_asset_key()` in `asset_id.hxx`, with the key list comment updated).
- New chunk `lbf_chunk::environment = make_fourcc('E','N','V','M')`, payload `cooked_environment_version = 1` (`include/serialisation/cooked_environment.hxx`):

  ```
  u32 width, u32 height, u32 vk_format (R16G16B16A16_SFLOAT only in v1), u32 layers (1 = equirect, 6 = cubemap)
  pixels: half-float RGBA, layer-major, byte-shuffled (all low bytes, then all high bytes) for zstd
  ```

  The decoder refuses:
  - `layers` other than 1 or 6;
  - equirects that aren't 2:1 or exceed 16384×8192;
  - cubemap faces that aren't square powers of two or exceed 2048;
  - other formats;
  - payload sizes that don't match exactly (hostile-input tests like the TEXR ones).
- `AssetCookRequest::environments`, `AssetCookReport::environments_cooked/copied`, `AssetPack::has_environment/load_environment` (thread-safe), with copy-from-source-pack the same as TEXR.
- `instantiate_scene` with `hdr_image` tries the packs first, then the source path. The META string gains `environment=1`.
- `docs/lathe-binary-format.md` gains the `ENVM` chunk, the section, and table rows.

## Editor

A new **Environment** window (`widget("Environment", ...)`, docked with Lighting via `DockBuilderDockWindow("Environment", bottom)`). The Lighting panel loses Azimuth/Elevation/Colour/Intensity/Ambient (`application.cxx:2229-2242`) and the Fog section (`:2267-2275`); shadow settings stay there.

- **Source**: combo (Flat ambient / Procedural sky / HDR image). Status line: `EnvironmentSystem::status()` (`idle | decoding | building n/7 | ready | error`) with a progress bar.
- **HDR**: path, Browse... (a new `gui::FileBrowser environment_browser`, filters `{".hdr", ".exr", ".ktx2"}`, handled next to `model_browser.draw` at `:2367`), cube size (512/1024), Rotation (−180…180°).
- **Sky**: Turbidity (2…10; Preetham is calibrated for roughly 2-6), Ground albedo (colour), Sun angular radius, Sky intensity scale.
- **Sun**: Azimuth, Elevation (−10…90° for the sky, 5…89° for HDR/flat), "Derive colour from sky", Colour, Intensity, "Drive directional light".
- **Lighting**: Exposure (EV), Diffuse IBL, Specular IBL (replace "Ambient intensity"), Specular occlusion, Multi-scatter, Draw skybox, Fog sky. Flat-ambient intensity shows only for Flat ambient.
- **Debug**: view combo (None, Diffuse IBL only, Specular IBL only, Specular occlusion, Sky = prefilter LOD [slider], Sky = SH irradiance, White furnace); "Amortize rebuilds"; "Rebuild now"; face viewer (radiance or prefilter set/mip, 6 `ImGui::Image`s through `gui::linear_source_texture_id(slot)` of the per-face `sampled_2d` slots, shown raw); BRDF LUT image; SH coefficients (optional readback); last build GPU time.

Edits write `active_scene()->environment`. The renderer picks them up in `submit_scene`, and the fingerprint marks the scene dirty.

## Testing and verification

**Unit tests (CPU, CI)**:

- `test/cube_map_test.cxx`:
  - face/uv ↔ direction round trips;
  - agreement with Vulkan's face table at face centres and corners;
  - texel solid angles summing to 4π (≤ 1e-5);
  - equirect convention (centre column = +Z, top = +Y).
- `test/spherical_harmonics_test.cxx` (`include/rendering/spherical_harmonics.hxx`, mirrored by `environment.slang`):
  - basis orthonormality by numeric integration;
  - constant radiance 1 → stored E/π = 1 everywhere;
  - a single bright texel → the analytic clamped-cosine response within L2 ringing bounds;
  - yaw rotation equals projecting a rotated cube;
  - cube projection via solid angle equals Monte Carlo.
- `test/sky_model_test.cxx`:
  - Preetham state finite, with positive luminance and chromaticity inside the sRGB gamut, over a grid of (turbidity, elevation);
  - continuity across the horizon blend;
  - `sun_transmittance` red/blue ratio rising monotonically as elevation drops;
  - the zenith normalisation (`F(θ,γ)/F(0,θs)` = 1 at the zenith when the sun is at the zenith).
- `test/brdf_lut_test.cxx`: a CPU split-sum reference (the same Hammersley/GGX/V). A + B ≤ 1, (N·V = 1, r → 0) → (1, 0), and a few published values within 1e-2. The same reference also judges the GPU readback below.
- `test/lbf_test.cxx` additions:
  - environment section round-trip;
  - `light_field.lbf` and a payload without section 14 both decode to `SceneEnvironment{}`;
  - poisoned values refused;
  - `ENVM` round-trip and hostile sizes refused;
  - fingerprint sees an exposure edit;
  - `cook_assets` with a small generated `.exr`.
- `.hdr` decode: a tiny generated Radiance file decodes to the expected half floats, including the clamp.
- KTX2 float decode: a tiny equirect and a 4² cubemap (built with `ktxTexture2_Create` if libktx links into `lathe-tests`, otherwise a checked-in generated fixture under 2 KB) decode to the expected half floats and layer counts; compressed or block-transcoded KTX2 files are refused.

**GPU checks (manual; CI has no GPU)**, in the style of `docs/occlusion-culling.md` "Verifying on a GPU":

1. Validation layers with synchronisation validation clean through startup, every source switch, slider drags during amortized builds, resize, and hot reload of each `env_*.slang` and `skybox.slang`.
2. **White furnace**: the debug view fills the radiance cube with 1.0. Expected: SH reads back as [√π·… L0 only]; a white dielectric (albedo 1, metallic 0) sphere ≈ 1 − small. White metallic spheres read A + B (matching the CPU LUT) with multi-scatter off and ≈ 1 with it on. No sphere exceeds 1.
3. **Readback comparisons**: "Validate against CPU" in the Debug panel copies the SH slot and the 32² radiance faces to a readback buffer, projects them on the CPU with `spherical_harmonics.hxx`, and compares (relative 1e-3). It does the same for 16 LUT texels against the CPU reference (1e-2).
4. **Amortized == immediate**: a screenshot after an amortized build and after "Rebuild now" with the same inputs must be bit-identical (same dispatches, different frames).
5. **Back-compat**: `light_field.lbf` screenshot bit-identical to a pre-change build, since the flat path is unchanged.
6. **Reference images**: a generated, MIT-licensed test HDRI and a roughness × metallic sphere grid scene (`assets/scenes/ibl_spheres.lbf`). The HDRI is `assets/textures/generate_environment.py`, run as `uv run --with numpy --with OpenEXR python ...`: sky gradient, a small bright "sun" patch, coloured walls. Compare against glTF-Sample-Viewer or Filament `cmgen` with the same HDRI (qualitative), and against stored screenshots with `tools/compare_images.py` (`uv run --with numpy --with pillow`; RMSE/FLIP thresholds). Never pip or venv.
7. **RenderDoc** (`include/gpu/renderdoc.hxx`), on a build frame:
   - face orientation and seams in the texture viewer;
   - per-mip roughness progression;
   - barrier and layout sequence in the event browser;
   - skybox fragments only where depth = 0;
   - the sun disc absent from the radiance cube but present in the skybox.
8. Tracy: the Environment and Skybox zones appear, named and coloured. `--benchmark` JSON gets an `environment` stage (`tools/perf/compare_benchmarks.py` lists new stages; run it with `uv run`).

## Performance budget (to confirm on hardware)

| Work | When | Budget (1080p, mid-range desktop GPU) |
|---|---|---|
| Environment stage, idle | every frame | 0 (empty timestamps) |
| BRDF LUT | startup / hot reload | ≤ 0.5 ms once |
| Sky capture 256² + mips + SH + prefilter mip 0 | build frame 0 | ≤ 0.15 ms |
| HDR projection 512² + mips + SH + full prefilter (immediate) | source change | ≤ 1 ms once (plus CPU decode, off thread) |
| Prefilter, one face (mips 1-5, ~0.85 M samples) | amortized frames 1-6 | ≤ 0.1 ms |
| Forward IBL (1 cube fetch, 1 LUT fetch, 9 SH MADs ×3) | every frame | ≤ +5% of Forward Pass |
| Skybox (sky pixels only) | every frame | ≤ 0.1 ms |
| Memory | | ≤ 26 MB resident (HDR 512); transient upload ≤ 128 MB |

## Risks

- **Bright HDRI suns** (> 65504) overflow half floats and cause prefilter fireflies. Mitigated by the clamp on projection plus filtered importance sampling. A baked sun also double-counts with the directional light; the panel warns when `hdr_image` and "Drive directional light" with intensity > 0 are both on. A follow-up "Align sun to HDRI" (brightest-texel direction via a reduction) would help.
- **Descriptor-update timing**: images created after `gpu_resource_table_.prepare_frame` are invisible until the next frame. All creation is in `EnvironmentSystem::prepare` to avoid this.
- **The black cube fallback is required**: binding 5 is statically used by the forward shader, and the table is not `PARTIALLY_BOUND`.
- **Storage format support**: `R16G16_SFLOAT` storage needs `shaderStorageImageExtendedFormats` and the format's `STORAGE_IMAGE` feature. Check at initialize and fall back to RGBA16F. Existing compute shaders write RGBA16F through format-less `RWTexture2D<float4>`; the new ones declare `[format(...)]` explicitly, as `hiz_build.slang` does.
- **Sunrise/sunset shadows**: the light elevation is clamped to 5° while the sky goes to −10°, so shadows stop following the sun near the horizon. Light intensity fades out over that band to hide it.
- **Cascade cache churn**: every sun move redraws all cascades. Acceptable for editing; it matters for any future time-of-day.
- **Preetham accuracy and units**: the model is calibrated for turbidity ~2-6 and loses accuracy at low sun elevations and above the horizon band, and it has no ground term. Both are accepted for v1 and the UI says so; Hosek-Wilkie is a follow-up. Its output is in kcd/m², so it needs the "matches today's ambient" calibration above or every scene switched to the sky changes exposure.
- **N = V = R prefilter error** at grazing angles (stretched reflections are lost). This is known and accepted.
- **Pipeline statistics** now include sky fragments.

## Decisions

Questions settled before implementation:

1. **Radiance cube size**: 512 for HDR equirects (option 1024 in the UI). The equirect is not sampled directly by the skybox.
2. **Sky model**: analytic Preetham, no vendored data. Hosek-Wilkie stays a follow-up.
3. **Fog**: in v1. `fog_from_environment` tints the in-scattered colour with the blurred environment, so the fog follows sunsets and HDRIs.
4. **KTX2 float sources**: in v1 (equirect or cubemap, uncompressed float formats only).
5. **Multi-scatter energy compensation**: in v1, for IBL specular, behind `multi_scatter` (default on in new scenes).
6. **Flat ambient** stays a scalar, which keeps old scenes identical.

## As built

What differs from the design above, and what is not done yet.

**Differences**

- **One face per dispatch.** Capture, downsample and prefilter each dispatch one face (the face index and that face's `storage_2d` slot are push constants), so no shader needs an array of slots. `env_sh_project` and `env_prefilter` read the radiance through the cube binding, and the SH reduction is a groupshared tree rather than wave sums, which keeps it independent of the wave size.
- **BRDF LUT** is RGBA16F (see the resources table).
- **Sky model** is Preetham, with the calibration constant `sky_calibration` computed so the default sun gives today's 0.15 ambient (checked by `test/sky_model_test.cxx`). The sun disc's radiance is clamped to 2000, so it blooms without flooding the frame.
- **Fog** takes its colour from the blurriest prefilter mip (`fog_from_environment`), and the fog settings are saved in the environment section.
- **The KTX2 path** accepts Zstd-supercompressed float files (libktx inflates them on load). `lathe-env-bake` (`tools/env_bake`) writes them from an equirect, and `assets/environments/belfast_sunset_puresky_512.ktx2` is its output.
- **A cooked environment** (`ENVM`) reaches the renderer through `EnvironmentSystem::provide_hdr`, which `instantiate_scene` calls when a pack has the chunk, so loading a saved scene does not need the source file.
- **No nested Skybox GPU zone** (see Pass graph placement).
- **Debug views** `diffuse only`, `specular only` and `specular occlusion` replace the ambient term only; direct light still shows.

**Verified**

- CPU maths, codecs and serde by unit tests (`cube_map`, `spherical_harmonics`, `brdf_lut`, `sky_model`, `hdr_image`, `environment_scene`).
- On a GPU (NVIDIA, debug build with validation layers, procedural sky and the vendored cubemap): no validation messages, and "Validate against CPU" reports the LUT within 5e-4 and the SH within 4e-6 relative of the CPU references.
- `cargo xtask tidy` and `tools/check_shaders.py` pass.

**Not done**

- The roughness x metallic sphere-grid scene, the generated test HDRI and the reference-image comparison against glTF-Sample-Viewer or Filament (`tools/compare_images.py`).
- A bit-identical comparison of an amortized build against "Rebuild now", and a RenderDoc review of face seams.
- Hardware timings for the performance budget; the numbers there are targets, not measurements.
- Per-environment BRDF multi-scatter for punctual and directional lights (a follow-up already listed below).

## Milestones

Each one builds with `cargo xtask build`, passes `cargo xtask test`, and has a GPU check. Build per milestone, not per edit.

- **M1 — Infrastructure and CPU maths.**
  - Bindless binding 5 plus `ImageStorage::black_cube()`.
  - `bindless.slang` `sampled_cube[]`.
  - `cube_map.hxx`, `spherical_harmonics.hxx` with tests.
  - `.hdr` and KTX2 float decode with tests.

  Check: the app renders bit-identically; validation clean.
- **M2 — EnvironmentSystem skeleton and BRDF LUT.**
  - `environment.hxx/.cxx`, `RenderStage::Environment` (+ `to_string`, benchmark name), `record_environment_pass`, empty timestamps.
  - `env_brdf_lut.slang` + reflection line, hot-reload detection.
  - Debug panel showing the LUT; `brdf_lut_test.cxx`.

  Check: LUT readback against the CPU reference; Tracy zone.
- **M3 — HDR environment and skybox.**
  - `request_hdr` (thread pool), equirect upload, radiance cube creation and retirement.
  - `env_equirect_to_cube.slang`, `env_downsample.slang`.
  - `skybox.slang`, `ForwardDynamicStateMode::sky`, `SkyboxDrawInfo` inside `forward_geometry`.
  - UBO env block (static_asserts, slang mirror).
  - KTX2 cube sources: faces uploaded straight into radiance mip 0 (no projection), then the downsample chain.
  - Minimal Environment window (source + browse + rotation + exposure); not saved yet.

  Check: orientation (+Z centre), MSAA edges, depth = 0 only, RenderDoc faces and seams, hot reload.
- **M4 — IBL shading.**
  - `env_sh_project.slang`, `env_prefilter.slang` (immediate builds only), SH buffer.
  - Forward IBL + specular occlusion + multi-scatter compensation + flat fallback, debug views, white furnace, "Validate against CPU".

  Check: furnace, readback tolerances, `light_field.lbf` unchanged (still flat), reference sphere grid against glTF-Sample-Viewer.
- **M5 — Procedural sky and sun link.**
  - `sky_model.hxx/.cxx` (Preetham, no vendored data), tests.
  - Environment-derived fog colour (`fog_from_environment`, `environment_fog_colour`).
  - `env_sky_to_cube.slang`; skybox sky + sun disc.
  - Sun → `DirectionalLight` (direction clamp, transmittance colour, fade).
  - Double-buffered sets, amortized schedule, finish-then-restart, "Amortize rebuilds".

  Check: amortized == immediate bit-identical, sunset colours on sky, light and shadows, per-face cost in the Tracy/benchmark `environment` stage.
- **M6 — Scene serde.**
  - `SceneEnvironment` (including `SceneFog`) on `Scene`/`SceneDescription`, section 14, validation, fingerprint, `play()` copy, `submit_scene` push.
  - `environment_asset_key`, `ENVM` chunk + cook/copy + `AssetPack` loaders, instantiate path (packs, then source).
  - `docs/lathe-binary-format.md` update; lbf tests.

  Check: save/load round trip in the editor; an old engine build loads a new scene (skipped-section warning); `light_field.lbf` unchanged.
- **M7 — Editor polish.**
  - Full Environment window (sun and fog moved out of Lighting, status/progress, face viewer, SH table), removal of the Lighting "Ambient intensity" slider.
  - Generated test HDRI script + `ibl_spheres.lbf` + `tools/compare_images.py`.

  Check: the reference-image comparison passes; benchmark before/after with `compare_benchmarks.py`.
- **Follow-ups**: align sun to HDRI, aerial perspective, multi-scatter for punctual and directional lights, Hosek-Wilkie as a higher-quality sky model.
