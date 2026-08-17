# Migration Plan: `VkPipeline` → `VK_EXT_shader_object`

Scope: `pipeline.hxx/.cxx`, `renderer.hxx/.cxx`, `PipelineGraphRepository`, and every call site that binds a pipeline or sets forward/shadow/composite dynamic state.

---

## Phase 0 — Feasibility gate

- [ ] Confirm `VK_EXT_shader_object` support on all target drivers/GPUs (AMD, NVIDIA, Intel — check current driver versions in CI/dev machines).
- [ ] Confirm `VK_EXT_vertex_input_dynamic_state`, `VK_EXT_extended_dynamic_state{,2,3}` are supported (usually bundled as shader object's required set).
- [ ] Decide fallback story: keep the `VkPipeline` path behind a compile-time or runtime flag until shader objects are validated on all target hardware, or accept shader-object-only going forward.
- [ ] Spike: create one `VkShaderEXT` (e.g. compute `frustum_cull.slang`) end-to-end to validate the build/driver/validation-layer story before touching the rest of the renderer.

**Exit criteria:** a single working `VkShaderEXT` compute dispatch, validation-layer clean.

---

## Phase 1 — Device/context changes

File: wherever `VkPhysicalDeviceFeatures2` chains live (likely `context.cxx`).

- [ ] Add `VkPhysicalDeviceShaderObjectFeaturesEXT` to the feature chain, `shaderObject = VK_TRUE`.
- [ ] Confirm extended-dynamic-state feature structs are already enabled (or add them) — `extendedDynamicState`, `extendedDynamicState2`, `extendedDynamicState3` (specifically `ColorBlendEnable`, `ColorBlendEquation`, `ColorWriteMask`, `RasterizationSamples`, `SampleMask`, `AlphaToCoverageEnable`, `PolygonMode`, `LogicOpEnable` — check `VkPhysicalDeviceExtendedDynamicState3FeaturesEXT` for exact flag names against your Vulkan SDK version).
- [ ] Add `VK_EXT_shader_object`, `VK_EXT_vertex_input_dynamic_state` to the enabled device extension list; verify `VK_EXT_extended_dynamic_state3` is present (may already be enabled given current dynamic-state usage in `pipeline.cxx`).
- [ ] Load extension function pointers if not using a loader that does this automatically (`vkCreateShadersEXT`, `vkDestroyShaderEXT`, `vkCmdBindShadersEXT`, `vkGetShaderBinaryDataEXT`, `vkCmdSetVertexInputEXT`, `vkCmdSetRasterizationSamplesEXT`, `vkCmdSetSampleMaskEXT`, `vkCmdSetAlphaToCoverageEnableEXT`, `vkCmdSetPolygonModeEXT`, `vkCmdSetColorBlendEnableEXT`, `vkCmdSetColorBlendEquationEXT`, `vkCmdSetColorWriteMaskEXT`, `vkCmdSetLogicOpEnableEXT`, `vkCmdSetDepthClampEnableEXT`).

**Exit criteria:** device creation succeeds with all new features/extensions enabled; existing pipeline-based rendering still works unchanged (no behavior change yet).

---

## Phase 2 — New `ShaderObjectSet` type (parallel to `Pipeline`)

New files: `shader_object_set.hxx` / `.cxx` (kept separate from `pipeline.hxx` initially so the old path stays intact for comparison/rollback).

- [ ] Define `ShaderObjectSet`:
  ```cpp
  class ShaderObjectSet {
  public:
      static auto create_linked(VulkanContext &context, ShaderObjectCreateInfo const &create_info,
                                 VkDescriptorSetLayout global_layout)
              -> std::expected<ShaderObjectSet, ShaderObjectError>;

      static auto create_compute(VulkanContext &context, ComputeShaderCreateInfo const &create_info,
                                  VkDescriptorSetLayout global_layout)
              -> std::expected<ShaderObjectSet, ShaderObjectError>;

      auto bind(VkCommandBuffer command_buffer) const noexcept -> void;
      auto layout() const noexcept -> VkPipelineLayout { return layout_; }

      ~ShaderObjectSet();
      ShaderObjectSet(ShaderObjectSet &&) noexcept;
      auto operator=(ShaderObjectSet &&) noexcept -> ShaderObjectSet &;
      ShaderObjectSet(ShaderObjectSet const &) = delete;
      auto operator=(ShaderObjectSet const &) -> ShaderObjectSet & = delete;

  private:
      auto destroy() noexcept -> void;

      VulkanContext *context_ = nullptr;
      std::array<VkShaderEXT, 4> shaders_{}; // max: task, mesh/vertex, geometry(unused), fragment
      std::array<VkShaderStageFlagBits, 4> stages_{};
      std::uint32_t count_ = 0;
      VkPipelineLayout layout_ = VK_NULL_HANDLE;
  };
  ```
- [ ] `create_linked`: build `VkShaderCreateInfoEXT` per stage with `flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT`, correct `nextStage` chaining (vertex→fragment, or task→mesh→fragment), call `vkCreateShadersEXT` once for the whole group.
- [ ] `create_compute`: single unlinked `VkShaderCreateInfoEXT`, `flags = 0`.
- [ ] `bind()`: calls `vkCmdBindShadersEXT` with the full stage array (including `VK_NULL_HANDLE` for any stage in the group not populated — relevant once mesh/vertex paths share a bind helper).
- [ ] Reuse the existing `layouts` (push global + `additional_descriptor_set_layouts`) and `VkPipelineLayoutCreateInfo` logic from `Pipeline::create_graphics` verbatim — layout creation doesn't change.
- [ ] Port `make_error`-style error type (`ShaderObjectError`) mirroring `PipelineError`.
- [ ] Unit/integration test: create a `ShaderObjectSet` for a trivial vertex+fragment pair against a `NullDevice`/headless context if available; assert non-null handles and clean destruction (no validation errors).

**Exit criteria:** `ShaderObjectSet` compiles, creates, and destroys cleanly in isolation; no renderer integration yet.

---

## Phase 3 — Dynamic state helper expansion

File: `renderer.cxx`, anonymous namespace near `set_forward_dynamic_state` / `set_shadow_dynamic_state` / `set_composite_dynamic_state`.

- [ ] Extend each of the three setters with the newly-mandatory dynamic state:
  - `vkCmdSetVertexInputEXT` (replaces `default_bindings()`/`empty_vertex_input` from `pipeline.cxx` — becomes a per-draw toggle, not a pipeline permutation)
  - `vkCmdSetRasterizationSamplesEXT`, `vkCmdSetSampleMaskEXT`, `vkCmdSetAlphaToCoverageEnableEXT`
  - `vkCmdSetPolygonModeEXT`
  - `vkCmdSetColorBlendEnableEXT` / `vkCmdSetColorBlendEquationEXT` / `vkCmdSetColorWriteMaskEXT` (per color attachment — forward pass has 1, shadow/prepass have 0)
  - `vkCmdSetLogicOpEnableEXT`
  - `vkCmdSetDepthClampEnableEXT`
- [ ] Confirm `vkCmdSetRasterizerDiscardEnable`, `vkCmdSetPrimitiveRestartEnable`, `vkCmdSetCullMode`, `vkCmdSetFrontFace`, `vkCmdSetDepthTestEnable/WriteEnable/CompareOp`, `vkCmdSetDepthBiasEnable`, `vkCmdSetStencilTestEnable` are unchanged (already dynamic-state-2, already called) — no action needed here.
- [ ] Write these as pure functions taking the values that currently live in `GraphicsPipelineCreateInfo` (`polygon_mode`, `cull_mode`, `front_face`, `blending`, `samples`, vertex-input on/off) so each render pass site can call them with its own constants, mirroring today's `ForwardDynamicStateMode` enum pattern.
- [ ] Add a `set_mesh_shader_dynamic_state` variant if task/mesh pipelines (light icons) need anything beyond what `set_forward_dynamic_state` already sets — mesh shader pipelines skip vertex input entirely, so `vkCmdSetVertexInputEXT` with zero bindings/attributes (or simply not called if the spec allows omitting it for mesh-only draws — confirm against spec).

**Exit criteria:** dynamic-state setters compile against the new function pointers; not yet invoked from a live shader-object draw path.

---

## Phase 4 — Registry integration (`PipelineGraphRepository`)

This is the highest-risk phase because hot-reload (`on_files_changed`, `process_dirty`) currently assumes `VkPipeline` recompilation semantics.

- [ ] Add a second storage path in the repository: `std::vector<ShaderObjectSet>` alongside (or replacing) `std::vector<Pipeline>`, keyed the same way `register_pipeline` currently returns handles.
- [ ] `register_pipeline` (or a new `register_shader_objects`) takes the same `PipelineRegisterInfo`, builds `ShaderObjectCreateInfo` from the same `ShaderCompileRequest` list, and stores the resulting `ShaderObjectSet`.
- [ ] Hot-reload path: on `on_files_changed`, instead of rebuilding a whole `VkGraphicsPipelineCreateInfo`, call `vkDestroyShaderEXT` on the stale `VkShaderEXT`s for just the changed stage(s) and recreate — confirm whether `VK_SHADER_CREATE_LINK_STAGE_BIT_EXT` requires recreating the *whole linked group* on any single-stage change (likely yes, since linked shaders are validated together) — if so, `process_dirty` keeps its current "rebuild the whole set" granularity, just swapping the rebuild target from `Pipeline` to `ShaderObjectSet`.
- [ ] Decide retirement/lifetime story: current `tick_retirement()` presumably defers destruction of in-flight `VkPipeline`s until safe. Confirm the same deferred-destroy pattern applies to `VkShaderEXT` (it should — same "don't destroy while in-flight command buffers reference it" rule) and reuse the existing retirement queue, just storing shader-object handles instead of pipeline handles.
- [ ] Collapse pipeline variants where dynamic state now covers the difference:
  - `forward_pipeline_` + `forward_blend_pipeline_` → one `ShaderObjectSet`, blend toggled via `vkCmdSetColorBlendEnableEXT` at draw time.
  - `depth_prepass_pipeline_` + `depth_prepass_mask_pipeline_` → these differ by fragment-stage presence (mask has a fragment shader for alpha-cutoff, prepass doesn't) — **cannot** collapse into one shader-object set since stage presence isn't dynamic state; keep as two registered sets, same as today.
  - `shadow_pipeline_` + `shadow_mask_pipeline_` → same reasoning as above, keep separate.
  - Net result: `forward_blend_pipeline_` is retired as a separate registration; everything else keeps its current count but as `ShaderObjectSet`s instead of `Pipeline`s.

**Exit criteria:** repository can register, resolve, hot-reload, and retire a `ShaderObjectSet` end-to-end, verified against one pipeline (start with `composite_pipeline_` — simplest, single color attachment, no depth, no variants).

---

## Phase 5 — Per-pass migration (incremental, one pass at a time)

Migrate `record_frame`/`prepare_frame` call sites one render pass at a time, in increasing order of complexity, keeping the old `Pipeline` path compiled behind a flag until each pass is verified:

1. **Composite pass** (simplest — single draw, no variants, no depth).
   - [ ] Replace `vkCmdBindPipeline(... composite_pipeline->pipeline())` with `composite_shader_objects->bind(command_buffer)`.
   - [ ] Add `vkCmdSetVertexInputEXT` with zero bindings/attributes (fullscreen triangle, no vertex buffer) before `set_composite_dynamic_state`.
   - [ ] Add the new blend/sample/polygon dynamic-state calls per Phase 3.
   - [ ] Visual diff against current output (RenderDoc capture before/after) — pixel-identical expected.

2. **Frustum cull compute pass** (`frustum_cull_pipeline_`).
   - [ ] Compute shaders need far less new dynamic state (none of the rasterization state applies) — mostly a bind-call swap.
   - [ ] Verify push constants (`CullPushConstants`) still land correctly through the shared `VkPipelineLayout`.

3. **Depth prepass** (opaque + mask).
   - [ ] Two `ShaderObjectSet`s as decided in Phase 4.
   - [ ] Add vertex input (via `default_bindings()`'s binding/attribute data, now fed to `vkCmdSetVertexInputEXT` instead of a static `VkPipelineVertexInputStateCreateInfo`).
   - [ ] No color attachments — confirm color-blend dynamic state calls are skippable/no-ops when `colorAttachmentCount == 0` in `VkPipelineRenderingCreateInfo`; check spec/validation layer output here specifically, since this is a common gap.

4. **Shadow pass** (opaque + mask, 4 cascades).
   - [ ] Same shape as depth prepass; verify per-cascade `vkCmdSetViewport`/`vkCmdSetScissor` inside `set_shadow_dynamic_state` still executes correctly alongside the new calls (order: bind shaders once per pipeline switch, but dynamic state re-set per cascade loop iteration — no change to loop structure, only to what's inside `set_shadow_dynamic_state`).

5. **Forward pass** (opaque + mask + blend + light icons).
   - [ ] Migrate `forward_pipeline_` and retire `forward_blend_pipeline_`, using `vkCmdSetColorBlendEnableEXT`/`vkCmdSetColorBlendEquationEXT` to toggle blend between the opaque/mask draw and the blend draw within the same bound shader objects — reduces one `vkCmdBindShadersEXT` call per frame where previously two full pipeline binds were needed.
   - [ ] Migrate `light_icon_pipeline_` (task+mesh+fragment) — confirm `vkCmdSetVertexInputEXT` is skippable for mesh-shader-only draws (no vertex input stage exists in a mesh pipeline); check validation layer messages closely on first run, this is the most likely spec-compliance gap given it's the newest/least-tested part of the extension.
   - [ ] Confirm push-constant stage flags (`VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT`) work identically against the shared `VkPipelineLayout` — no change expected here.

**Exit criteria per pass:** RenderDoc frame capture visually and numerically matches the pre-migration `VkPipeline` output; no new validation errors; frame time delta measured and logged (shader objects are usually neutral-to-slightly-faster once bound state is cached, but confirm — see Phase 6).

---

## Phase 6 — Cleanup and validation

- [ ] Remove `pipeline.hxx/.cxx`'s `Pipeline::create_graphics`/`create_compute` once all call sites are migrated and verified, or keep as a documented fallback behind a build flag for one release cycle.
- [ ] Remove `forward_dynamic_states` array entries that are now redundant with the expanded per-pass setters (audit for duplicates).
- [ ] Re-run full validation-layer pass (`VK_LAYER_KHRONOS_validation`) across every pass with `debug_draw_light_icons_` toggled both ways, MSAA on and off (`is_msaa` branch), and frustum culling toggled both ways (`frustum_culling_enabled_`) — these four booleans multiply out to the actual state-permutation matrix shader objects need to survive.
- [ ] Benchmark: frame time before/after per pass (timestamp queries you already have via `RenderStage` enum and `last_frame_timings_` give this for free — no new instrumentation needed).
- [ ] Optional: wire up `vkGetShaderBinaryDataEXT` to cache compiled shader blobs to disk, replacing the lost `VkPipelineCache` warm-start behavior — worth doing given the existing hot-reload infrastructure already tracks per-shader-file state.

**Exit criteria:** old pipeline path deleted (or flagged deprecated), full validation-clean run across the state matrix above, frame-time parity or improvement documented.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Linked-stage groups may force whole-group shader recreation on hot-reload of a single stage | Confirm via spec/testing early (Phase 4); if true, accept — no worse than today's whole-pipeline rebuild |
| Mesh-shader-only draws + `vkCmdSetVertexInputEXT` interaction unclear | Validate with validation layers before shipping; this is the newest corner of the extension |
| Driver/validation-layer maturity lags pipeline path | Keep RenderDoc as primary debugging tool for this migration; don't rely solely on VVL output |
| No cross-run pipeline cache equivalent | `vkGetShaderBinaryDataEXT` blob caching (Phase 6, optional, can ship later) |
| Retirement/lifetime semantics for `VkShaderEXT` under `tick_retirement()` unverified | Explicit test: destroy a `ShaderObjectSet` mid-flight (multiple frames in flight) and confirm no validation error before relying on it in the hot path |
