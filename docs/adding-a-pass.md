# Adding a frame graph pass

The frame graph is rebuilt every frame in `Renderer::record_frame`
(`src/rendering/renderer_frame_graph.cxx`). Declaration order is the schedule: a pass goes after the passes that
produce what it reads and before the ones that consume what it writes. Barriers, queue assignment, culling and
transient memory are derived from what a pass declares, so everything a pass touches has to be declared.

## From a game

Games never see the `FrameGraph`. Override `IGame::on_frame_graph(GameGraph &, float)`; the engine calls it once per
`GameSlot` while it declares the frame (`include/rendering/game_graph.hxx`).

| Slot               | Declared after                      | Engine exports               |
| ------------------ | ----------------------------------- | ---------------------------- |
| `frame_start`      | the overlay prepare and environment | none                         |
| `after_depth`      | the late depth prepass              | `scene_depth()`              |
| `after_lighting`   | the forward pass                    | `scene_depth()`, `scene_hdr()` |
| `before_composite` | bloom                               | `scene_depth()`, `scene_hdr()` |

1. Register shaders once, from `on_populate`, through `renderer.game_gpu()` (`GameGpu`). Shaders are shader objects
   (`register_compute`, `register_graphics`). Buffers are not registered: the graph creates them (below).
2. Pass names must be unique within the frame (`duplicate_pass_name`), also across slots.
3. In `on_frame_graph`, return early unless `graph.slot()` is the slot you need.
4. Buffers are asked for from the graph, by name, and the engine owns the memory:
   - `graph.create_buffer({.size = n, .zero = true}, "name")` lives for this frame's graph. The engine reuses a pooled
     buffer of at least that size; it is not aliased with other resources (the transient allocator only handles
     images). Without `zero` the contents are undefined and a read before any write rejects the slot; with it a clear
     pass `"<name>_clear"` is added, so keep names unique.
   - `graph.persistent_buffer("name", {.size = n})` keeps its contents from frame to frame and is zero-filled when
     created. Asking for another size recreates it (zero-filled) and destroys the old one only after the frames in
     flight are done with it. Frames in flight share it, so every pass that touches it is put on the graphics queue
     and its first use waits for the previous frame's work.
   Sizes are positive multiples of four. Neither call returns a `VkBuffer`.
5. The usual way to add a compute pass is a parameter struct, which is both the access declaration and the push
   constants:

   ```cpp
   struct SimParams {
       BufferReadWrite particles;      // read + write; pushes the buffer address
       ImageRead depth;                // sampled; pushes the bindless index (from an EngineImage or GameImage)
       ImageWrite output;              // storage write; built from a GameImage or a GameImageDesc to create
       std::uint32_t count = 0;        // any other member pushes itself
       using Layout = PushLayout<&SimParams::particles, &SimParams::depth, &SimParams::output, &SimParams::count>;
   };
   graph.add_compute("game_sim", params, shader, Threads{.x = count, .group_x = 64});
   ```

   The members in `Layout`, in that order, are the push constants, laid out like the equivalent C++ struct (at most
   `max_game_push_bytes`); the shader declares the same fields. Declared reads and writes, barriers, the queue
   (`compute_preferred`, or graphics for a persistent buffer) and the bindless indices all come from the struct, and
   the resource members of `params` are advanced to the versions the pass wrote, so they wire the next pass. Pass a
   record callable `(GameComputeContext &, P const &)` to dispatch several times with `context.dispatch()`, and
   `GameComputeOptions` for the queue, `side_effect` and the label.
6. The lower level is still there: `graph.add_compute_pass(name, profile, setup)` takes a setup that declares accesses
   on a `GameComputeBuilder` (`sample`, `read`, `write`, `read_write`, `create_image`, `queue`, `side_effect`) and
   returns a callable taking a `GameComputeContext &`. In the callable, `bind` a shader, `push` constants, and get
   indices and addresses only from the `Declared*` values the builder returned.
   `dispatch_threads(x, group_x)` rounds up and refuses a count the device cannot dispatch.
7. To draw into the scene, `graph.add_scene_draw(name, setup)`: the setup declares the buffers it reads
   (`GameDrawBuilder::read`) and returns a callable taking a `GameDrawContext &`. The draw runs inside the forward
   pass, and declaring the read is what orders it after the passes that wrote the buffer. Only possible in
   `frame_start` and `after_depth`.
8. A post-process writes a game image in `after_lighting` or `before_composite` and calls
   `graph.replace_scene_colour(image)`; composition then samples that image instead of the HDR image.

What the graph enforces, on the CPU, every frame:

- engine resources reach a game as `EngineImage`, which only the `sample` overload accepts; a forged id is rejected
  with `foreign_write`;
- a game transient must have the descriptor views (`sampled_2d`, `storage_2d`) and format its uses need
  (`unsupported_usage`), a version that is still current (`stale_version`) and a write before any read
  (`read_before_write`);
- a record function that looks up a resource its pass did not declare gets a violation: the pass stops recording and
  the problem is logged once;
- a slot whose declarations are rejected is rolled back and the engine frame renders without the game's passes;
- on every plan change `frame_graph::lint` warns about game passes that were culled, wrote a transient nobody reads or
  declared nothing.

What it cannot prove is what a shader does with a bindless index or an address it was given. Run a Debug build (sync
validation is on) and look for `[E] Vulkan validation` in the log.

`src/rendering/particle_system.cxx` is a complete example, and it is built on exactly this API: per
`Components::ParticleEmitter` a parameter-struct compute pass simulates the particles in a persistent buffer and a
scene draw renders them, with shaders in `assets/shaders/particles_*.slang`. Lua only sets the component's data
(`entity:add_particles{...}`); the engine owns the passes and buffers.

A piece of a slot that should not take the others down with it (an emitter, an effect) is declared inside
`graph.isolated(name, fn)`: if only that piece is rejected, only it is rolled back, and the reason is returned and
logged once. For a pass whose resources are only known at run time there is `add_compute(name, DynamicParams &, ...)`:
the same declaration and push-constant rules, built from a list of `BufferRead`/`ImageWrite`/`PushBytes` values.

## From Lua

Lua never sees a pass, a handle or a buffer; it attaches data, and the engine declares the passes through the API above.

- **Particles.** `entity.add_particles(e, {count = 2000, rate = 500, lifetime = 2, gravity = {0, -9.8, 0}, shape = "cone",
  speed = 5, size_start = 0.1, colour_start = {1, 0.8, 0.4, 1}, material = m, ...})`, `entity.set_particles(e, {...})`
  (only the fields given change) and `entity.remove_particles(e)`; the same calls also exist as entity methods
  (`e:add_particles{...}`). Every field is checked (`scripting/lua_particles.cxx`) and a
  bad one is a Lua error. `Components::ParticleEmitter` is saved with the scene (its material too; a missing one loads as none) and has an
  inspector section. Particles are alpha-blended, depth tested, no depth writes, drawn last in the forward pass, after
  the engine's own opaque and transparent draws and unsorted against them;
  `ParticleSystem` (`rendering/particle_system.cxx`) simulates and draws it. Example: `assets/scripts/particles/main.lua`.
- **Compute effects.** `compute.load("assets/shaders/effects/tint.json")` reads a manifest
  (`rendering/effect_manifest.hxx` describes the format): the shader, its image and buffer bindings, its named params
  with types and ranges, and the dispatch rule. `compute.instance(shader, {strength = 0.5, tint = {1, 0.6, 0.3}})` makes
  an effect, `fx:set(name, value)` changes it, `scene.add_effect(fx, "before_composite")` runs it in a slot
  (`frame_start`, `after_depth`, `after_lighting`, `before_composite`) and `scene.remove_effect(fx)` takes it out.
  `compute.buffer(n)` is an opaque run of `n` floats that only effects can bind. An effect whose output `replaces`
  the scene colour feeds the next effect and composition. Example: `assets/scripts/effects/main.lua`.
- **Reloading.** The shader source is reloaded by the engine's shader watcher like any other shader (effect shaders
  live under `assets/shaders`, which is watched when not in player mode). The manifest is re-read when its file changes
  (checked twice a second), or with `compute.reload(shader)`; a manifest that no longer parses leaves the old one in
  place, and instances keep the values that still fit.
- **Failure.** Bad names, types, ranges and slots are Lua errors. An effect the graph rejects is dropped from that
  frame alone and `fx:problem()` says why. The shader itself is trusted (it is an engine asset, never script text):
  the manifest is the contract for its push constants, std430-laid-out, and nothing checks the shader against it.

## In the engine

1. Register the shaders (`PipelineRegisterInfo` with `global_push_constant_range`) during renderer init.
2. Add the pass in `record_frame` at the right position, with a unique name and a `PassProfile` (`name_id`, `label`,
   Tracy colour). Inside helpers use `ZoneScopedNC` / `TracyVkZoneC`.
3. Declare every image and buffer the shader touches with the right `Use` and `ShaderStages`, and reassign the handle
   on every write. A shader that samples an image through a bindless index must still declare the read (composition
   once did not, which left the HDR image unsynchronised and its memory free for reuse).
4. Create transients with `pass.create`; include `sampled_2d` / `storage_2d` in `descriptor_views` when shaders reach
   them through the bindless table. The first access must be a write.
5. `pass.queue(QueueAffinity::compute_preferred)` for overlappable compute. Make sure something reads the output or
   call `pass.side_effect()`, otherwise the pass is culled silently.
6. Rebuild the chain in a test (see `test/frame_graph_skin_test.cxx`, `test/frame_graph_game_boundary_test.cxx`) and
   run `test::check_happens_before` on every topology.
7. Run with `--frame-graph-dump` and inspect the pass in the Frame graph panel.
