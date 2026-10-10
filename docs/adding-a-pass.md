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

1. Register shaders and buffers once, from `on_populate`, through `renderer.game_gpu()` (`GameGpu`). Shaders are
   shader objects (`register_compute`, `register_graphics`); a buffer has one zero-filled copy per frame in flight.
2. Pass names must be unique within the frame (`duplicate_pass_name`), also across slots.
3. In `on_frame_graph`, return early unless `graph.slot()` is the slot you need.
4. `graph.import(handle)` gives this frame's copy of a buffer. `graph.add_compute_pass(name, profile, setup)` takes a
   setup that declares accesses on a `GameComputeBuilder` (`sample`, `read`, `write`, `read_write`, `create_image`,
   `queue`, `side_effect`) and returns a callable taking a `GameComputeContext &`. The profile's label and colour
   become the pass's Tracy zones and GPU timestamp.
5. In the callable, `bind` a shader, `push` constants, and get indices and addresses only from the `Declared*` values
   the builder returned. `dispatch_threads(x, group_x)` rounds up and refuses a count the device cannot dispatch.
6. To draw into the scene, `graph.add_scene_draw(name, setup)`: the setup declares the buffers it reads
   (`GameDrawBuilder::read`) and returns a callable taking a `GameDrawContext &`. The draw runs inside the forward
   pass, and declaring the read is what orders it after the passes that wrote the buffer. Only possible in
   `frame_start` and `after_depth`.
7. A post-process writes a game image in `after_lighting` or `before_composite` and calls
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

`game/src/particle_field.cxx` is a complete example: a compute pass simulates particles in a game buffer and a scene
draw renders them, with shaders in `assets/shaders/game/`.

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
