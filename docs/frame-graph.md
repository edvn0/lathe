# Frame Graph with Async Compute: Implementation Plan

## Objective

`Renderer::record_frame` (`src/rendering/renderer.cxx:4790`) and the culling and light-clustering regions of `Renderer::prepare_frame` (`renderer.cxx:2510`, `#pragma region Culling` / `LightClustering` at about 3337 and 3449) put about a dozen `record_*` functions in order by hand. Each one has hand-written `VkBufferMemoryBarrier2` / `VkImageMemoryBarrier2` / `transition_image_layout` calls, and timestamps come from fixed `RenderStage * 2` query slots. This plan replaces that with a **FrameGraph**. (The name avoids confusion with `PipelineGraphRepository`, which is the shader hot-reload DAG.) The FrameGraph:

1. Takes passes that declare their image and buffer accesses. From those declarations it derives barriers, layout transitions and execution order, and it culls passes nobody consumes. `write_empty_stage` and the "always write both timestamps" rule go away.
2. Schedules passes across a **graphics queue and an async compute queue**. The compiler derives queue-family ownership transfers (release/acquire pairs for images and buffers), cross-queue dependencies using one timeline semaphore per queue, and submission batches split at those dependencies. Each queue gets its own command pools, timestamp pools and Tracy contexts. When the device has no separate compute queue, the same graph compiles with every pass assigned to graphics: there is one code path and only the queue assignment changes.
3. Manages **transient** resources, with lifetime analysis and memory aliasing that accounts for queue overlap. Aliasing can be turned off with a debug switch. It also handles **imported** resources (the swapchain, viewport target, shadow atlas, Hi-Z history, per-frame buffers) with a declared entry and exit state.
4. Derives per-pass, per-queue GPU timestamps and named, coloured Tracy zones from pass names. This replaces `RenderStage`.

The compiler is a pure function from a graph description to a compiled plan. It uses Vulkan enum and flag types as plain data but never touches a device, so `cargo xtask test` can unit-test it exhaustively.

**The order of phases matters.** The compiler (Phase 1) and the queue/submission infrastructure (Phase 2) are independent of each other. Both must be in place before the recording backend (Phase 3). Pass migration (Phase 4) runs every pass on graphics. Async compute is only switched on (Phase 6) after aliasing (Phase 5) has landed. That way every step before Phase 6 can be checked against bit-identical screenshots on a single queue, and each Phase 6 change is a one-line affinity edit that can be measured on its own.

## Confirmed constraints (verified against source; the brief is corrected where it was wrong)

- **Frame structure.** `main.cxx` `draw()` (about line 210) calls `Swapchain::begin_frame()`. That waits on the slot's `in_flight` fence, acquires, resets the slot's single command buffer and begins it. Next come `TerrainSystem::process_ready` (uploads), `submit_scene`, ImGui `begin_frame`/`on_ui`/`end_frame`, `Renderer::prepare_frame` and `Renderer::record_frame`, all recording into **that one command buffer**. Finally `Swapchain::end_frame` (`src/gpu/swapchain.cxx:207`) does a single `vkQueueSubmit2` on the graphics queue: it waits on `image_available` at `COLOR_ATTACHMENT_OUTPUT`, signals the per-image binary `render_finished`, signals the `in_flight` fence, then calls `vkQueuePresentKHR`. `frames_in_flight = 2` (`include/core/config.hxx`). So a lot of non-graph work (uploads, texture/model streaming, `image_storage_.prepare_frame`, `environment_.prepare`, `material_storage_.prepare_frame`, `upload_frame_data`) precedes the passes. **That work becomes the graphics "prologue" and stays outside the graph.**
- **GPU culling (`main_cs`) and light clustering are recorded in `prepare_frame`, not `record_frame`.** They write `RenderStage::Culling` / `LightClustering` timestamps themselves. The occlusion-stats and meshlet-bitset `vkCmdFillBuffer` clears and their barriers are also in `prepare_frame` (about lines 2955 and 3270).
- **Barrier sites.** `renderer.cxx` has 13 `vkCmdPipelineBarrier2` sites and `render_passes.cxx` has 20 barrier or `transition_image_layout` sites. Together that is about 33 in the frame path, not about 25. `environment.cxx` has 19 more, internal to the IBL builds, and `screenshot.cxx` has 2. The 10 in `image_storage.cxx` belong to the upload path and stay outside the graph. Buffer barriers are already whole-buffer (`VK_WHOLE_SIZE`) everywhere except `meshlet_visibility_buffer` (sized) and `cluster_lights_buffer`, which is split into a stats range and a lists range with different consumers.
- **Timestamps.** There is one timestamp pool **per frame in flight** (`timestamp_queries_[frame]`), sized `total_query_count = stage_count*2 + max_overlays*4` (`renderer.cxx:43`), plus a pipeline-statistics pool per frame. Pools are reset with `vkCmdResetQueryPool` at the top of `prepare_frame`, although `hostQueryReset` is enabled. `write_empty_stage` is used **only** for the three occlusion stages. Shadow, bloom and AO write their own begin/end pair when they skip. Stage timestamps are written **inside** `render_pass::*` bodies (`render_passes.cxx:555, 736, 828, 1035, 1252, 1373`). Consumers of `RenderStage`/`stage_count`: `application.cxx` (timings plot, about 2060), `application.hxx` (`timing_buffers`), `main.cxx` (about 290), `benchmark.cxx`/`.hxx` (`benchmark_stage_id`, which produces the stable JSON ids `gpu_culling`, `depth_prepass`, ...), `render_passes.cxx/.hxx` and `overlay.hxx`.
- **Tracy.** There is one `HostQueryContext` (`src/gpu/host_query_context.cxx`). It is host-calibrated when possible; otherwise it falls back to `TracyVkContext[Calibrated]` on `graphics_queue` with `one_time_command_buffers[0]`. Only `TracyVkCollectHost` is ever called (`renderer.cxx:4909`). The comment at `renderer.cxx:4106` notes that Tracy zone names must be literals, and overlays use `TracyVkZoneTransient`.
- **Queues.** `QueueFamilies` has only `graphics`/`present` (`include/gpu/context.hxx`). `find_queue_families` (`vulkan_bootstrap.cxx:418`) never looks for a compute family. `create_device` (`vulkan_bootstrap.cxx:680`) creates one queue for each unique family out of {graphics, present}. **`timelineSemaphore` is not enabled** (`vulkan12_features`, about line 724) and device selection does not require it. `one_time_submit` (`src/gpu/context.cxx:59`) uses the graphics queue and a fence. All images (`image.cxx:173`) and buffers (`buffer.cxx:265`) are `VK_SHARING_MODE_EXCLUSIVE`. The swapchain path never transfers ownership to the present family when it differs from graphics; that is an existing gap.
- **Resources.** `gpu::Image` does not track layout. Per-frame-slot resources are `RendererFrame` (`renderer.hxx:802`) buffers plus `forward_target`, `viewport_target`, `bloom_target` (image plus per-mip bindless slots) and `ao_target` (raw/denoised), all built by `create_frame_targets` (`renderer.cxx:4913`). `shadow_atlas_` is one image shared by both slots; it is LOADed across frames (`preserve_contents = shadow_atlas_initialized_`). `hiz_` (`HizPyramid`, `renderer.hxx:951`) is **one pyramid shared by both frames in flight**. Its comment depends on "one graphics queue and in-order submission", and **that assumption breaks under async compute**. `resize()` (`renderer.cxx:5053`) calls `wait_idle()` and rebuilds everything that depends on extent. `ImageStorage` has no deferred destruction.
- **Device addresses.** Almost every pass reaches buffers through `VkDeviceAddress` push constants (`CullPushConstants`, `DepthPrepassInfo::*_address`, the UBO, lights, clusters), so buffer hazards can only be tracked per whole buffer. **Synchronization validation cannot see buffer-device-address accesses**, so it cannot be the only check for buffer hazards (see Risks).
- **Overlays and ImGui.** `OverlayDesc::prepare` records arbitrary GPU writes at the start of `record_frame`, followed by one global memory barrier (`record_overlay_prepares`, `renderer.cxx:4612`). Scene overlays draw inside forward's rendering scope; UI overlays draw inside composite or `ui_only`. ImGui samples bindless images by index: the viewport target (`application.cxx:481`), the Hi-Z debug view (`application.cxx:2194`), environment-panel cubes and the BRDF LUT (`environment_panel.cxx:50, 215`), and asset thumbnails. `imgui_renderer.cxx:423` already extracts the per-draw-command `texture_id`.
- **Sync validation** is not enabled anywhere. Only `VK_LAYER_KHRONOS_validation` is listed in Debug builds (`vulkan_bootstrap.cxx:24`), with no `validate_sync` setting.
- **Tests.** `test/CMakeLists.txt` globs `*_test.cxx` into `lathe-tests`, which links `lathe::core`, an umbrella over all module libraries. New tests need no CMake edit. New sources go into `src/rendering/CMakeLists.txt` (`engine_rendering`) or `src/gpu/CMakeLists.txt` (`engine_gpu`). The tests use doctest `TEST_SUITE("unit")` with `DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS`.
- **Benchmarks.** The benchmark JSON records per-stage ids and the `occlusion_culling`/`meshlet_occlusion` flags, but **no occlusion statistics**. Those only appear in the Scene stats panel.

## Design overview

### Module layout

```
include/rendering/frame_graph/
  types.hxx            LogicalQueue, QueueAffinity, PassType, Use, LoadOp/StoreOp, handles, ResourceState
  use_table.hxx        constexpr Use -> {stages, access, layout, is_write, discards}
  frame_graph.hxx      FrameGraph (per-frame builder), PassBuilder, PassContext
  compiled_graph.hxx   CompiledGraph, Batch, BarrierSet, OwnershipTransfer, TransientPlan
  compiler.hxx         QueueTopology, CompileOptions, compile()
  frame_graph_error.hxx FrameGraphError{type, pass, resource} + std::formatter specialisations
  aliasing.hxx         pure transient memory planner (Phase 5)
  executor.hxx         Vulkan recording backend (Phase 3)
  pass_profiler.hxx    per-queue timestamp pools + interned Tracy source locations (Phase 3)
  transient_allocator.hxx VMA-backed transient images (Phase 5)
src/rendering/frame_graph/
  frame_graph.cxx compiler.cxx aliasing.cxx executor.cxx pass_profiler.cxx transient_allocator.cxx vk_translate.cxx
include/gpu/queue_set.hxx, src/gpu/queue_set.cxx      (Phase 2)
src/rendering/renderer_frame_graph.cxx                 (Phase 4: Renderer::build_frame_graph + add_*_pass)
```

`frame_graph.cxx`, `compiler.cxx` and `aliasing.cxx` include only `<volk.h>` (for enum and flag types), `core/handle.hxx` and the STL. They must not include `gpu/image.hxx`, `gpu/buffer.hxx`, the renderer or the context. Phase 1 enforces this with a test translation unit that includes only `rendering/frame_graph/compiler.hxx`.

### Core types (sketch)

```cpp
namespace frame_graph {

enum class LogicalQueue : std::uint8_t { graphics, compute, count };
inline constexpr std::size_t logical_queue_count = 2;

// graphics: always graphics.
// compute_preferred: async compute when available and enabled, unless the compiler demotes it because no graphics
//   pass can overlap it (see Compiler step 3).
// compute_required: async compute whenever available and enabled; never demoted (measurement/forcing).
// Both compute affinities fall back to graphics on a single-queue topology.
enum class QueueAffinity : std::uint8_t { graphics, compute_preferred, compute_required };

enum class PassType : std::uint8_t { raster, compute, transfer };

enum class Use : std::uint8_t {
    // images
    color_attachment, color_resolve, depth_attachment, depth_resolve,
    sampled, storage_read, storage_write, storage_read_write,
    transfer_src, transfer_dst, present,
    // buffers (always whole-buffer)
    indirect_read, index_read, shader_read, shader_write, shader_read_write,
    transfer_read, transfer_write, host_read,
    // token resources: no VkImage/VkBuffer; produce VkMemoryBarrier2 only
    token_write, token_read,
};

// Shader stages that a sampled/storage/shader_* use touches; mapped to VkPipelineStageFlags2.
enum class ShaderStage : std::uint8_t { vertex = 1, task = 2, mesh = 4, fragment = 8, compute = 16 };
using ShaderStages = std::uint8_t;

struct ImageTag;  struct BufferTag;  struct PassTag;
using ImageId  = Handle<ImageTag, 0>;   // generation = resource version (1-based); index = resource slot
using BufferId = Handle<BufferTag, 0>;
using PassId   = Handle<PassTag, 0>;

struct ResourceState {          // used for import entry/exit and as the compiler's tracked state
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;   // ignored for buffers
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    LogicalQueue queue = LogicalQueue::graphics;
};

enum class Sharing : std::uint8_t { exclusive, concurrent };

struct TransientImageDesc {
    VkFormat format; VkExtent3D extent; std::uint32_t mip_levels = 1, array_layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT; VkImageType type = VK_IMAGE_TYPE_2D;
    std::string_view debug_name;   // usage is derived from declared Uses
};

struct ImportedImageDesc {
    PhysicalImage physical;        // VkImage, default view, per-mip views, aspect, bindless index; opaque to the compiler
    ResourceState entry;           // state the graph may assume at frame start (prologue already applied)
    ResourceState exit;            // state the graph must leave; exit.queue must be graphics (validated)
    Sharing sharing = Sharing::exclusive;
    std::string_view debug_name;
};
// ImportedBufferDesc: same, with PhysicalBuffer{VkBuffer, VkDeviceAddress, size}.

using RecordFn = std::move_only_function<std::expected<void, RendererError>(PassContext &)>;

class PassBuilder {
public:
    auto queue(QueueAffinity affinity) -> void;                        // the one-line async switch
    [[nodiscard]] auto read(ImageId, Use, ShaderStages = 0) -> ImageId;
    [[nodiscard]] auto write(ImageId, Use, ShaderStages = 0, ExitUse = {}) -> ImageId; // returns next version
    [[nodiscard]] auto read(BufferId, Use, ShaderStages = 0) -> BufferId;
    [[nodiscard]] auto write(BufferId, Use, ShaderStages = 0) -> BufferId;
    [[nodiscard]] auto color(ImageId, LoadOp, StoreOp, std::optional<ResolveTarget> = {}) -> ImageId;
    [[nodiscard]] auto write_depth(ImageId, LoadOp, StoreOp, std::optional<DepthResolve> = {}) -> ImageId;
    [[nodiscard]] auto create(TransientImageDesc const &) -> ImageId;   // transient, version 0 (unwritten)
    auto side_effect() -> void;    // never culled (readbacks, stats copies, screenshot)
    auto legacy() -> void;         // Phase 4 shim: global ALL_COMMANDS barrier before and after
};

struct PassContext {
    VkCommandBuffer command_buffer;
    std::uint32_t frame_index;
    LogicalQueue queue;
    [[nodiscard]] auto image(ImageId) const -> PhysicalImage const &;
    [[nodiscard]] auto buffer(BufferId) const -> PhysicalBuffer const &;
    auto profile_scope(std::string_view name) -> ScopedTimestamp;      // overlay sub-timings
};

class FrameGraph {
public:
    auto reset() -> void;                                           // clear(), keeps capacity
    [[nodiscard]] auto import_image(ImportedImageDesc const &) -> ImageId;
    [[nodiscard]] auto import_buffer(ImportedBufferDesc const &) -> BufferId;
    [[nodiscard]] auto import_token(std::string_view name, ResourceState entry, ResourceState exit) -> BufferId;
    template<std::invocable<PassBuilder &> Setup>                   // Setup returns RecordFn
    auto add_pass(std::string_view name, PassType type, PassProfile profile, Setup &&setup) -> PassId;
    [[nodiscard]] auto description() const -> GraphDesc const &;    // what compile() consumes
};
}
```

Key decisions:

- **Versioned handles.** Every `write` returns a new version. Reading or writing a version older than the latest is an error (`stale_version`). This makes declaration order the authoritative execution order and gives culling exact producer→consumer edges.
- **Declaration order is the baseline schedule; the compiler may reorder to create overlap.** Declaration order is always a valid schedule, and Phases 0-5 compile in it so output stays bit-identical. From Phase 6 a list scheduler may hoist independent passes into the gap between a cross-queue producer and its consumer. It only reorders where the dependency edges allow it, never across passes with side effects (`legacy()` passes, overlays, readbacks, anything flagged `pinned`), and `SchedulerMode::declaration_order` disables it for bisecting. The scheduler is a pure function of the dependency graph, so it is unit-tested without Vulkan (see the Phase 1 test list).
- **`PassProfile { std::string_view name_id; std::string_view label; std::uint32_t color; }`.** `name_id` is the stable benchmark id (`depth_prepass`, matching today's `benchmark_stage_id` strings). `label` is the UI/Tracy text ("Depth prepass"). `color` is a `tracy::Color`.
- **`ExitUse`** lets a pass that manages subresources internally (bloom and Hi-Z per-mip chains, environment builds) say "I enter as `storage_write`, I leave as `sampled`". The graph tracks **whole-resource state only**. Barriers inside one pass on resources that pass exclusively uses are allowed and stay in the pass body.
- **Token resources** replace ad-hoc global barriers (overlay `prepare()` writes → scene/UI overlay reads). They compile to `VkMemoryBarrier2` and never need an ownership transfer, because overlays are graphics-only (validated).
- **Imported resources leave the frame on the graphics queue family** (validated: `exit.queue == graphics`). This rule is what keeps cross-frame history, including Hi-Z, correct without a cross-frame timeline wait (see Compiler step 6 and Risks).

### Use table (`use_table.hxx`, constexpr)

| Use | Stages | Access | Layout | Write | Discards contents |
|---|---|---|---|---|---|
| color_attachment (LOAD) | COLOR_ATTACHMENT_OUTPUT | COLOR_ATTACHMENT_READ\|WRITE | COLOR_ATTACHMENT_OPTIMAL | yes | no |
| color_attachment (CLEAR/DONT_CARE) | COLOR_ATTACHMENT_OUTPUT | COLOR_ATTACHMENT_WRITE | COLOR_ATTACHMENT_OPTIMAL | yes | yes |
| color_resolve | COLOR_ATTACHMENT_OUTPUT | COLOR_ATTACHMENT_WRITE | COLOR_ATTACHMENT_OPTIMAL | yes | yes |
| depth_attachment | EARLY\|LATE_FRAGMENT_TESTS | DEPTH_STENCIL_ATTACHMENT_READ\|WRITE | DEPTH_ATTACHMENT_OPTIMAL | yes | if CLEAR/DONT_CARE |
| depth_resolve | COLOR_ATTACHMENT_OUTPUT\|LATE_FRAGMENT_TESTS | DEPTH_STENCIL_ATTACHMENT_WRITE\|COLOR_ATTACHMENT_WRITE | DEPTH_ATTACHMENT_OPTIMAL | yes | yes |
| sampled(stages) | from stages | SHADER_SAMPLED_READ | SHADER_READ_ONLY_OPTIMAL | no | no |
| storage_read / write / read_write(stages) | from stages | SHADER_STORAGE_READ / WRITE / both | GENERAL | per kind | write: yes |
| transfer_src / transfer_dst | ALL_TRANSFER | TRANSFER_READ / WRITE | TRANSFER_SRC/DST_OPTIMAL | dst | dst: yes |
| present | NONE | NONE | PRESENT_SRC_KHR | no | no |
| indirect_read | DRAW_INDIRECT | INDIRECT_COMMAND_READ | n/a | no | no |
| shader_read / write / read_write(stages) | from stages | SHADER_STORAGE_READ / WRITE (BDA accesses are storage accesses) | n/a | per kind | no (buffers never discard by default) |
| transfer_read / transfer_write | ALL_TRANSFER | TRANSFER_READ / WRITE | n/a | write | no |
| host_read | HOST | HOST_READ | n/a | no | no |

`depth_resolve` deliberately uses the union of stages that the existing code relies on ("depth resolves run in `COLOR_ATTACHMENT_OUTPUT`", `docs/occlusion-culling.md`). Sync validation decides whether it can be narrowed. A buffer write can be flagged `discard` explicitly (`write(id, Use::shader_write, stages, ExitUse{}, Discard::yes)`) when the pass overwrites the whole buffer; this lets the compiler skip an ownership transfer.

### Compiled plan (sketch)

```cpp
enum class OwnershipOp : std::uint8_t { none, release, acquire };

struct ImageBarrier {
    std::uint32_t resource;  // slot index
    VkPipelineStageFlags2 src_stages; VkAccessFlags2 src_access;
    VkPipelineStageFlags2 dst_stages; VkAccessFlags2 dst_access;
    VkImageLayout old_layout, new_layout;
    std::uint32_t src_family = VK_QUEUE_FAMILY_IGNORED, dst_family = VK_QUEUE_FAMILY_IGNORED;
    OwnershipOp op = OwnershipOp::none;
};
struct BufferBarrier { /* same minus layouts; offset 0, size VK_WHOLE_SIZE always */ };
struct MemoryBarrier { VkPipelineStageFlags2 src_stages; VkAccessFlags2 src_access; VkPipelineStageFlags2 dst_stages; VkAccessFlags2 dst_access; };
struct BarrierSet { std::vector<ImageBarrier> images; std::vector<BufferBarrier> buffers; std::vector<MemoryBarrier> memory; };

struct CompiledPass {
    std::uint32_t pass;                       // index into GraphDesc::passes
    BarrierSet before;                        // one vkCmdPipelineBarrier2
    std::optional<RenderingDesc> rendering;   // raster: attachments, load/store, resolve modes, render area
    std::uint32_t timestamp_slot;             // begin = 2*slot, end = 2*slot+1, in this queue's pool
};

struct SemaphoreWait { LogicalQueue queue; std::uint32_t signal_index; VkPipelineStageFlags2 stages; };

struct Batch {
    LogicalQueue queue;
    BarrierSet acquires;                      // recorded first: ownership acquires on entry
    std::vector<CompiledPass> passes;
    BarrierSet releases;                      // recorded last: ownership releases + exit transitions (epilogue)
    std::vector<SemaphoreWait> waits;         // at most one per other queue (max value)
    std::uint32_t signal_index;               // every batch signals its queue's timeline
    bool is_prologue = false;                 // graphics batch 0: prologue commands recorded before it
    bool waits_swapchain_acquire = false;     // first batch that touches the swapchain image
    VkPipelineStageFlags2 swapchain_wait_stages = VK_PIPELINE_STAGE_2_NONE;
    bool signals_render_finished = false;     // last graphics batch
};

struct OwnershipTransfer {                    // also kept for tests and the debug view
    std::uint32_t resource; bool is_image;
    LogicalQueue from, to; std::uint32_t release_batch, acquire_batch;
    VkImageLayout old_layout, new_layout;
};

struct CompiledGraph {
    std::vector<Batch> batches;               // submission order: every wait refers to an earlier batch
    std::vector<OwnershipTransfer> transfers;
    std::vector<bool> pass_culled;
    std::array<std::uint32_t, logical_queue_count> signal_count{};
    std::array<std::vector<std::uint32_t>, logical_queue_count> timestamp_passes; // slot -> pass
    std::vector<ImageUsageFlagsPerResource> derived_usage;  // VkImageUsageFlags/VkBufferUsageFlags per transient
    TransientPlan transients;                 // Phase 5
    std::uint64_t hash;                       // declarations only, not lambdas
};
```

**Timeline values.** `signal_index` is relative to the frame. At submission, `QueueSet` turns it into an absolute value: `value = queue_base[q] + signal_index + 1`. After submitting, `queue_base[q] += signal_count[q]`. The value of the slot's last batch on each queue is stored for the host-side slot wait.

### Queue topology and fallback

```cpp
struct QueueTopology {
    std::array<std::uint32_t, logical_queue_count> family{};      // Vulkan family index per logical queue
    std::array<std::uint32_t, logical_queue_count> queue_index{};  // index within that family
    [[nodiscard]] auto same_queue(LogicalQueue a, LogicalQueue b) const -> bool;   // family && index equal
    [[nodiscard]] auto same_family(LogicalQueue a, LogicalQueue b) const -> bool;
};
struct CompileOptions { bool async_compute = true; bool alias_transients = true; bool serialize = false; };
```

There are three topologies, and they all use the same compiler:

1. **Dedicated family** (COMPUTE without GRAPHICS): semaphores plus ownership transfers.
2. **Same family, second queue** (graphics family with `queueCount >= 2`): semaphores, no ownership transfers (`*_family = IGNORED`).
3. **Single queue** (or `async_compute = false`): the compiler resolves every pass to `LogicalQueue::graphics` before anything else. Batches only split at the prologue, there are no semaphores and no ownership transfers.

`serialize = true` is a debug mode that emits full `ALL_COMMANDS` barriers between all passes, for A/B checks of hazards that validation cannot see.

### Compiler algorithm (`compiler.cxx`)

1. **Validate declarations.** Check for duplicate pass names; stale versions; a read of an unwritten transient; writes to an import declared read-only; one resource used with two conflicting layouts in one pass; attachments in a non-raster pass; a raster pass with compute affinity; `exit.queue != graphics`; and token use on a compute-affinity pass. Each failure returns `FrameGraphError{type, pass name, resource name}`.
2. **Cull.** Roots are passes with `side_effect()`, passes that write an imported resource's final version, and passes that touch the swapchain. Walk backwards along producer edges (version reads) and mark live passes. Culled passes record nothing, get no timestamp slot, and their accesses are ignored.
3. **Resolve queues.** On a single-queue topology, or with `async_compute == false`, every pass goes to graphics. Otherwise `compute_required` goes to compute. `compute_preferred` goes to compute unless it is *non-overlappable*: every live graphics pass is either an ancestor or a descendant of it in the dependency DAG (reachability via bitsets, at most 64 passes). That case would cost two semaphores and gain nothing, so the pass is demoted.
4. **Add virtual prologue and epilogue.** Pass −1 (graphics) owns every import's `entry` state. Pass N (graphics) applies every import's `exit` state.
5. **Track state per resource.** Walk live passes in declaration order and keep this per resource:
   ```
   owner_queue, layout,
   last_write {pass, queue, stages, access},
   reads_since_write[queue] {stages, access, last_pass},
   visible_to {stages, access}   // already made visible to these on owner queue
   ```
   For each access A by pass P on queue Q:
   - **Queue switch, `sharing == exclusive`, contents needed** (A does not discard): record an ownership transfer. The release goes on `owner_queue` immediately after the last pass on that queue that accessed the resource. Its src scope is that queue's last write plus reads since; dst is NONE. The acquire goes on Q before P, with src NONE and dst equal to A's stages and access. Both halves carry the same `old_layout`/`new_layout`, so the transition happens exactly once. Families are taken from the topology (IGNORED when they are the same family). Add a cross-queue edge from that last pass to P.
   - **Queue switch where A discards, or `sharing == concurrent`**: no transfer. Add a cross-queue edge from every last access on the other queue (WAR/WAW/RAW as applicable) to P. On Q, emit a barrier only for the layout transition, from UNDEFINED when A discards. For concurrent sharing, reads on different queues add no edges between each other.
   - **Same queue, write**: barrier with src = last write (memory) plus reads since (execution only, access NONE); dst = A. Clear `reads_since_write` and `visible_to`.
   - **Same queue, read**: if the layout changes, or A's stages/access are not already within `visible_to`, emit a barrier from last write to A and widen `visible_to`. Otherwise emit nothing. This is how a second sampled read in the same layout costs no barrier.
   - **Tokens** emit `MemoryBarrier` entries only.
   - **Legacy passes** get one `MemoryBarrier{ALL_COMMANDS, MEMORY_READ|WRITE → ALL_COMMANDS, MEMORY_READ|WRITE}` before and after, plus layout transitions to the layouts declared for them.
   All barriers needed before a pass go into its `BarrierSet` (one `vkCmdPipelineBarrier2`).
6. **Import exit (epilogue).** For each import, transition from its tracked state to `exit`. If it ended on compute, emit a release in the compute batch holding its last access and an acquire in the graphics epilogue batch. That batch then waits on the compute batch; this wait usually exists already.
7. **Split batches.** For each queue, walk its passes in order. Start a new batch before any pass with an incoming cross-queue edge (its wait goes at the batch start; acquires go in `Batch::acquires`). End the current batch after any pass that is the source of an outgoing cross-queue edge (signal at batch end; releases go in `Batch::releases`). The graphics prologue is always batch 0. The epilogue barriers go in the last graphics batch.
8. **Order submissions** by each batch's first pass index. Because cross edges always run from an earlier pass to a later one, every wait then refers to an already-submitted batch, so there is no wait-before-signal. A debug assertion checks this.
9. **Resolve waits.** Each batch keeps at most one wait per other queue: the maximum `signal_index`, with `stages` = the union of the acquire dst stages and the first-use stages of the edges it covers. The swapchain acquire wait goes on the first batch that touches the swapchain import, with that access's stages. `render_finished` is signalled by the last graphics batch.
10. **Assign timestamp slots** per queue in order for live passes.
11. **Derive usage**: OR together each transient's image/buffer usage from its Uses.
12. **Plan transients** (Phase 5, `aliasing.cxx`): see Phase 5.
13. **Hash** (FNV-1a over pass names, types, affinities, accesses, transient descs, import descriptors excluding physical handles, topology and options). `FrameGraphCompiler` caches the last `CompiledGraph`; on an equal hash it reuses the plan and only rebinds record lambdas and physical handles. The transient allocator also uses the hash to decide whether its images can stay.

### Recording backend (`executor.cxx`)

`record(CompiledGraph const &, GraphDesc &, FrameResources const &, std::span<VkCommandBuffer> per_batch)`. For each batch:

1. `acquires` (one `vkCmdPipelineBarrier2`).
2. For each pass: the `before` barriers; the CPU Tracy zone and GPU Tracy zone (interned source location, see `pass_profiler`); the begin timestamp (`TOP_OF_PIPE`, after the barriers); `vkCmdBeginRendering` from `RenderingDesc` for raster passes; the record lambda; `vkCmdEndRendering`; the end timestamp (`BOTTOM_OF_PIPE`).
3. `releases`.

`vk_translate.cxx` turns `ImageBarrier`+`PhysicalImage` into `VkImageMemoryBarrier2` (aspect from format, all mips and layers) and `BufferBarrier` into `VkBufferMemoryBarrier2` (offset 0, `VK_WHOLE_SIZE`). It is a pure function, so it is unit-testable without a device.

**Tracy names come from pass names.** `PassProfiler` keeps a `std::deque<tracy::SourceLocationData>` that is never freed, interned per `(label, color)`. It opens zones with the source-location-pointer constructors of `tracy::ScopedZone` (CPU) and `tracy::VkCtxScope` (GPU, using the pass's queue context). Zones therefore stay named and coloured even though names are runtime strings. This is the equivalent of `ZoneScopedNC`/`TracyVkZoneC`; no bare or transient zones are used.

---

## Phase 0: Baseline harness (do this first, small)

**Files:** `src/app/benchmark.cxx`, `include/app/benchmark.hxx`, `tools/perf/compare_screenshots.py` (new; run with `uv run`), `docs/perf-benchmark.md`

1. Add the occlusion statistics (frustum-visible, phase 1 drawn, candidates, phase 2 drawn, meshlets deferred and occluded) and the cluster statistics totals to the benchmark JSON as per-frame means and the final-keyframe values. They are already folded into `FrameStats` by `consume_culled_readback`.
2. Add `compare_screenshots.py --base dir --head dir`. It requires bit-identical PNGs in keyframe order (Pillow through a uv inline script header) and reports the first differing pixel.
3. Capture the baselines with `--benchmark-screenshots`: occlusion off; occlusion on; on with `--meshlet-occlusion=on`; and the `always_defer` stub. Store them outside the repo (for example `perf/baseline/`, gitignored).

**Acceptance check:** two consecutive runs of the same build compare identical (`compare_screenshots.py` exits 0), and the occlusion statistics match exactly for a parked camera.

## Phase 1: Core compiler (queue-aware, no Vulkan device)

**Files:** `include/rendering/frame_graph/{types,use_table,frame_graph,compiled_graph,compiler,scheduler,frame_graph_error}.hxx`, `src/rendering/frame_graph/{frame_graph,compiler,scheduler}.cxx`, `src/rendering/CMakeLists.txt` (add `frame_graph/frame_graph.cxx frame_graph/compiler.cxx frame_graph/scheduler.cxx`), `include/core/error_types.def` (`X(FrameGraphError)`), `include/core/renderer_error.hxx` (`RendererErrorType::frame_graph_error` plus a formatter case), `src/rendering/rendering_error_describe.cxx`, and the tests `test/frame_graph_test_support.hxx`, `test/frame_graph_compiler_test.cxx`, `test/frame_graph_queue_test.cxx`, `test/frame_graph_isolation_test.cxx`.

1. Implement the types, the list scheduler (`scheduler.cxx`: dependency graph in, ordered passes out, declaration order by default), the `FrameGraph`/`PassBuilder` front end (vectors reset each frame; record lambdas stored but never called by the compiler) and `compile()` steps 1–11 and 13. Step 12 waits for Phase 5.
2. Add `FrameGraphError { FrameGraphErrorType type; std::string pass; std::string resource; }` with `std::formatter<FrameGraphErrorType>`, following the `RendererError` and `PipelineGraphError` pattern.
3. Write the test support: a **happens-before checker**. Given a `CompiledGraph`, it builds a graph whose nodes are (batch, pass) with these edges:
   - same-queue program order, counted only where a barrier's scopes cover the hazard pair;
   - semaphore edges from signal batch to wait batch;
   - in-order queue semantics, so a signal covers everything earlier on its queue.
   For every hazard pair (RAW/WAR/WAW on the same resource, using the declarations) it asserts that the pair is ordered and covered by a barrier or semaphore whose scopes include both accesses. For exclusive resources whose contents are needed it also asserts that every queue-family change has a matching release/acquire with identical layouts. A random-DAG generator (fixed seeds, 2–24 passes, 1–12 resources, random affinities and uses) feeds the property tests.

**Tests (all in `TEST_SUITE("unit")`):**

- *Barriers:* RAW color→sampled gives exactly one image barrier `COLOR_ATTACHMENT_OUTPUT/COLOR_ATTACHMENT_WRITE → FRAGMENT/SAMPLED_READ`, `COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL`. A read after read in the same layout and stages gives no barrier; a later read from a new stage gives one barrier that widens only the dst. WAR sampled→storage_write gives an execution-only barrier (src access NONE) plus the layout change. Buffer WAW gives a write→write barrier. Every buffer barrier is whole-buffer. Several barriers before one pass end up in one `BarrierSet`.
- *Imports:* the entry state becomes the src of the first barrier. The swapchain goes UNDEFINED → COLOR_ATTACHMENT, then → PRESENT_SRC in the epilogue. The shadow atlas with entry SHADER_READ_ONLY and LOAD keeps its contents (old layout is not UNDEFINED). Hi-Z with entry UNDEFINED (first frame) still transitions correctly.
- *Culling:* a transient written and never read is culled, transitively. `side_effect` keeps a pass. A write to an import's final version keeps a pass. A culled pass produces no barriers and no timestamp slot. A graph where AO is disabled culls the AO chain.
- *Validation errors:* each `FrameGraphErrorType` is triggered once, including a stale version, a transient read before its write, a raster pass with `compute_required`, and an import with `exit.queue == compute`.
- *Queue topologies:* the same random graph compiled under single-queue, same-family and dedicated topologies. Single-queue has 1 batch (2 if a prologue split is needed), zero waits and zero transfers, and its `BarrierSet`s are identical to compiling with every affinity set to graphics on the dedicated topology. That is the "one code path" property. Same-family has waits but every family field is IGNORED. Dedicated has transfers with matching layouts and families.
- *Async shape:* graphics A writes depth → compute B samples it → graphics C writes depth gives batches [G0: A + release], [C0: acquire + B], [G1: …]. C0 waits on G0's index. When C discards (CLEAR), C0 → G1 is a WAR semaphore with **no** transfer and an UNDEFINED old layout.
- *Prologue release:* an import whose first use is on compute gets its release in graphics batch 0, and the compute batch waits on batch 0.
- *Epilogue acquire:* an import whose last use is on compute gets a release at the end of the compute batch and an acquire in the last graphics batch, which waits on that compute batch.
- *Concurrent sharing:* two compute readers and one graphics reader of a concurrent buffer produce no transfers, and the readers are not ordered with each other.
- *Exclusive ping-pong:* a read-only exclusive buffer read alternately G, C, G, C produces three transfers. This documents the cost that motivates Phase 8.
- *Wait merging:* two transfers into the same batch from one queue give a single wait with the max index and the union of stages.
- *Submission order:* in every random graph, every wait refers to an earlier batch (no wait-before-signal), and the happens-before checker passes.
- *Demotion:* a `compute_preferred` pass with no independent graphics pass is demoted; adding an independent graphics pass (shadows) keeps it on compute. `compute_required` is never demoted. `async_compute=false` demotes everything.
- *Swapchain wiring:* exactly one batch has `waits_swapchain_acquire`, and it is the first to touch the swapchain. Exactly one batch has `signals_render_finished`, and it is the last graphics batch.
- *Two-frame Hi-Z:* compile the real occlusion-chain shape (main_cs G, early prepass G, hiz_build C, late_cs C, late prepass G, forward G) twice, then join the two plans in the checker with in-order per-queue semantics. Hi-Z leaves frame N owned by graphics. Frame N+1's `main_cs` read is ordered after frame N's build, and frame N+1's build write is ordered after frame N's last read.
- *Timestamps:* slots per queue are contiguous in order; culled passes have none; names map back to slots.
- *Hash:* the same declarations give the same hash, different lambdas still give the same hash, and changing one Use changes it.
- *Legacy shim:* a legacy pass gets global barriers on both sides and the declared entry layouts.
- *Scheduler:* with `SchedulerMode::declaration_order` the schedule equals declaration order. With reordering on, every schedule is a topological order of the dependency graph (property test over the random graphs, checked by the happens-before checker). A compute producer followed by its graphics consumer, with an independent graphics pass declared after the consumer, gets that pass hoisted into the gap. `pinned` and `legacy()` passes never move or get crossed. Ties break by declaration index, so the result is deterministic. The reordered plan never has more cross-queue waits than the declaration-order plan.
- *Isolation:* `frame_graph_isolation_test.cxx` includes only `rendering/frame_graph/compiler.hxx` and compiles a graph. A code-review rule backs this up: no `gpu/` includes under `src/rendering/frame_graph/{frame_graph,compiler,aliasing}.cxx`.

**Acceptance check:** `cargo xtask test` is green, including about 30 deterministic cases and the property tests over at least 500 seeds. `cargo xtask tidy` is clean. The engine binary is unchanged in behaviour (nothing calls the graph yet).

## Phase 2: Queue and sync infrastructure (exercised on one queue)

**Files:** `include/gpu/context.hxx`, `src/vulkan_bootstrap.cxx`, `include/gpu/queue_set.hxx` + `src/gpu/queue_set.cxx` (new), `src/gpu/CMakeLists.txt`, `include/gpu/swapchain.hxx`, `src/gpu/swapchain.cxx`, `include/gpu/host_query_context.hxx`, `src/gpu/host_query_context.cxx`, `src/gpu/context.cxx`, `src/main.cxx`, `include/rendering/renderer.hxx` / `renderer.cxx` (Tracy collect, `wait_idle`)

1. **Discovery.** Add `compute`, `compute_queue_index` and `QueueTopologyKind {single, same_family, dedicated}` to `QueueFamilies`. In `find_queue_families`, keep the current graphics/present choice. Then pick the compute family as follows:
   - a family with COMPUTE and without GRAPHICS, preferring `timestampValidBits > 0`; otherwise
   - the graphics family if its `queueCount >= 2` (queue index 1); otherwise
   - none (single).
   Log the result ("Queues: graphics family 0, compute family 2 (dedicated)").
2. **Device.** In device selection, require `vulkan12_features.timelineSemaphore`; enable it in `create_device`. Build the queue create infos from the unique families. The same-family case needs `queueCount = 2`, with priorities {1.0, 1.0}. Fetch `context.compute_queue`. On a single topology, `compute_queue == graphics_queue`.
3. **CLI.** Add `--async-compute=auto|off|same-family`. `off` forces the single topology in the compiler while still creating the queue, so the fallback can be tested on any hardware. `same-family` forces topology 2 when the graphics family has two queues. Add `--sync-validation`, which sets `validate_sync=true` (plus `syncval_submit_time_validation`) through `VkLayerSettingsCreateInfoEXT` on the instance in Debug builds.
4. **`QueueSet`** (`include/gpu/queue_set.hxx`):
   ```cpp
   struct GpuQueue { VkQueue queue; std::uint32_t family; VkSemaphore timeline; std::uint64_t value = 0; };
   struct FrameSlot {
       std::array<VkCommandPool, logical_queue_count> pools{};
       std::array<std::vector<VkCommandBuffer>, logical_queue_count> buffers{}; // grown on demand
       std::array<std::uint32_t, logical_queue_count> used{};
       std::array<std::uint64_t, logical_queue_count> last_value{};
   };
   class QueueSet {
   public:
       auto initialize(VulkanContext &) -> std::expected<void, DeviceError>;
       auto begin_slot(std::uint32_t slot) -> std::expected<void, QueueSetError>; // vkWaitSemaphores(2 s) + reset pools
       auto command_buffer(LogicalQueue) -> std::expected<VkCommandBuffer, QueueSetError>; // begun, ONE_TIME_SUBMIT
       auto submit(std::span<SubmitBatch const>, VkSemaphore acquire, VkSemaphore render_finished)
               -> std::expected<void, QueueSetError>;          // one vkQueueSubmit2 per batch, in order
       auto topology() const -> frame_graph::QueueTopology;
   };
   ```
   When both logical queues are the same physical queue, `QueueSet` aliases them: the compute pool and timeline *are* the graphics ones.
5. **Swapchain split.** Remove the `in_flight` fence, the command pool and the command buffers from `Swapchain`. It keeps `image_available[slot]` and `render_finished[image]` and gains `acquire(slot)` / `present(image_index, slot)`. `begin_frame()` becomes `QueueSet::begin_slot(slot)` (timeline waits replace the fence; a timeout still maps to `device_lost`), then `Swapchain::acquire`, then the prologue command buffer = `queue_set.command_buffer(graphics)`. `end_frame()` becomes end command buffer, `QueueSet::submit({one graphics batch})`, then `Swapchain::present`. `main.cxx` `draw()` follows this order. The rule "always retire the frame we began" stays.
6. **Per-queue Tracy.** `HostQueryContext` becomes per queue: `initialize(vulkan_context, VkQueue, family, VkCommandBuffer one_time)`. Host-calibrated contexts need no queue. For the fallback contexts, allocate a one-time command buffer from a pool on the compute family. Name them "graphics"/"compute". Collect both in `record_frame`.
7. **Smoke path.** Behind `--async-compute-smoke`, submit an empty compute batch each frame that waits on graphics batch 0's value and signals. The graphics batch, now split after the prologue, waits on it. This exercises timeline values, multi-batch submission and slot waits with no data dependency.

**Acceptance check:**
- The Phase 0 baselines (all four variants) are bit-identical and the occlusion statistics are equal.
- A Debug run with `--sync-validation` for 300 frames, including a resize and the occlusion toggles, has no new validation messages.
- With `--async-compute-smoke` on a dedicated-family GPU, Tracy shows both GPU contexts and validation is clean.
- With `--async-compute=off`, behaviour is identical.
- The startup log shows the topology on at least one AMD or NVIDIA GPU (dedicated) and one GPU or iGPU without a dedicated compute family (or lavapipe/llvmpipe for single).

## Phase 3: Recording backend

**Files:** `include/rendering/frame_graph/{executor,pass_profiler}.hxx`, `src/rendering/frame_graph/{executor,pass_profiler,vk_translate}.cxx`, `src/rendering/CMakeLists.txt`, `test/frame_graph_vk_translate_test.cxx`, `include/rendering/renderer.hxx`, `src/rendering/renderer.cxx` (`record_frame` wraps the old body), `include/rendering/renderer.hxx` (`FrameRecordInfo` gets a `QueueSet &`)

1. **`vk_translate.cxx`:** convert `ImageBarrier`/`BufferBarrier`/`MemoryBarrier` into `VkDependencyInfo` storage, and `RenderingDesc` into `VkRenderingInfo` and `VkRenderingAttachmentInfo` (load/store ops, clear values, `resolveMode`, resolve view and layout). Pure functions, unit-tested: release/acquire family fields, IGNORED for the same family, aspect from depth formats, whole-buffer size.
2. **`PassProfiler`:** one timestamp pool per (queue, slot), sized `2 * max_passes + overlay sub-scopes`. Reset on the host with `vkResetQueryPool` after `begin_slot` (`hostQueryReset` is already enabled). Read back on slot reuse into `std::vector<PassTiming{std::string name_id; std::string label; LogicalQueue queue; float ms;}>`. If the compute family has `timestampValidBits == 0`, compute passes record no timestamps and report `std::nullopt`. Also provides the interned Tracy source locations (see the design overview).
3. **`Executor::record`:** as described in the design overview. Batch 0 appends to the prologue command buffer; later batches get fresh command buffers from `QueueSet`. Build the `SubmitBatch` list: per batch, the waits translated to absolute values, the swapchain acquire wait on the flagged batch, `render_finished` on the flagged batch.
4. **Wrap the whole current frame.** `record_frame` builds a graph with one `legacy()` graphics pass ("frame_legacy") around the old body, plus imports for the swapchain (entry UNDEFINED, exit PRESENT, declared as the legacy pass's layouts). Because the legacy body still does its own present transition, its declared exit layout is PRESENT and the epilogue emits nothing. Compile, execute, submit.

**Acceptance check:** baselines are bit-identical; sync validation is clean; the new "Frame graph" timings list shows `frame_legacy` on graphics; Tracy shows a named, coloured "Frame (legacy)" GPU zone; `RenderStage` timings are unchanged (both systems run side by side until Phase 7); `vk_translate` tests pass.

## Phase 4: Migrate passes back to front (all on graphics)

**Files:** `src/rendering/renderer_frame_graph.cxx` (new, added to `src/rendering/CMakeLists.txt`; holds `Renderer::build_frame_graph` and one `add_*_pass` member per pass, so `renderer.cxx` shrinks rather than grows), `include/rendering/renderer.hxx`, `src/rendering/renderer.cxx`, `include/rendering/render_passes.hxx`, `src/rendering/render_passes.cxx`, `include/rendering/screenshot.hxx`, `src/rendering/screenshot.cxx`, `src/rendering/environment.cxx`/`.hxx` (resource exposure only), `src/rendering/imgui_renderer.cxx` (texture-id list), `include/rendering/overlay.hxx` (contract comment)

**The shim.** Unmigrated work stays inside legacy passes, so the frame becomes `[legacy: everything not yet migrated] → migrated passes`. A migrated pass deletes its hand barriers and transitions and declares its uses. A legacy pass declares, for every graph-visible resource it touches, the layout it expects on entry and the layout it leaves (exactly what its internal `transition_image_layout` calls assume today), so the graph hands it the right layout. Each step below is a separate commit with the Phase 0 acceptance check. `render_pass::Context` gains no new fields. Its `timestamp_query_pool` and the `stage*2` writes are deleted from each migrated `render_pass::*` body, because the executor now writes timestamps. Until Phase 7, `RenderStage` slots of migrated passes are filled by a temporary `write_empty_stage` in the shim, so the old timings panel stays valid.

Imports declared once per frame in `build_frame_graph`:

- **Swapchain:** entry UNDEFINED, exit PRESENT.
- **Viewport target:** entry SHADER_READ_ONLY, exit SHADER_READ_ONLY.
- **Shadow atlas:** entry SHADER_READ_ONLY if `shadow_atlas_initialized_`, else UNDEFINED; exit SHADER_READ_ONLY.
- **Hi-Z:** entry SHADER_READ_ONLY if `hiz_.layout_initialised`, else UNDEFINED; exit SHADER_READ_ONLY.
- **Environment images:** entry and exit SHADER_READ_ONLY.
- **Forward, AO and bloom targets:** imported in Phase 4 and made transient in Phase 5.
- **Per-frame buffers:** entry = the prologue's last write. For example `draw_buffer` gets `{ALL_TRANSFER/COPY, TRANSFER_WRITE}` from `upload_frame_data`, and host-written buffers get `{HOST, HOST_WRITE}`. Exit = graphics.
- **Material, texture and geometry storage:** not imported. Contract: graphics-only, always SHADER_READ_ONLY, never accessed by a compute-affinity pass (validated by keeping them out of `PassContext`).
- **`overlay_data`:** a token. `ui_sampled` comes from the ImGui draw data (see step 8).

Migration order:

1. **Composite, `ui_only` and screenshot.** `composite` is a raster pass: `color(swapchain or viewport, DONT_CARE, STORE)` plus `read(resolved_hdr, sampled, fragment)` and `read(bloom, sampled, fragment)`. `ui` is a raster pass on the swapchain with CLEAR; in viewport mode it reads the viewport target as sampled. `screenshot` is a transfer pass with `side_effect()` that reads swapchain or viewport as `transfer_src`. `ScreenshotImage` gets a `managed_by_graph` mode in which `screenshot.cxx` skips its own before/after transitions. Delete `detail::transition_swapchain_to_attachment` / `_to_present`, `present_swapchain` and `transition_to_shader_read`.
2. **Bloom.** Compute: `read(resolved_hdr, sampled, compute)`; `write(bloom, storage_write, compute, ExitUse{sampled})`. The per-mip transitions stay internal; delete the initial UNDEFINED→GENERAL transition (the graph provides it). Only add the pass when `bloom_settings_.enabled`. Composite then reads the emissive fallback, and culling removes nothing else.
3. **Forward.** Raster: `color(hdr, CLEAR, STORE, resolve=resolved_hdr)`; `write_depth(depth, LOAD, STORE)` (EQUAL test, depth writes on); reads of the shadow atlas (sampled, fragment), the AO output (sampled, fragment) and `cluster_lights_buffer`/`lights`/UBO/`visible_*`/`merged_indirect` or `culled_indirect` (indirect plus `shader_read` task/mesh/vertex/fragment); `meshlet_visibility_buffer` `shader_read` task when meshlet occlusion is active. The scene overlays run inside the record lambda and read the `overlay_data` token. Delete `record_meshlet_visibility_barrier` for the forward case and `transition_hdr_to_shader_read`.
4. **AO.** Compute: `read(resolved_depth, sampled, compute)`, `write(ao_raw, storage_write)`, then (in the same pass, since the chain is internal) `write(ao_denoised, storage_write, ExitUse{sampled})`. Alternatively split it into `gtao` and `gtao_denoise`, which is preferred: two passes, so the raw→denoised barrier is derived. Delete the five `transition_image_layout` calls at `render_passes.cxx:929–1000`. With AO disabled the pass is not added, and forward reads `image_storage_.white()`.
5. **Depth prepass, Hi-Z and the occlusion chain.** `depth_prepass_early` / `_only`: `write_depth(depth, CLEAR, STORE, resolve={resolved_depth, MIN or SAMPLE_ZERO})`; reads of `culled_indirect`, visible draws/transforms, the occlusion views and the meshlet bitset (`shader_read_write` task). `hiz_build`: `read(resolved_depth, sampled, compute)`, `write(hiz, storage_write, compute, ExitUse{sampled})`; its per-level transitions stay internal; delete the source-depth round trip and the MSAA-depth self-barrier (`render_passes.cxx:845–895`). `late_cs`: compute, reads candidates/counts/`culled_indirect`/`occlusion_views`/hiz (sampled compute), writes visible draws/transforms (`shader_read_write`), `late_indirect`, `merged_indirect` and stats. Delete both barrier blocks in `record_occlusion_cull_pass`; the execution-only "after early prepass" barrier is now derived from the WAR on the visible buffers. `depth_prepass_late`: `write_depth(depth, LOAD, …, resolve={resolved_depth, SAMPLE_ZERO})`, which discards `resolved_depth`. `occlusion_stats_readback`: transfer, `side_effect`, `stats → transfer_read`, `readback → transfer_write`; the host-read barrier is emitted through the import exit `{HOST, HOST_READ}`. When occlusion is inactive these passes are not added: no `write_empty_stage`. The `hiz_history_*` bookkeeping stays in `record_frame` and keys off whether `hiz_build` was added.
6. **Shadows.** Raster: `write_depth(shadow_atlas, LOAD if preserve_contents else DONT_CARE, STORE)`; the per-cascade `vkCmdClearAttachments` stays in the body. Only added when `update_mask != 0`; forward still reads the atlas through the import entry state. Delete `transition_shadow_atlas_to_attachment` / `_to_shader_read`. The CPU cache bookkeeping in `record_shadow_pass` runs only when the pass was added.
7. **Culling and light clustering** (moved out of `prepare_frame`). `prepare_frame` keeps the CPU work and prologue uploads, and records **no** dispatches. Add `occlusion_stats_clear` and `meshlet_visibility_clear` (transfer, `fill → transfer_write`), `gpu_culling` (`main_cs`, compute) and `light_cull` + `light_cluster` (compute; `cluster_stats_clear` as a transfer pass; `cluster_stats_readback` transfer `side_effect`). Delete the barrier blocks at `renderer.cxx:2979, 3287, 3442, 3533, 3589, 3627`. Timestamp reset and readback move to `PassProfiler`. `cluster_lights_buffer` loses its sub-range barriers (whole-buffer, union of consumers).
8. **Environment builds.** One compute pass, `environment`, added only when `environment_` has work pending (add `Environment::has_pending_record()`). It declares writes to the radiance, prefilter and BRDF LUT images and the SH buffer, with `ExitUse{sampled}`. The 19 internal barriers stay; only its entry and exit transitions are removed. Forward declares the reads.
9. **Overlays and UI.** `overlay_prepare`: a graphics pass, `side_effect`, writes the `overlay_data` token (stages ALL_TRANSFER|COMPUTE, access TRANSFER_WRITE|SHADER_STORAGE_WRITE). It replaces `record_overlay_prepares`' global barrier, and the per-overlay timestamps become `PassContext::profile_scope(entry.desc.name)`. The UI pass reads the token. **ImGui contract:** before the UI pass is declared, `ImGuiRenderer` exposes the bindless texture indices referenced by this frame's draw data (it already computes them at `imgui_renderer.cxx:423`). For each index that maps to a graph resource (`FrameGraph::find_by_bindless(index)`), the UI pass declares `read(id, sampled, fragment)`. Graph transients are never given ImGui ids (asserted). This covers the viewport, the Hi-Z debug view and the environment panel automatically.

**Acceptance check (after every step):** bit-identical screenshots for all four Phase 0 variants; identical occlusion and cluster statistics; Debug `--sync-validation` for 300 frames with resize, culling/meshlet/AO/bloom toggles and the debug views open: no errors. Also run `--frame-graph-serialize` (the `CompileOptions::serialize` debug mode) and get identical screenshots. RenderDoc event order matches the `docs/occlusion-culling.md` "Verifying" list. At the end of Phase 4: no `legacy()` passes remain, `grep vkCmdPipelineBarrier2 src/rendering/render_passes.cxx src/rendering/renderer.cxx` finds only intra-pass sites (bloom/Hi-Z mip chains), and `upload_frame_data`'s prologue barrier remains.

## Phase 5: Transient allocation and aliasing (simplifies resize)

**Files:** `include/rendering/frame_graph/aliasing.hxx`, `src/rendering/frame_graph/aliasing.cxx`, `include/rendering/frame_graph/transient_allocator.hxx`, `src/rendering/frame_graph/transient_allocator.cxx`, `include/gpu/image.hxx`, `src/gpu/image.cxx` (`Image::create_aliased`), `include/gpu/image_storage.hxx` / `src/gpu/image_storage.cxx` (`create_aliased_image`, registers bindless slots), `include/rendering/renderer.hxx`, `src/rendering/renderer.cxx` (`create_frame_targets`, `resize`, `OwnedFrameTargets`), `src/rendering/renderer_frame_graph.cxx`, `include/rendering/forward_target.hxx` / `src/rendering/forward_target.cxx` (deleted or reduced to a desc helper), `test/frame_graph_aliasing_test.cxx`

1. **Pure planner** `plan_transients(CompiledGraph const &, std::span<MemoryRequirement const>, bool alias) -> TransientPlan`. `MemoryRequirement{size, alignment, memory_type_bits}` comes from the caller, so tests need no Vulkan. Lifetime = the set of live passes using the resource. Two transients **conflict** unless every access of one happens-before every access of the other in the compiled happens-before graph (same-queue order, or semaphore chains). Conflict is *not* decided by index order; that is what makes it overlap-aware. Greedy placement: sort by size descending; first-fit offset in blocks grouped by compatible `memory_type_bits`; images only (buffers stay persistent; `bufferImageGranularity` is avoided by never mixing them). For each aliased resource whose block was used before, its first-use barrier gets `old_layout = UNDEFINED` and src scope = the previous occupant's last access (for a cross-queue predecessor the semaphore already covers it, and no transfer is needed because contents are discarded). `alias=false` gives every transient its own block.
2. **`TransientAllocator`:** per frame slot, keyed by `CompiledGraph::hash` plus the transient descs. On a hash change: retire the old blocks and images to the slot's deletion list (freed at the next `begin_slot` of that slot), allocate blocks with `vmaAllocateMemory` (requirements via `vkGetDeviceImageMemoryRequirements`; `maintenance4` is already enabled), and create images through `vmaCreateAliasingImage2` with the derived usage. Register bindless slots, plus per-mip slots for bloom. Images and bindless indices are stable while the hash is unchanged, so passes keep passing texture indices in push constants.
3. **Convert to transients:** forward hdr/depth (MSAA), resolved hdr/depth, `ao_raw`, `ao_denoised`, the bloom chain. The viewport target and Hi-Z stay imported (ImGui panel / history).
4. **Simplify `resize()`:** it only recreates the viewport targets and the Hi-Z pyramid and clears `hiz_history_valid_`. Transient descs follow `extent_`, so the next compile's hash change triggers reallocation. `wait_idle` stays for the imports. `OwnedFrameTargets` shrinks to the viewport target.
5. **Debug flag:** `--frame-graph-alias=off` and a Lighting/Debug UI checkbox map to `CompileOptions::alias_transients`. A memory readout shows aliased versus unaliased transient bytes.

**Tests:** disjoint same-queue lifetimes share an offset; overlapping ones don't; disjoint by index but concurrent across queues (no semaphore path) don't alias; ordered through a semaphore do alias; `alias=false` means no shared offsets; the aliasing barrier has UNDEFINED and the predecessor's src scope; incompatible `memory_type_bits` go to separate blocks; the property test (random graphs) checks no two conflicting transients overlap in memory.

**Acceptance check:** baselines are bit-identical with aliasing on and off; sync validation is clean across 20 resizes (drag-resize) and AO/bloom toggles; Tracy memory or the VMA stats show the transient total dropping with aliasing on (expect AO raw/denoised to share with the bloom chain); no leaks after 1000 resize cycles (`memory_tracker` / VMA budget stable).

## Phase 6: Enable async compute, one pass at a time, by measurement

**Files:** `src/rendering/renderer_frame_graph.cxx` (affinity and declaration-order changes), `include/rendering/renderer.hxx` (a `async_compute_` toggle mirrored in the UI), `src/app/application.cxx` (Debug panel: async toggle, per-pass queue column in timings), `include/rendering/hiz_occlusion.hxx` comment + `renderer.hxx` `HizPyramid` comment, `docs/occlusion-culling.md`, `docs/clustered-lighting.md`

Each candidate is a single `b.queue(QueueAffinity::compute_preferred)` line. The list scheduler (see the design overview) then places independent graphics work between the compute producer and its consumer; where it cannot find a legal placement, a declaration-order edit remains the manual fallback. The overlap each candidate relies on:

1. **Light cull and clustering.** Depends only on lights, UBO and frustum planes; overlaps the depth prepass and shadows. This needs transfers of `lights_buffer`, the UBO, `frustum_planes`, `visible_lights` and `cluster_lights` (whole buffer) unless Phase 8 lands first.
2. **Hi-Z build and `late_cs`.** Declare **shadows after the early prepass** so they overlap the compute chain; the late prepass waits on `late_cs`. Requires transfers of `resolved_depth` (graphics→compute; the back transfer is skipped because the late prepass resolve discards it), the Hi-Z (prologue/epilogue rule), and the visible/indirect/stats buffers.
3. **GTAO and denoise.** With shadows declared between the late prepass and forward (if not already moved by candidate 2), GTAO overlaps the shadow rasterisation. The AO images are transients on compute, and forward acquires `ao_denoised`.

For each candidate: capture a Tracy trace (both GPU contexts, check the overlap is real), then run `tools/perf/run_benchmark.sh` on and off (3000 frames, 300 warmup) on at least one dedicated-family GPU. Keep the change only if the `full_frame` median improves by more than the noise floor reported by `compare_benchmarks.py`. Otherwise revert it to graphics and record the numbers in the PR.

**Acceptance check (per candidate):**
- Bit-identical screenshots with async on, off, and `--async-compute=same-family`.
- Identical occlusion and cluster statistics.
- `--sync-validation` is clean for 300 frames with resize and toggles, *including the QFOT checks*.
- `--frame-graph-serialize` gives identical output.
- On a single-family device (or forced `--async-compute=off`) the same build renders identically. This is the fallback test and runs in every PR on whatever hardware the author has.
- The per-pass timings panel shows the pass on the compute queue.

## Phase 7: Delete `RenderStage` and `write_empty_stage`

**Files:** `include/rendering/render_stage.hxx` (deleted), `include/rendering/renderer.hxx` (`StageTimings` → `std::vector<PassTiming>` plus `full_frame_ms`; `FrameTimestamps`, `timestamp_queries_`, `write_empty_stage`, `overlay_query`, `total_query_count` removed), `src/rendering/renderer.cxx`, `include/rendering/render_passes.hxx` (`Context::timestamp_query_pool` removed), `src/rendering/render_passes.cxx`, `include/rendering/overlay.hxx`, `src/app/application.cxx` (timings plot keyed by `name_id`, `std::flat_map<std::string, ScrollingBuffer>`), `include/app/application.hxx`, `src/main.cxx` (timing accumulation loop), `src/app/benchmark.cxx` / `include/app/benchmark.hxx` (`samples_ms_` keyed by `name_id`; a pass absent in a frame contributes 0 ms once seen, matching today's empty-stage semantics; `benchmark_stage_id` deleted, since the ids now come from `PassProfile::name_id`), `tools/perf/compare_benchmarks.py` (accept renamed or new stage ids), `docs/perf-benchmark.md`, `docs/occlusion-culling.md` (replace "Frame order"/"Barriers" with a pointer to the graph and a generated listing), new `docs/frame-graph.md` usage section

1. Remove the temporary shim `write_empty_stage` calls and the stage pool. `full_frame` = first graphics timestamp (prologue begin) to last (epilogue end); compute work runs inside that span.
2. Add a debug dump: `FrameGraph::describe(CompiledGraph const &) -> std::string` (passes, queues, batches, waits, transfers, aliasing) to a Debug panel and a `--frame-graph-dump` CLI flag. It replaces the prose barrier documentation.

**Acceptance check:** `grep -r RenderStage src include` is empty; benchmark JSON stage ids match the old ones for unchanged passes (`gpu_culling`, `depth_prepass`, `hiz_build`, `occlusion_culling`, `depth_prepass_late`, `ambient_occlusion`, `forward_pass`, `composition`, `bloom`, `environment`, `shadow_pass`, `light_clustering`, `full_frame`), with new ids only for split passes, and `compare_benchmarks.py` against a Phase 0 base JSON works; baselines are bit-identical; tidy is clean.

## Phase 8 (later, optional): Concurrent sharing for read-mostly buffers

**Files:** `include/gpu/buffer.hxx`, `src/gpu/buffer.cxx` (a `BufferCreateInfo::sharing` with the {graphics, compute} family list when the topology is dedicated), `src/rendering/renderer.cxx` (UBOs, `lights_buffer`, `frustum_planes_buffer`, `draw/transform/batch_bounds/indirect` as concurrent), `src/rendering/renderer_frame_graph.cxx` (`Sharing::concurrent` on their imports)

The compiler already supports `Sharing::concurrent` from Phase 1. This phase only changes allocation flags and import descriptors, and removes the transfer ping-pong measured in Phase 6. Images stay exclusive: concurrent sharing can disable compression on some vendors.

**Acceptance check:** the number of transfers per frame in `--frame-graph-dump` drops to images only; benchmark on vs off on a dedicated-family GPU; validation clean; screenshots identical.

---

## Risks

- **Sync validation is a required check but cannot see everything.** Every phase's acceptance runs Debug with `--sync-validation` (Phase 2 adds the flag). Buffer-device-address accesses are invisible to sync validation, and this engine dereferences almost every buffer through BDA. So the compiler's own happens-before property tests are the primary guarantee for buffer hazards. `--frame-graph-serialize` A/B runs and the "every BDA buffer a pass dereferences must be declared" review rule are secondary. Consider an optional per-pass debug check that compares the push-constant addresses actually used with the declared buffers' address ranges (cheap, CPU-side, Debug only).
- **Hi-Z history across frames and queues.** Today the shared pyramid relies on one in-order queue (the `HizPyramid` comment). Under async compute, frame N+1's `main_cs` (graphics) reads what frame N's build (compute) wrote, and frame N+1's build writes while frame N's readers might still be in flight. The chosen fix is the **exit-on-graphics rule**: every import must leave the frame owned by the graphics family. A compute-side last use is released at the end of the compute batch and acquired in the graphics epilogue, which waits on that batch. Frame N+1's graphics readers therefore follow it in queue order, and frame N+1's compute writer must wait on a frame N+1 graphics batch (via the prologue release), which comes after all of frame N's graphics work. That is how the "previous-frame timeline wait" is made explicit: the semaphore from the prologue or epilogue batch carries it. The two-frame Hi-Z test locks this down. If a later design wants an import to stay on compute across frames, the persistent import state must carry `{owner queue, last timeline value per queue}`, and the next frame's first batch must wait on the previous frame's absolute value. Do not relax the rule without adding that.
- **Whole-buffer transfers for device-address buffers.** Every transfer is whole-buffer (`VK_WHOLE_SIZE`), and an exclusive buffer read on both queues ping-pongs ownership every switch. The `cluster_lights_buffer` stats/lists split and the sized meshlet-bitset barrier become whole-buffer, which over-synchronises slightly. Phase 8's concurrent sharing removes the ping-pong for read-mostly buffers. Never split buffers into sub-range graph resources without first proving that no shader dereferences across the split.
- **The overlays/ImGui contract.** Overlays run only on graphics. `prepare()` writes are ordered through the `overlay_data` token. Overlays never touch graph transients. ImGui may only sample images that are imports in SHADER_READ_ONLY on the graphics family, or images entirely outside the graph. The UI pass declares reads for every draw-data texture id that maps to a graph resource, so a new debug view (for example of an AO image) must import that image or copy it to an imported one. A transient id reaching ImGui is an assertion.
- **Screenshot and viewport consumers.** The screenshot copy becomes a `side_effect` transfer pass reading the swapchain or viewport as `transfer_src`, and the import exit state restores PRESENT or SHADER_READ_ONLY. The editor Viewport panel samples the *previous* recorded viewport target by bindless index, so the viewport target stays imported and per slot. Do not make it transient.
- **Hardware variance.** Dedicated compute families (AMD, NVIDIA, recent Intel) need transfers. Some devices expose only multiple queues of the graphics family. Others (older iGPUs, llvmpipe) have one queue. A compute family may have `timestampValidBits == 0`, giving no per-pass compute timings. Mitigations: one compiler path with three topologies, all covered by unit tests; `--async-compute=off|same-family` to force the fallbacks on any hardware; every PR from Phase 2 onward runs the baselines with `off`. Present-family ≠ graphics-family is an existing gap that this plan does not fix; `QueueSet` logs a warning when it is detected.
- **Tracy and queries per queue.** The fallback (non-host-calibrated) Tracy context needs a command buffer from a compute-family pool. Timestamps on different queues are not directly comparable without calibration, so `full_frame` uses graphics timestamps only.
- **Submission cost and correctness.** One `vkQueueSubmit2` per batch, in order, means no wait-before-signal. This is about 3–6 submits per frame with async, against 1 today, which is acceptable. The bounded `vkWaitSemaphores` keeps the existing "device lost on 2 s timeout" behaviour.
- **`renderer.cxx` is 5.4k lines.** All graph construction goes into the new `src/rendering/renderer_frame_graph.cxx`. Each `record_*` moves there as it is migrated, and the dead `record_*` and barrier code is deleted in the same commit. `renderer.cxx` should shrink by at least the 13 barrier blocks and the `prepare_frame` culling and clustering regions (about 400 lines).
- **Per-frame allocation churn.** `std::move_only_function` record lambdas and per-frame vectors allocate. Keep `FrameGraph` vectors alive across frames (`reset()` keeps capacity), and if Tracy shows the allocations, move them onto `core/arena_allocator.hxx`.
- **Reordering changes the schedule, not the result.** The scheduler is deterministic (ties break by declaration index) and any legal order is semantically equivalent, but floating-point-sensitive passes or accidental undeclared dependencies can make output differ. Mitigations: `SchedulerMode::declaration_order` for bisecting, pass baselines compared in both modes, and `--frame-graph-dump` making the schedule reviewable in PRs. An undeclared dependency that declaration order hid will surface here, which is a reason to enable reordering only in Phase 6.

## Phase 9: Parallel batch recording

**Files:** `include/rendering/frame_graph/executor.hxx`, `src/rendering/frame_graph/executor.cxx`, `include/gpu/queue_set.hxx` (per-thread command pools), `include/rendering/renderer.hxx`, `src/rendering/renderer.cxx`, `docs/frame-graph.md`

Once every pass is a graph pass, record lambdas only touch their own command buffer, so batches can be recorded concurrently on `Renderer::thread_pool()`.

1. **Per-thread command pools.** `QueueSet` hands out pools keyed by (queue family, worker thread, frame slot); command pools are externally synchronized, so a worker never shares one.
2. **Record tasks.** The executor splits each batch into recording units (a batch, or a run of passes within a long batch, each with its own secondary or primary command buffer) and submits them to the pool. Barriers are computed at compile time, so recording units need no coordination beyond the per-unit buffer ordering.
3. **Pass contract.** A record lambda must not touch shared mutable renderer state. Passes that do (overlays, ImGui, screenshot) declare `pinned_main_thread` and run on the calling thread.
4. **Tracy and profiling.** Zones are per-thread already; the per-queue GPU timestamps are unaffected.
5. **Toggle.** `parallel_recording_` (default on once validated), so single-threaded recording remains the debug reference.

**Acceptance check:** baselines bit-identical with it on and off; sync and thread-safety validation clean (run under `SANITIZE=1` with TSan if the build supports it); CPU record time for the frame drops measurably in Tracy.

## Open questions (resolved)

1. **Per-pass Tracy CPU zones:** on by default (cheap with `TRACY_ON_DEMAND`).
2. **AO granularity:** two passes, `gtao` + `gtao_denoise`.
3. **Compiler reordering:** yes. A deterministic list scheduler may reorder to create queue overlap, subject to the rules in the design overview and enabled in Phase 6.
4. **Hash-skip compile cache:** implement in Phase 3 (the transient allocator needs the hash for stable images regardless).
5. **Memoryless MSAA attachments:** out of scope (tiler-only); revisit after Phase 5 measurements.
6. **Parallel recording of batches across threads:** in scope, as Phase 9.
