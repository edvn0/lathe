#pragma once

#include <volk.h>

#include <glm/mat4x4.hpp>

#include <array>
#include <cstring>
#include <tuple>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "rendering/frame_graph/frame_graph.hxx"
#include "rendering/frame_graph/pass_context.hxx"
#include "rendering/overlay.hxx"
#include "rendering/pipeline_graph_repository.hxx"

struct Renderer;

// The frame graph as a game sees it. The engine calls the game at a few fixed points of the frame (GameSlot) with a
// GameGraph; everything the game declares through it is checked on the CPU:
//   - it never holds a FrameGraph, a PassBuilder or a command buffer;
//   - engine resources are handed out as EngineImage, which only the sampling overloads accept, so a game pass
//     cannot write one, and the graph rejects a forged id with foreign_write anyway;
//   - a record function can only look up indices and addresses of resources its own pass declared;
//   - a slot whose declarations are rejected is rolled back and the engine frame renders without it.
// What a shader does with a bindless index it was given is not something the graph can prove.

enum class GameSlot : std::uint8_t {
    // Before any engine pass that reads or writes scene data. No scene images exist yet.
    frame_start,
    // After the late depth prepass: scene_depth() is final.
    after_depth,
    // After the forward pass: scene_hdr() exists. Too late to add reads to the forward pass.
    after_lighting,
    // After bloom, right before composition. Last chance for replace_scene_colour().
    before_composite,
};

inline constexpr auto game_slot_count = 4U;

// What outlives one frame's GameGraph. Pass names and profiles are stored as views, and a copy of the graph is kept
// by the node editor, so the strings have to live as long as the renderer; messages are logged once.
class GameGraphMemory {
public:
    auto intern(std::string_view text) -> std::string_view { return *names_.emplace(text).first; }

    // True the first time `message` is seen.
    auto report_once(std::string_view message) -> bool { return reported_.emplace(message).second; }

private:
    std::set<std::string, std::less<>> names_;
    std::set<std::string, std::less<>> reported_;
};

struct GameComputeShader {
    PipelineNodeHandle node{};
};

struct GameGraphicsShader {
    PipelineNodeHandle node{};
    // Drawn with source-alpha blending and without depth writes (depth is still tested).
    bool blending = false;
};

class GameGraph;
class GameComputeBuilder;
class GameDrawBuilder;

class EngineImage {
public:
    EngineImage() = default;

private:
    friend class GameGraph;
    friend class GameComputeBuilder;
    explicit EngineImage(frame_graph::ImageId id) : id_{id} {}

    frame_graph::ImageId id_{};
};

class GameImage {
public:
    GameImage() = default;

    [[nodiscard]] auto valid() const noexcept -> bool { return id_.generation != 0; }

private:
    friend class GameGraph;
    friend class GameComputeBuilder;
    explicit GameImage(frame_graph::ImageId id) : id_{id} {}

    frame_graph::ImageId id_{};
};

class GameBuffer {
public:
    GameBuffer() = default;

    [[nodiscard]] auto valid() const noexcept -> bool { return id_.generation != 0; }

private:
    friend class GameGraph;
    friend class GameComputeBuilder;
    friend class GameDrawBuilder;
    explicit GameBuffer(frame_graph::BufferId id) : id_{id} {}

    frame_graph::BufferId id_{};
};

// What a builder returns for a declared access. The record function hands it back to its context to get an index or
// an address, and the context checks the pass really declared it.
class DeclaredImage {
public:
    DeclaredImage() = default;

private:
    friend class GameComputeBuilder;
    friend class GameComputeContext;
    DeclaredImage(std::uint32_t resource, bool writable) : resource_{resource}, writable_{writable} {}

    std::uint32_t resource_ = 0;
    bool writable_ = false;
};

class DeclaredBuffer {
public:
    DeclaredBuffer() = default;

private:
    friend class GameComputeBuilder;
    friend class GameDrawBuilder;
    friend class GameComputeContext;
    friend class GameDrawContext;
    DeclaredBuffer(std::uint32_t resource, bool writable) : resource_{resource}, writable_{writable} {}

    std::uint32_t resource_ = 0;
    bool writable_ = false;
};

struct GameImageDesc {
    VkFormat format = VK_FORMAT_UNDEFINED;
    // Zero in either dimension means the size of the scene targets.
    VkExtent2D extent{};
    std::string_view name;
};

struct GameBufferDesc {
    // In bytes, a positive multiple of four.
    VkDeviceSize size = 0;
    // create_buffer only: fill with zeros before the first pass. Without it the contents are undefined and the graph
    // rejects a read that comes before any write. A persistent buffer is always zero-filled when it is (re)created.
    bool zero = false;
};

struct GamePassProfile {
    std::string_view label;
    std::uint32_t color = 0;
};

// What a GameGraph asks the engine for when a game names a buffer.
struct GameBufferRequest {
    std::string_view name;
    VkDeviceSize size = 0;
    // Shared by all frames in flight and kept between frames, instead of private to this frame.
    bool persistent = false;
};

// The state a persistent buffer is imported with and leaves the frame in. Frames in flight share it, so its first use
// waits for everything the previous frame did with it.
inline constexpr auto persistent_buffer_state = frame_graph::ResourceState{
        .stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
};

// What the engine lends a game graph. Everything is optional so the declaration side can run without a GPU.
struct GameGraphServices {
    Renderer *renderer = nullptr;
    std::uint32_t frame_index = 0;
    VkExtent2D scene_extent{};
    std::array<std::uint32_t, 3> max_group_count{65535U, 65535U, 65535U};

    // Bindless index of the transient image behind a graph resource.
    std::function<std::uint32_t(std::uint32_t resource)> bindless_index;
    // Allocates (or finds) a buffer and imports it into the frame graph, owned by the game. Persistent buffers are
    // imported with persistent_buffer_state.
    std::function<std::optional<frame_graph::BufferId>(GameBufferRequest const &)> acquire_buffer;
};

inline constexpr auto max_game_push_bytes = std::size_t{128};

// Number of workgroups to cover `threads` with groups of `group_size`, or nothing if that is zero or beyond the
// device's limit for the dimension.
[[nodiscard]] constexpr auto dispatch_group_count(std::uint32_t threads, std::uint32_t group_size,
                                                  std::uint32_t max_groups) noexcept -> std::optional<std::uint32_t> {
    if (threads == 0 || group_size == 0) {
        return std::nullopt;
    }
    auto const groups = (threads + group_size - 1U) / group_size;
    return groups <= max_groups ? std::optional{groups} : std::nullopt;
}

// A compute pass described by one struct (GameGraph::add_compute). The struct's resource members declare the pass's
// accesses, and the members listed in its Layout, in that order, are the push constants the shader receives: a
// resource member pushes the address (buffers, 8 bytes) or bindless index (images, 4 bytes) it resolves to, any other
// member pushes itself. Offsets are those of the equivalent C++ struct, so the shader declares the same fields:
//
//   struct SimParams {
//       BufferReadWrite particles;
//       ImageRead depth;
//       std::uint32_t count = 0;
//       float delta_time = 0;
//       using Layout = PushLayout<&SimParams::particles, &SimParams::depth, &SimParams::count, &SimParams::delta_time>;
//   };
template<auto... Members>
struct PushLayout {};

// Read-only storage buffer; pushes its address.
struct BufferRead {
    using push_type = VkDeviceAddress;
    GameBuffer buffer;
    DeclaredBuffer declared;
};

// Written without being read (the old contents are discarded); pushes its address.
struct BufferWrite {
    using push_type = VkDeviceAddress;
    GameBuffer buffer;
    DeclaredBuffer declared;
};

// Read and written; pushes its address.
struct BufferReadWrite {
    using push_type = VkDeviceAddress;
    GameBuffer buffer;
    DeclaredBuffer declared;
};

// Sampled image; pushes its bindless index.
struct ImageRead {
    ImageRead() = default;
    ImageRead(EngineImage image) : source{image} {}
    ImageRead(GameImage image) : source{image} {}

    using push_type = std::uint32_t;
    std::variant<EngineImage, GameImage> source;
    DeclaredImage declared;
};

// Storage image the pass writes; pushes its bindless index. Built from an existing image, or from a description the
// pass creates the image from.
struct ImageWrite {
    ImageWrite() = default;
    ImageWrite(GameImage existing) : image{existing} {}
    ImageWrite(GameImageDesc const &desc) : create{desc} {}

    using push_type = std::uint32_t;
    GameImage image;
    std::optional<GameImageDesc> create;
    DeclaredImage declared;
};

// A few bytes of plain data pushed as they are: a float, a vec3, a uint. Only for DynamicParams. `align` is the
// alignment the shader's field has (4 for a scalar, 8 for a float2, 16 for a float3 or float4 under std430).
struct PushBytes {
    std::array<std::byte, 16> bytes{};
    std::uint8_t size = 0;
    std::uint8_t align = 4;

    template<typename T>
        requires(std::is_trivially_copyable_v<T> && sizeof(T) <= 16 && sizeof(T) % 4 == 0)
    [[nodiscard]] static auto of(T const &value, std::uint8_t align = 4) -> PushBytes {
        auto result = PushBytes{.size = sizeof(T), .align = align};
        std::memcpy(result.bytes.data(), &value, sizeof(T));
        return result;
    }
};

using DynamicParam = std::variant<BufferRead, BufferWrite, BufferReadWrite, ImageRead, ImageWrite, PushBytes>;

// What a parameter struct says at compile time, as a list built at run time (a manifest, see EffectSystem). The same
// rules: resource members declare the pass's accesses, and every member, in order, is pushed with the alignment of
// the equivalent C++ struct (buffers 8, everything else 4).
struct DynamicParams {
    std::vector<DynamicParam> members;
};

template<typename T>
concept ResourceParam = requires { typename std::remove_cvref_t<T>::push_type; };

template<typename P>
concept ParamStruct = requires { typename P::Layout; };

// How a compute pass is scheduled and shown. The queue is only a preference: a pass that touches a persistent buffer
// always runs on the graphics queue.
struct GameComputeOptions {
    frame_graph::QueueAffinity queue = frame_graph::QueueAffinity::compute_preferred;
    // Keeps the pass even when nothing in the graph reads its output.
    bool side_effect = false;
    // Defaults to the pass name.
    std::string_view label;
    std::uint32_t color = 0xFF1493;
};

// The dispatch of a pass: threads in x and y, and the workgroup size the shader was compiled with. The group count is
// rounded up and refused if the device cannot dispatch it.
struct Threads {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t group_x = 64;
    std::uint32_t group_y = 1;

    [[nodiscard]] static constexpr auto of_extent(VkExtent2D extent, std::uint32_t group_x = 8,
                                                  std::uint32_t group_y = 8) noexcept -> Threads {
        return {.x = extent.width, .y = extent.height, .group_x = group_x, .group_y = group_y};
    }
};

class GameComputeContext {
public:
    auto bind(GameComputeShader shader) -> void;

    template<typename Push>
    auto push(Push const &constants) -> void {
        static_assert(std::is_trivially_copyable_v<Push>);
        static_assert(sizeof(Push) <= max_game_push_bytes);
        push_bytes(&constants, sizeof(Push));
    }

    // Dispatches the threads the pass was declared with (GameGraph::add_compute); can be called more than once.
    auto dispatch() -> void;

    [[nodiscard]] auto sampled_index(DeclaredImage image) -> std::uint32_t;
    [[nodiscard]] auto storage_index(DeclaredImage image) -> std::uint32_t;
    [[nodiscard]] auto address(DeclaredBuffer buffer) -> VkDeviceAddress;

    // One thread per element, rounded up to whole groups; refuses a count the device cannot dispatch.
    auto dispatch_threads(std::uint32_t x, std::uint32_t group_x, std::uint32_t y = 1, std::uint32_t group_y = 1)
            -> void;

    // The first thing the pass did that it had not declared. Once set, binds, pushes and dispatches do nothing.
    [[nodiscard]] auto violation() const noexcept -> std::string_view { return violation_; }

private:
    friend class GameGraph;
    GameComputeContext(frame_graph::PassContext &pass, GameGraphServices const &services)
        : pass_{&pass}, services_{&services} {}

    auto fail(std::string_view what) -> void;
    auto push_bytes(void const *data, std::size_t size) -> void;
    // Binds the shader and remembers the threads dispatch() covers.
    auto begin(GameComputeShader shader, Threads threads) -> void;
    // Push constant ranges are whole words; the blob is zero beyond `size`.
    auto push_layout(void const *data, std::size_t size) -> void { push_bytes(data, (size + 3U) & ~std::size_t{3}); }
    [[nodiscard]] auto image_index(DeclaredImage image, bool writable) -> std::uint32_t;

    frame_graph::PassContext *pass_;
    GameGraphServices const *services_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::optional<Threads> threads_;
    // The shader is still compiling or failed to: the pass is skipped for this frame without being an error.
    bool skipped_ = false;
    std::string violation_;
};

class GameComputeBuilder {
public:
    auto queue(frame_graph::QueueAffinity affinity) -> void {
        if (!graphics_only_) {
            pass_->queue(affinity);
        }
    }
    // Keeps the pass even when nothing in the graph reads its output.
    auto side_effect() -> void { pass_->side_effect(); }

    [[nodiscard]] auto sample(EngineImage image) -> DeclaredImage;
    [[nodiscard]] auto sample(GameImage image) -> DeclaredImage;
    [[nodiscard]] auto read(GameBuffer buffer) -> DeclaredBuffer;
    // Advance the handle they are given, so a later pass reads what this one wrote.
    [[nodiscard]] auto write(GameImage &image) -> DeclaredImage;
    [[nodiscard]] auto write(GameBuffer &buffer) -> DeclaredBuffer;
    [[nodiscard]] auto read_write(GameBuffer &buffer) -> DeclaredBuffer;
    [[nodiscard]] auto create_image(GameImageDesc const &desc) -> GameImage;

    // There is deliberately no overload that writes an EngineImage.

private:
    friend class GameGraph;
    GameComputeBuilder(frame_graph::PassBuilder &pass, GameGraph &graph) : pass_{&pass}, graph_{&graph} {}

    // The buffer is shared by frames in flight, so the pass has to stay on the one queue that orders them.
    auto touch(GameBuffer buffer, bool reads) -> void;

    frame_graph::PassBuilder *pass_;
    GameGraph *graph_;
    bool graphics_only_ = false;
};

class GameDrawContext {
public:
    auto bind(GameGraphicsShader shader) -> void;

    template<typename Push>
    auto push(Push const &constants) -> void {
        static_assert(std::is_trivially_copyable_v<Push>);
        static_assert(sizeof(Push) <= max_game_push_bytes);
        push_bytes(&constants, sizeof(Push));
    }

    [[nodiscard]] auto address(DeclaredBuffer buffer) -> VkDeviceAddress;
    [[nodiscard]] auto view_projection() const noexcept -> glm::mat4 const & { return view_projection_; }

    auto draw(std::uint32_t vertex_count, std::uint32_t instance_count = 1) -> void;

    [[nodiscard]] auto violation() const noexcept -> std::string_view { return violation_; }

private:
    friend class GameGraph;
    GameDrawContext(frame_graph::PassContext const &pass, GameGraphServices const &services,
                    OverlayRecordContext const &overlay, std::span<std::uint32_t const> reads)
        : pass_{&pass}, services_{&services}, command_buffer_{overlay.command_buffer},
          view_projection_{overlay.view_projection}, scope_{overlay.scope}, reads_{reads} {}

    auto fail(std::string_view what) -> void;
    auto push_bytes(void const *data, std::size_t size) -> void;

    frame_graph::PassContext const *pass_;
    GameGraphServices const *services_;
    VkCommandBuffer command_buffer_;
    glm::mat4 view_projection_;
    OverlayScope scope_;
    std::span<std::uint32_t const> reads_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    // The shader is still compiling or failed to: the pass is skipped for this frame without being an error.
    bool skipped_ = false;
    std::string violation_;
};

class GameDrawBuilder {
public:
    // The draw runs inside the forward pass, which declares this read on the game's behalf.
    [[nodiscard]] auto read(GameBuffer buffer,
                            frame_graph::ShaderStages stages = frame_graph::stages_of(frame_graph::ShaderStage::vertex))
            -> DeclaredBuffer;

private:
    friend class GameGraph;
    explicit GameDrawBuilder(GameGraph &graph) : graph_{&graph} {}

    GameGraph *graph_;
    std::vector<std::uint32_t> reads_;
};

class GameGraph {
public:
    GameGraph(frame_graph::FrameGraph &graph, GameGraphMemory &memory, GameGraphServices services)
        : graph_{&graph}, memory_{&memory}, services_{std::move(services)} {}

    [[nodiscard]] auto slot() const noexcept -> GameSlot { return slot_; }
    // Which of the frames in flight this is; the index of the buffer copy import() gives.
    [[nodiscard]] auto frame_slot() const noexcept -> std::uint32_t { return services_.frame_index; }
    [[nodiscard]] auto scene_extent() const noexcept -> VkExtent2D { return services_.scene_extent; }

    // Engaged from after_depth / after_lighting on.
    [[nodiscard]] auto scene_depth() const noexcept -> std::optional<EngineImage> { return scene_depth_; }
    [[nodiscard]] auto scene_hdr() const noexcept -> std::optional<EngineImage> { return scene_hdr_; }

    // A buffer that lives for this frame's graph. The engine allocates it (a buffer of at least `desc.size`, reused
    // between frames, never aliased with another resource) and the game never sees the VkBuffer. Name it uniquely: the
    // optional clear pass is called "<name>_clear".
    [[nodiscard]] auto create_buffer(GameBufferDesc const &desc, std::string_view name) -> GameBuffer;

    // A buffer that keeps its contents from frame to frame, looked up by name. The first request creates it
    // zero-filled; a request with another size recreates it, zero-filled, and the old one is destroyed once the
    // frames in flight are done with it. Every pass that touches it runs on the graphics queue.
    [[nodiscard]] auto persistent_buffer(std::string_view name, GameBufferDesc const &desc) -> GameBuffer;

    // Orders the forward pass (and the overlays recorded in it) after the passes that wrote `buffer`. Only possible
    // in frame_start and after_depth, before the forward pass is declared.
    auto scene_overlay_reads(GameBuffer buffer, frame_graph::ShaderStages stages) -> void;

    // Composition samples `image` instead of the forward pass's HDR image. Only in after_lighting / before_composite.
    auto replace_scene_colour(GameImage image) -> void;

    // `setup` receives a GameComputeBuilder, declares what the pass touches and returns a callable taking a
    // GameComputeContext &.
    template<typename Setup>
    auto add_compute_pass(std::string_view name, GamePassProfile profile, Setup &&setup) -> void {
        auto const pass_name = memory_->intern(name);
        auto const label = memory_->intern(profile.label);
        graph_->add_pass(
                pass_name, frame_graph::PassType::compute,
                frame_graph::PassProfile{.name_id = pass_name, .label = label, .color = profile.color},
                frame_graph::Owner::game, [&](frame_graph::PassBuilder &pass) {
                    auto builder = GameComputeBuilder{pass, *this};
                    return frame_graph::RecordFn{
                            [this, pass_name, record = std::forward<Setup>(setup)(builder)](
                                    frame_graph::PassContext &context) mutable {
                                auto game_context = GameComputeContext{context, services_};
                                record(game_context);
                                report_violation(pass_name, game_context.violation());
                            }};
                });
    }

    // A compute pass declared by its parameter struct (see PushLayout). The resource members of `params` are declared
    // on the pass and updated with the handles the pass wrote, so `params` can be used to wire the next pass. The
    // engine binds `shader`, pushes the layout and dispatches `threads`; `record` replaces that last step when the
    // pass needs several dispatches or other pushes in between, and may call context.dispatch().
    template<ParamStruct P, typename Record>
        requires std::invocable<Record &, GameComputeContext &, P const &>
    auto add_compute(std::string_view name, P &params, GameComputeShader shader, Threads threads, Record &&record,
                     GameComputeOptions const &options = {}) -> void {
        add_compute_pass(name,
                         GamePassProfile{.label = options.label.empty() ? name : options.label, .color = options.color},
                         [&](GameComputeBuilder &pass) {
                             pass.queue(options.queue);
                             if (options.side_effect) {
                                 pass.side_effect();
                             }
                             declare_params(pass, params, typename P::Layout{});

                             return [shader, threads, snapshot = params,
                                     record = std::forward<Record>(record)](GameComputeContext &context) mutable {
                                 // Resolved before binding, so an index or address problem is reported either way.
                                 auto const packed = pack_params(
                                         snapshot, [&](auto const &param) { return resolve(context, param); });
                                 context.begin(shader, threads);
                                 context.push_layout(packed.bytes.data(), packed.size);
                                 record(context, std::as_const(snapshot));
                             };
                         });
    }

    template<ParamStruct P>
    auto add_compute(std::string_view name, P &params, GameComputeShader shader, Threads threads,
                     GameComputeOptions const &options = {}) -> void {
        add_compute(
                name, params, shader, threads, [](GameComputeContext &context, P const &) { context.dispatch(); },
                options);
    }

    // add_compute for a parameter list built at run time. Too many push constants rejects the slot, as any
    // declaration problem does.
    auto add_compute(std::string_view name, DynamicParams &params, GameComputeShader shader, Threads threads,
                     GameComputeOptions const &options = {}) -> void;

    [[nodiscard]] static auto dynamic_push_size(DynamicParams const &params) noexcept -> std::size_t;

    // Declares one self-contained piece of a slot (a particle emitter, an effect). If anything it declares is
    // rejected, only that piece is dropped, with the same rollback as a slot; the reason is returned and logged once.
    // The slot carries on, and its other pieces are untouched.
    auto isolated(std::string_view what, std::function<void()> const &declare) -> std::optional<std::string>;
    [[nodiscard]] auto rejected_units() const noexcept -> std::uint32_t { return rejected_units_; }

    // The image composition would sample right now: the one a game replaced the scene colour with, else the forward
    // pass's HDR image. Nothing before after_lighting.
    [[nodiscard]] auto scene_colour_source() const -> std::optional<ImageRead>;

    // A draw recorded inside the forward pass, after the engine's own scene overlays. `setup` receives a
    // GameDrawBuilder, declares the buffers it reads and returns a callable taking a GameDrawContext &.
    template<typename Setup>
    auto add_scene_draw(std::string_view name, Setup &&setup) -> void {
        auto builder = GameDrawBuilder{*this};
        auto record = std::forward<Setup>(setup)(builder);
        scene_draws_.push_back(SceneDraw{
                .name = memory_->intern(name),
                .reads = std::move(builder.reads_),
                .record = std::move(record),
        });
    }

    // The bytes of a parameter struct's push constants (see PushLayout). `resolve` turns a resource member into what it
    // pushes: a VkDeviceAddress for buffers, a std::uint32_t index for images. Public so the layout can be checked
    // without a GPU.
    struct PackedPush {
        std::array<std::byte, max_game_push_bytes> bytes{};
        std::size_t size = 0;
    };

    template<typename T>
    struct PushField {
        using type = T;
    };
    template<ResourceParam T>
    struct PushField<T> {
        using type = typename T::push_type;
    };

    template<typename P, auto Member>
    using PushValue = typename PushField<std::remove_cvref_t<decltype(std::declval<P const &>().*Member)>>::type;

    template<typename P, auto... Members>
    static constexpr auto push_size(PushLayout<Members...>) noexcept -> std::size_t {
        auto offset = std::size_t{0};
        ((offset = ((offset + alignof(PushValue<P, Members>) - 1U) & ~(alignof(PushValue<P, Members>) - 1U)) +
                   sizeof(PushValue<P, Members>)),
         ...);
        return offset;
    }

    template<ParamStruct P, typename Resolve>
    static auto pack_params(P const &params, Resolve &&resolve) -> PackedPush {
        return pack_layout(params, resolve, typename P::Layout{});
    }

    // Engine side.

    struct ForwardRead {
        frame_graph::BufferId buffer{};
        frame_graph::ShaderStages stages = 0;
    };

    // Runs `hook` for one slot. If anything it declared is rejected, the whole slot is rolled back.
    auto run_slot(GameSlot slot, std::optional<frame_graph::ImageId> depth, std::optional<frame_graph::ImageId> hdr,
                  std::function<void(GameGraph &)> const &hook) -> void;

    [[nodiscard]] auto forward_reads() const noexcept -> std::span<ForwardRead const> { return forward_reads_; }
    [[nodiscard]] auto scene_colour() const noexcept -> std::optional<frame_graph::ImageId> { return scene_colour_; }
    [[nodiscard]] auto has_scene_draws() const noexcept -> bool { return !scene_draws_.empty(); }
    [[nodiscard]] auto rolled_back_slots() const noexcept -> std::uint32_t { return rolled_back_slots_; }
    [[nodiscard]] auto problems() const noexcept -> std::span<std::string const> { return problems_; }

    auto record_scene_draws(frame_graph::PassContext const &pass, OverlayRecordContext const &overlay) -> void;

private:
    friend class GameComputeBuilder;
    friend class GameDrawBuilder;
    friend class GameComputeContext;

    struct SceneDraw {
        std::string_view name;
        std::vector<std::uint32_t> reads;
        std::move_only_function<void(GameDrawContext &)> record;
    };

    struct BufferRecord {
        std::string_view name;
        std::uint32_t resource = 0;
        VkDeviceSize size = 0;
        bool persistent = false;
        // Created without zero-fill and not written yet: a read would see whatever the pooled buffer held.
        bool undefined = false;
    };

    [[nodiscard]] auto find_buffer(std::uint32_t resource) const -> BufferRecord const *;
    // Records a problem for a read of a buffer nothing wrote yet. True if the read is fine.
    auto check_initialised(std::uint32_t resource) -> bool;
    [[nodiscard]] auto acquire(std::string_view name, GameBufferDesc const &desc, bool persistent) -> GameBuffer;
    [[nodiscard]] auto valid_buffer_desc(std::string_view name, GameBufferDesc const &desc) -> bool;

    template<typename P, auto... Members>
    auto declare_params(GameComputeBuilder &pass, P &params, PushLayout<Members...>) -> void {
        (declare_param(pass, params.*Members), ...);
    }

    template<typename T>
    auto declare_param(GameComputeBuilder &, T &) -> void {
        static_assert(std::is_trivially_copyable_v<T>, "a push constant member must be trivially copyable");
    }
    auto declare_param(GameComputeBuilder &pass, BufferRead &param) -> void { param.declared = pass.read(param.buffer); }
    auto declare_param(GameComputeBuilder &pass, BufferWrite &param) -> void {
        param.declared = pass.write(param.buffer);
    }
    auto declare_param(GameComputeBuilder &pass, BufferReadWrite &param) -> void {
        param.declared = pass.read_write(param.buffer);
    }
    auto declare_param(GameComputeBuilder &pass, ImageRead &param) -> void {
        param.declared = std::visit([&](auto image) { return pass.sample(image); }, param.source);
    }
    auto declare_param(GameComputeBuilder &pass, ImageWrite &param) -> void {
        if (!param.image.valid() && param.create) {
            param.image = pass.create_image(*param.create);
        }
        param.declared = pass.write(param.image);
    }

    static auto resolve(GameComputeContext &context, BufferRead const &param) -> VkDeviceAddress {
        return context.address(param.declared);
    }
    static auto resolve(GameComputeContext &context, BufferWrite const &param) -> VkDeviceAddress {
        return context.address(param.declared);
    }
    static auto resolve(GameComputeContext &context, BufferReadWrite const &param) -> VkDeviceAddress {
        return context.address(param.declared);
    }
    static auto resolve(GameComputeContext &context, ImageRead const &param) -> std::uint32_t {
        return context.sampled_index(param.declared);
    }
    static auto resolve(GameComputeContext &context, ImageWrite const &param) -> std::uint32_t {
        return context.storage_index(param.declared);
    }

    template<typename P, typename Resolve, auto... Members>
    static auto pack_layout(P const &params, Resolve &resolve, PushLayout<Members...> layout) -> PackedPush {
        static_assert(sizeof...(Members) > 0, "a Layout needs at least one member");
        static_assert(push_size<P>(PushLayout<Members...>{}) <= max_game_push_bytes,
                      "the push constants of a pass are limited to max_game_push_bytes");
        auto packed = PackedPush{.size = push_size<P>(layout)};
        auto offset = std::size_t{0};
        auto const append = [&]<auto Member>() {
            using Value = PushValue<P, Member>;
            auto const &member = params.*Member;
            auto const value = [&]() -> Value {
                if constexpr (ResourceParam<decltype(member)>) {
                    return resolve(member);
                } else {
                    return member;
                }
            }();
            offset = (offset + alignof(Value) - 1U) & ~(alignof(Value) - 1U);
            std::memcpy(packed.bytes.data() + offset, &value, sizeof(Value));
            offset += sizeof(Value);
        };
        (append.template operator()<Members>(), ...);
        return packed;
    }

    struct Unit {
        frame_graph::FrameGraph::Checkpoint mark;
        std::vector<ForwardRead> forward_reads;
        std::optional<frame_graph::ImageId> scene_colour;
        std::size_t draws = 0;
        std::size_t buffers = 0;
        std::size_t problems = 0;
    };

    [[nodiscard]] auto begin_unit() const -> Unit;
    // Rolls back to `unit` if anything since was rejected and returns what was dropped; with `keep_problems` false the
    // problems since are removed too, so they do not reject whatever encloses the unit.
    [[nodiscard]] auto end_unit(Unit const &unit, bool keep_problems) -> std::optional<std::string>;

    auto report_violation(std::string_view pass, std::string_view violation) -> void;
    auto add_forward_read(frame_graph::BufferId buffer, frame_graph::ShaderStages stages) -> bool;

    frame_graph::FrameGraph *graph_;
    GameGraphMemory *memory_;
    GameGraphServices services_;
    GameSlot slot_ = GameSlot::frame_start;
    std::optional<EngineImage> scene_depth_;
    std::optional<EngineImage> scene_hdr_;
    std::optional<frame_graph::ImageId> scene_colour_;
    std::vector<ForwardRead> forward_reads_;
    std::vector<SceneDraw> scene_draws_;
    std::vector<BufferRecord> buffers_;
    std::vector<std::string> problems_;
    std::uint32_t rolled_back_slots_ = 0;
    std::uint32_t rejected_units_ = 0;
};
