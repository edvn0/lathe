#pragma once

#include <volk.h>

#include <glm/mat4x4.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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
};

// A GPU buffer owned by GameGpu, with one copy per frame in flight so frame N never races frame N - 1.
struct GameBufferHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] auto valid() const noexcept -> bool { return generation != 0; }
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

private:
    friend class GameGraph;
    friend class GameComputeBuilder;
    explicit GameImage(frame_graph::ImageId id) : id_{id} {}

    frame_graph::ImageId id_{};
};

class GameBuffer {
public:
    GameBuffer() = default;

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
private:
    friend class GameComputeBuilder;
    friend class GameComputeContext;
    DeclaredImage(std::uint32_t resource, bool writable) : resource_{resource}, writable_{writable} {}

    std::uint32_t resource_ = 0;
    bool writable_ = false;
};

class DeclaredBuffer {
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

struct GamePassProfile {
    std::string_view label;
    std::uint32_t color = 0;
};

// What the engine lends a game graph. Everything is optional so the declaration side can run without a GPU.
struct GameGraphServices {
    Renderer *renderer = nullptr;
    std::uint32_t frame_index = 0;
    VkExtent2D scene_extent{};
    std::array<std::uint32_t, 3> max_group_count{65535U, 65535U, 65535U};

    // Bindless index of the transient image behind a graph resource.
    std::function<std::uint32_t(std::uint32_t resource)> bindless_index;
    // Imports this frame's copy of a game buffer.
    std::function<std::optional<frame_graph::BufferId>(GameBufferHandle)> import_buffer;
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

class GameComputeContext {
public:
    auto bind(GameComputeShader shader) -> void;

    template<typename Push>
    auto push(Push const &constants) -> void {
        static_assert(std::is_trivially_copyable_v<Push>);
        static_assert(sizeof(Push) <= max_game_push_bytes);
        push_bytes(&constants, sizeof(Push));
    }

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
    [[nodiscard]] auto image_index(DeclaredImage image, bool writable) -> std::uint32_t;

    frame_graph::PassContext *pass_;
    GameGraphServices const *services_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    // The shader is still compiling or failed to: the pass is skipped for this frame without being an error.
    bool skipped_ = false;
    std::string violation_;
};

class GameComputeBuilder {
public:
    auto queue(frame_graph::QueueAffinity affinity) -> void { pass_->queue(affinity); }
    // Keeps the pass even when nothing in the graph reads its output.
    auto side_effect() -> void { pass_->side_effect(); }

    [[nodiscard]] auto sample(EngineImage image) -> DeclaredImage;
    [[nodiscard]] auto sample(GameImage image) -> DeclaredImage;
    [[nodiscard]] auto read(GameBuffer buffer) -> DeclaredBuffer;
    // Advance the handle they are given, so a later pass reads what this one wrote.
    [[nodiscard]] auto write(GameImage &image) -> DeclaredImage;
    [[nodiscard]] auto read_write(GameBuffer &buffer) -> DeclaredBuffer;
    [[nodiscard]] auto create_image(GameImageDesc const &desc) -> GameImage;

    // There is deliberately no overload that writes an EngineImage.

private:
    friend class GameGraph;
    GameComputeBuilder(frame_graph::PassBuilder &pass, GameGraph &graph) : pass_{&pass}, graph_{&graph} {}

    frame_graph::PassBuilder *pass_;
    GameGraph *graph_;
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
          view_projection_{overlay.view_projection}, reads_{reads} {}

    auto fail(std::string_view what) -> void;
    auto push_bytes(void const *data, std::size_t size) -> void;

    frame_graph::PassContext const *pass_;
    GameGraphServices const *services_;
    VkCommandBuffer command_buffer_;
    glm::mat4 view_projection_;
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

    [[nodiscard]] auto import(GameBufferHandle buffer) -> GameBuffer;

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

    struct SceneDraw {
        std::string_view name;
        std::vector<std::uint32_t> reads;
        std::move_only_function<void(GameDrawContext &)> record;
    };

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
    // Resource index of every game buffer imported this frame, by GameBufferHandle::index.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> imported_;
    std::vector<std::string> problems_;
    std::uint32_t rolled_back_slots_ = 0;
};
