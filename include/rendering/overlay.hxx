#pragma once

#include <volk.h>

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Overlays: draws that ride along inside a rendering scope a real pass has
// already opened -- debug lines and light icons over the scene, ImGui over
// the final image. They are registered once and run every frame until their
// OverlayRegistration is destroyed.
//
// The contract, which is what keeps barriers and GPU timing out of an
// overlay's hands:
//
//   record()  runs inside an open vkCmdBeginRendering scope owned by the
//             host pass. It may only bind shaders/descriptors, set dynamic
//             state, push constants and draw. It must not begin or end
//             rendering, record barriers or dispatches, write timestamps or
//             change image layouts. Before *each* overlay's record() the
//             host resets the baseline dynamic state documented on
//             set_overlay_baseline_state() (render_passes.hxx), so an
//             overlay only sets what differs from that baseline and never
//             inherits state from whatever drew before it.
//
//   prepare() (optional) runs before the first pass of the frame, outside
//             any rendering scope. It may record transfer/compute work that
//             writes buffers the overlay owns, and returns whether it did.
//             If any overlay did, the renderer records one global memory
//             barrier from those writes to every stage an overlay's draw
//             can read them from (indirect, index, vertex, task, mesh,
//             fragment). Barriers *between* an overlay's own prepare
//             commands are the overlay's business. Host writes to mapped
//             memory need neither -- vkQueueSubmit already makes them
//             visible -- so an overlay that only memcpy's into a mapped
//             buffer returns nothing_recorded.
//
// Per-frame resources an overlay writes must be indexed by frame_index:
// the renderer only guarantees that frame slot's previous use has retired.
//
// The renderer writes a GPU timestamp pair around every overlay's
// prepare() and record() and opens a Tracy zone named after it, so neither
// callback times itself. See StageTimings::overlays.
//
// Registration is render-thread only. Adding or removing an overlay from
// inside a prepare()/record() callback is allowed; removals are applied once
// the frame's recording finishes, additions take effect next frame.

enum class OverlayStage : std::uint8_t {
    // Inside the forward pass, after every scene draw: HDR colour +
    // reverse-Z depth (tested GREATER_OR_EQUAL, written by the scene),
    // MSAA at the forward target's sample count. view_projection is valid.
    scene,

    // The last scope that renders into the swapchain image: after
    // tonemapping in fullscreen play, or the editor's UI pass otherwise.
    // Swapchain-format colour, no depth, single sample.
    ui,

    count,
};

inline constexpr auto overlay_stage_count = static_cast<std::size_t>(OverlayStage::count);

// Describes the rendering scope an overlay records into. With shader
// objects nothing is baked against it; it exists so an overlay can check
// its assumptions and size its draws.
struct OverlayScope {
    VkExtent2D extent{};
    VkFormat colour_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED; // UNDEFINED: no depth attachment
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    [[nodiscard]] auto has_depth() const noexcept -> bool { return depth_format != VK_FORMAT_UNDEFINED; }
};

struct OverlayPrepareContext {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    std::uint32_t frame_index = 0;
};

struct OverlayRecordContext {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    std::uint32_t frame_index = 0;
    OverlayScope scope{};

    // The camera this frame was prepared with (projection * view).
    glm::mat4 view_projection{1.0F};
};

enum class OverlayPrepareResult : std::uint8_t {
    nothing_recorded,
    recorded_gpu_writes,
};

struct OverlayDesc {
    std::string name;
    OverlayStage stage = OverlayStage::scene;

    // Lower draws first within a stage; ties keep registration order.
    std::int32_t order = 0;

    std::move_only_function<OverlayPrepareResult(OverlayPrepareContext const &)> prepare{};
    std::move_only_function<void(OverlayRecordContext const &)> record{};
};

enum class OverlayRegistryError : std::uint8_t {
    empty_name,
    missing_record_callback,
    invalid_stage,
    capacity_exceeded,
};

using OverlayId = std::uint32_t;

class OverlayRegistry;

// Owns one registered overlay; destroying (or reset()ing) it unregisters
// the overlay. Must not outlive the registry it came from -- in practice,
// the Renderer.
class OverlayRegistration {
public:
    OverlayRegistration() = default;
    ~OverlayRegistration();

    OverlayRegistration(OverlayRegistration const &) = delete;
    auto operator=(OverlayRegistration const &) -> OverlayRegistration & = delete;

    OverlayRegistration(OverlayRegistration &&other) noexcept;
    auto operator=(OverlayRegistration &&other) noexcept -> OverlayRegistration &;

    auto reset() noexcept -> void;

    [[nodiscard]] auto valid() const noexcept -> bool { return registry_ != nullptr; }
    [[nodiscard]] auto id() const noexcept -> OverlayId { return id_; }

private:
    friend class OverlayRegistry;

    OverlayRegistration(OverlayRegistry &registry, OverlayId id) noexcept : registry_{&registry}, id_{id} {}

    OverlayRegistry *registry_ = nullptr;
    OverlayId id_ = 0;
};

// CPU-side bookkeeping for overlays: ordering, stable timing slots and
// deferred removal. Knows nothing about Vulkan beyond the callback types,
// so it is unit-testable without a device.
class OverlayRegistry {
public:
    // Upper bound on simultaneously registered overlays; sizes the
    // per-frame timestamp query range the renderer reserves for them.
    static constexpr std::uint32_t max_overlays = 16;

    struct Entry {
        OverlayId id = 0;

        // Stable for the registration's lifetime, in [0, max_overlays).
        std::uint32_t slot = 0;

        std::uint64_t sequence = 0;
        OverlayDesc desc;
    };

    OverlayRegistry();

    OverlayRegistry(OverlayRegistry const &) = delete;
    auto operator=(OverlayRegistry const &) -> OverlayRegistry & = delete;
    OverlayRegistry(OverlayRegistry &&) = delete;
    auto operator=(OverlayRegistry &&) -> OverlayRegistry & = delete;

    [[nodiscard]]
    auto add(OverlayDesc desc) -> std::expected<OverlayRegistration, OverlayRegistryError>;

    // Idempotent: removing an unknown or already-removed id is a no-op.
    auto remove(OverlayId id) noexcept -> void;

    // Overlays of one stage, in draw order.
    [[nodiscard]] auto stage(OverlayStage stage) noexcept -> std::span<Entry>;

    // Every overlay, grouped by stage and then in draw order.
    [[nodiscard]] auto all() noexcept -> std::span<Entry> { return entries_; }

    [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }

    // While the returned guard lives, add()/remove() are queued instead of
    // touching the entry vector the renderer is iterating.
    class IterationGuard {
    public:
        explicit IterationGuard(OverlayRegistry &registry) noexcept;
        ~IterationGuard();

        IterationGuard(IterationGuard const &) = delete;
        auto operator=(IterationGuard const &) -> IterationGuard & = delete;
        IterationGuard(IterationGuard &&) = delete;
        auto operator=(IterationGuard &&) -> IterationGuard & = delete;

    private:
        OverlayRegistry &registry_;
    };

    [[nodiscard]] auto iterate() noexcept -> IterationGuard { return IterationGuard{*this}; }

private:
    auto insert(Entry entry) -> void;
    auto erase(OverlayId id) noexcept -> void;
    auto flush_pending() -> void;

    std::vector<Entry> entries_;

    std::vector<Entry> pending_additions_;
    std::vector<OverlayId> pending_removals_;
    std::uint32_t iteration_depth_ = 0;

    // Bit i set: timing slot i is taken (by a live or pending entry).
    std::uint32_t used_slots_ = 0;
    static_assert(max_overlays <= 32);

    OverlayId next_id_ = 1;
    std::uint64_t next_sequence_ = 0;
};

// Wall-clock GPU time of one overlay's callbacks in a finished frame.
struct OverlayTiming {
    std::string name;
    OverlayStage stage = OverlayStage::scene;
    float prepare_milliseconds = 0.0F;
    float record_milliseconds = 0.0F;
};
