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

// Overlays are draws recorded inside a rendering scope a pass has already opened: debug lines and light icons
// over the scene, ImGui over the final image. They run every frame until their OverlayRegistration is
// destroyed.
//
//   record()  runs inside the host pass's vkCmdBeginRendering scope. It may bind shaders and descriptors, set
//             dynamic state, push constants and draw. It must not begin or end rendering, record barriers or
//             dispatches, write timestamps or change image layouts. The host resets the baseline dynamic
//             state (set_overlay_baseline_state() in render_passes.hxx) before each overlay's record().
//
//   prepare() optional; runs before the frame's first pass, outside any rendering scope. It may record
//             transfer/compute writes to buffers the overlay owns and returns whether it did. If any overlay
//             did, the renderer records one barrier from those writes to every stage a draw can read them
//             from. Barriers between an overlay's own prepare commands are its own business. Host writes to
//             mapped memory need no barrier, so memcpy-only overlays return nothing_recorded.
//
// Per-frame resources must be indexed by frame_index; only that slot's previous use is guaranteed retired.
//
// The renderer times each overlay's prepare() and record() on the GPU and opens a Tracy zone for them (see
// FrameTimings::overlays).
//
// Registration is render-thread only. Callbacks may add or remove overlays: removals apply once the frame's
// recording finishes, additions take effect next frame.

enum class OverlayStage : std::uint8_t {
    // Inside the forward pass after the scene draws: HDR colour, reverse-Z depth (GREATER_OR_EQUAL, written by
    // the scene), forward target MSAA. view_projection is valid.
    scene,

    // The last scope rendering into the swapchain image. Swapchain-format colour, no depth, single sample.
    ui,

    count,
};

inline constexpr auto overlay_stage_count = static_cast<std::size_t>(OverlayStage::count);

// The scope an overlay records into, so it can check its assumptions and size its draws.
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

    // projection * view
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

// Unregisters the overlay when destroyed or reset(). Must not outlive the registry (the Renderer).
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

// CPU-side overlay bookkeeping: ordering, stable timing slots and deferred removal. No Vulkan, so it can be
// unit tested.
class OverlayRegistry {
public:
    // Sizes the per-frame timestamp range reserved for overlays.
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

    // While the guard lives, add()/remove() are queued instead of touching the entries being iterated.
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

    // Bit i set: timing slot i is taken.
    std::uint32_t used_slots_ = 0;
    static_assert(max_overlays <= 32);

    OverlayId next_id_ = 1;
    std::uint64_t next_sequence_ = 0;
};

// GPU time of one overlay's callbacks in a finished frame.
struct OverlayTiming {
    std::string name;
    OverlayStage stage = OverlayStage::scene;
    float prepare_milliseconds = 0.0F;
    float record_milliseconds = 0.0F;
};
