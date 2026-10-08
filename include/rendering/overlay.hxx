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

enum class OverlayStage : std::uint8_t {
    scene,

    ui,

    count,
};

inline constexpr auto overlay_stage_count = static_cast<std::size_t>(OverlayStage::count);

struct OverlayScope {
    VkExtent2D extent{};
    VkFormat colour_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    std::uint32_t colour_attachment_count = 1;

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

    glm::mat4 view_projection{1.0F};
};

enum class OverlayPrepareResult : std::uint8_t {
    nothing_recorded,
    recorded_gpu_writes,
};

struct OverlayDesc {
    std::string name;
    OverlayStage stage = OverlayStage::scene;

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

class OverlayRegistry {
public:
    static constexpr std::uint32_t max_overlays = 16;

    struct Entry {
        OverlayId id = 0;

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

    auto remove(OverlayId id) noexcept -> void;

    [[nodiscard]] auto stage(OverlayStage stage) noexcept -> std::span<Entry>;

    [[nodiscard]] auto all() noexcept -> std::span<Entry> { return entries_; }

    [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }

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

    std::uint32_t used_slots_ = 0;
    static_assert(max_overlays <= 32);

    OverlayId next_id_ = 1;
    std::uint64_t next_sequence_ = 0;
};

struct OverlayTiming {
    std::string name;
    OverlayStage stage = OverlayStage::scene;
    float prepare_milliseconds = 0.0F;
    float record_milliseconds = 0.0F;
};
