#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "core/object_pool.hxx"
#include "rendering/script.hxx"
#include "scene/script_handle.hxx"

enum class ScriptStorageErrorType : std::uint8_t {
    invalid_argument,
    capacity_exceeded,
};

struct ScriptStorageError {
    ScriptStorageErrorType type = ScriptStorageErrorType::invalid_argument;
};

struct ScriptStorageCreateInfo {
    std::uint32_t capacity = 0;
};

// Backs ScriptHandle = Handle<ScriptSlotData, 0> (see scene/script_handle.hxx).
struct ScriptSlotData {
    std::unique_ptr<IScript> script;
};

// Parallel to MeshStorage: a generational-handle ObjectPool holding CPU-side
// script instances. Purely CPU-side, no GPU/Vulkan aspect at all -- unlike
// the other *Storage classes here, this one's only job is to own one shared
// IScript per handle so many entities can reference the same instance (see
// script.hxx).
class ScriptStorage {
public:
    ScriptStorage() = default;

    ScriptStorage(ScriptStorage const &) = delete;
    auto operator=(ScriptStorage const &) -> ScriptStorage & = delete;

    ScriptStorage(ScriptStorage &&) noexcept = default;
    auto operator=(ScriptStorage &&) noexcept -> ScriptStorage & = default;

    [[nodiscard]]
    static auto create(ScriptStorageCreateInfo const &create_info) -> std::expected<ScriptStorage, ScriptStorageError>;

    // Allocates ONE shared instance and returns a handle to it -- every
    // entity that gets this handle via Components::Script resolves to this
    // same IScript&. Do not call this once per entity; call it once per
    // logical behavior and share the returned handle.
    template<typename T, typename... Args>
    [[nodiscard]] auto emplace(Args &&...args) -> std::expected<ScriptHandle, ScriptStorageError> {
        auto allocation = slots_.allocate();

        if (!allocation) {
            return std::unexpected(ScriptStorageError{.type = ScriptStorageErrorType::capacity_exceeded});
        }

        allocation->second.script = std::make_unique<T>(std::forward<Args>(args)...);

        return allocation->first;
    }

    [[nodiscard]]
    auto get(ScriptHandle handle) noexcept -> IScript *;

    auto destroy(ScriptHandle handle) -> void;

private:
    ObjectPool<ScriptSlotData, 0> slots_;
};
