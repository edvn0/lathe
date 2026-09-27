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

struct ScriptSlotData {
    std::unique_ptr<IScript> script;
};

// Generational pool of script instances. Each handle owns one IScript shared by every entity referencing it.
class ScriptStorage {
public:
    ScriptStorage() = default;

    ScriptStorage(ScriptStorage const &) = delete;
    auto operator=(ScriptStorage const &) -> ScriptStorage & = delete;

    ScriptStorage(ScriptStorage &&) noexcept = default;
    auto operator=(ScriptStorage &&) noexcept -> ScriptStorage & = default;

    [[nodiscard]]
    static auto create(ScriptStorageCreateInfo const &create_info) -> std::expected<ScriptStorage, ScriptStorageError>;

    // Creates one shared instance. Call once per behaviour and share the handle, not once per entity.
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
