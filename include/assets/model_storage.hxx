#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "assets/load_model.hxx"
#include "assets/model.hxx"
#include "core/object_pool.hxx"

// One flattened draw: the mesh and its model-space transform with every ancestor folded in.
struct ModelDraw {
    MeshHandle mesh{};
    glm::mat4 local_transform{1.0F};
};

struct ModelSlotData {
    std::vector<ModelDraw> draws;

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};

    std::vector<ModelCpuLight> lights;

    // Callers holding this handle. The model caches retain it when handing it out again; destroy_model() only
    // frees the slot once it reaches zero. create_model()/upgrade_pending_model() handle it explicitly rather than
    // overwriting it.
    std::uint32_t ref_count = 1;
};

enum class ModelStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
};

struct ModelStorageError {
    ModelStorageErrorType type = ModelStorageErrorType::invalid_argument;
};

template<>
struct std::formatter<ModelStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(ModelStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case ModelStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case ModelStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case ModelStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
            }

            return "unknown_model_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct ModelStorageCreateInfo {
    std::uint32_t capacity = 0;
};

// Generational pool of CPU-side, flattened model scene graphs. Owns no Vulkan resources.
class ModelStorage {
public:
    ModelStorage() = default;

    ModelStorage(ModelStorage const &) = delete;
    auto operator=(ModelStorage const &) -> ModelStorage & = delete;

    ModelStorage(ModelStorage &&) noexcept = default;
    auto operator=(ModelStorage &&) noexcept -> ModelStorage & = default;

    [[nodiscard]]
    static auto create(ModelStorageCreateInfo const &create_info) -> std::expected<ModelStorage, ModelStorageError>;

    [[nodiscard]]
    auto create_model(ModelSlotData data) -> std::expected<ModelHandle, ModelStorageError>;

    // Reserves a slot holding a copy of `fallback`'s data until upgrade_pending_model() installs the real model.
    [[nodiscard]]
    auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, ModelStorageError>;

    // Replaces a slot's data in place; the handle stays the same.
    [[nodiscard]]
    auto upgrade_pending_model(ModelHandle handle, ModelSlotData data) -> std::expected<ModelHandle, ModelStorageError>;

    // Releases the slot. ref_count must already be zero and the draws' meshes torn down.
    [[nodiscard]]
    auto release(ModelHandle handle) -> std::expected<void, ModelStorageError>;

    [[nodiscard]]
    auto get(ModelHandle handle) noexcept -> ModelSlotData *;

    [[nodiscard]]
    auto get(ModelHandle handle) const noexcept -> ModelSlotData const *;

    [[nodiscard]]
    auto contains(ModelHandle handle) const noexcept -> bool {
        return get(handle) != nullptr;
    }

    [[nodiscard]]
    auto size() const noexcept -> std::uint32_t {
        return slots_.size();
    }

    [[nodiscard]]
    auto capacity() const noexcept -> std::uint32_t {
        return slots_.capacity();
    }

    auto destroy() noexcept -> void;

private:
    ObjectPool<ModelSlotData, 0> slots_;
};
