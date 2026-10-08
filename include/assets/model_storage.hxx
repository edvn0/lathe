#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "assets/load_model.hxx"
#include "assets/model.hxx"
#include "core/object_pool.hxx"

struct ModelDraw {
    MeshHandle mesh{};
    glm::mat4 local_transform{1.0F};
};

struct ModelSlotData {
    std::vector<ModelDraw> draws;

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};

    std::vector<ModelCpuLight> lights;

    std::shared_ptr<ModelAnimationData const> animation;
    float skin_inflate = 0.0F;

    std::uint32_t ref_count = 1;

    ModelHandle borrowed_from{};
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

    [[nodiscard]]
    auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, ModelStorageError>;

    [[nodiscard]]
    auto upgrade_pending_model(ModelHandle handle, ModelSlotData data) -> std::expected<ModelHandle, ModelStorageError>;

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
