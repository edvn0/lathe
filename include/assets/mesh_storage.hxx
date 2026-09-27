#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "core/config.hxx"
#include "assets/geometry.hxx"
#include "assets/material.hxx"
#include "assets/model.hxx"
#include "core/object_pool.hxx"

// One submesh: GeometryArena ranges per LOD, material and local bounds. Validated by Renderer::create_mesh.
struct Submesh {
    std::array<MeshGeometry, lod_count> lods{};
    MaterialHandle material{};

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

struct MeshSlotData {
    std::vector<Submesh> submeshes;
};

enum class MeshStorageErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
};

struct MeshStorageError {
    MeshStorageErrorType type = MeshStorageErrorType::invalid_argument;
};

template<>
struct std::formatter<MeshStorageErrorType> : std::formatter<std::string_view> {
    constexpr auto format(MeshStorageErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case MeshStorageErrorType::invalid_argument:
                    return "invalid_argument";
                case MeshStorageErrorType::invalid_handle:
                    return "invalid_handle";
                case MeshStorageErrorType::capacity_exceeded:
                    return "capacity_exceeded";
            }

            return "unknown_mesh_storage_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct MeshStorageCreateInfo {
    std::uint32_t capacity = 0;
};

// Generational pool of mesh records. The geometry itself lives in GeometryArena, so no Vulkan resources here.
class MeshStorage {
public:
    MeshStorage() = default;

    MeshStorage(MeshStorage const &) = delete;
    auto operator=(MeshStorage const &) -> MeshStorage & = delete;

    MeshStorage(MeshStorage &&) noexcept = default;
    auto operator=(MeshStorage &&) noexcept -> MeshStorage & = default;

    [[nodiscard]]
    static auto create(MeshStorageCreateInfo const &create_info) -> std::expected<MeshStorage, MeshStorageError>;

    [[nodiscard]]
    auto create_mesh(std::vector<Submesh> submeshes) -> std::expected<MeshHandle, MeshStorageError>;

    [[nodiscard]]
    auto destroy_mesh(MeshHandle handle) -> std::expected<void, MeshStorageError>;

    [[nodiscard]]
    auto get(MeshHandle handle) noexcept -> MeshSlotData *;

    [[nodiscard]]
    auto get(MeshHandle handle) const noexcept -> MeshSlotData const *;

    [[nodiscard]]
    auto contains(MeshHandle handle) const noexcept -> bool {
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
    ObjectPool<MeshSlotData, 0> slots_;
};
