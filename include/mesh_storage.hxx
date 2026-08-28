#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "config.hxx"
#include "geometry.hxx"
#include "material.hxx"
#include "model.hxx"
#include "object_pool.hxx"

// Renderer's persistent, GPU-upload-free record of one submesh: the
// GeometryArena ranges for each LOD, plus the material and local bounds a
// draw call needs. Built by Renderer::create_mesh (which owns the
// GeometryArena/MaterialStorage validation MeshStorage deliberately doesn't
// know about) and stored verbatim.
struct Submesh {
    std::array<MeshGeometry, lod_count> lods{};
    MaterialHandle material{};

    glm::vec3 bounds_min{-0.5F};
    glm::vec3 bounds_max{0.5F};
};

// Backs MeshHandle = Handle<MeshSlotData, 0> (see model.hxx).
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

// Parallel to PipelineStorage/MaterialStorage: a generational-handle
// ObjectPool holding CPU-side mesh records. Unlike the other storages this
// one owns no Vulkan resources at all -- the actual vertex/index data lives
// in GeometryArena, referenced by Submesh::lods -- so there's nothing here
// that needs a VulkanContext or a destroy() beyond dropping its containers.
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
