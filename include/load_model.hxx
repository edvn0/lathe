#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <vector>

#include "geometry_arena.hxx"

struct ModelVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec4 tangent{};
    glm::vec2 texcoord{};
};

struct ModelPrimitive {
    MeshGeometry geometry{};
    std::uint32_t material_index = 0;
};

struct ModelMesh {
    std::vector<ModelPrimitive> primitives;
};

struct ModelNode {
    glm::mat4 local_transform{1.0F};

    std::uint32_t mesh_index = std::numeric_limits<std::uint32_t>::max();

    std::vector<std::uint32_t> children{};
};

struct Model {
    std::vector<ModelMesh> meshes;
    std::vector<ModelNode> nodes;
    std::vector<std::uint32_t> scene_roots;
};

enum class ModelLoadErrorType : std::uint8_t {
    file_not_found,
    parse_error,
    unsupported_primitive,
    missing_position,
    invalid_accessor,
    geometry_upload_failed,
    tangent_generation_failed,
};

struct ModelLoadError {
    ModelLoadErrorType type = ModelLoadErrorType::parse_error;

    GeometryArenaError geometry_error{};
};

[[nodiscard]]
auto load_model(std::filesystem::path const &path, GeometryArena &geometry_arena)
        -> std::expected<Model, ModelLoadError>;
