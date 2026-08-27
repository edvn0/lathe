#pragma once

#include "error_context.hxx"

enum class RendererErrorType : std::uint8_t {
    invalid_argument,
    invalid_mesh,
    invalid_model,
    invalid_material,
    unsupported_index_type,
    capacity_exceeded,
    size_overflow,
    model_load_error,
    geometry_error,
    material_error,
    image_error,
    forward_target_error,
    device_error,
    pipeline_error,
    compiler_error,
    pipeline_storage_error,
    gpu_resource_table_error,
    pipeline_graph_error,
    invalid_pipeline,
};
struct RendererError {
    RendererErrorType type = RendererErrorType::invalid_argument;

    std::optional<ErrorCause> cause{std::nullopt};
};
