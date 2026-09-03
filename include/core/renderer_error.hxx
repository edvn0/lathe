#pragma once

#include <format>
#include <string_view>

#include "core/error_context.hxx"

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

template<>
struct std::formatter<RendererErrorType> : std::formatter<std::string_view> {
    constexpr auto format(RendererErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case RendererErrorType::invalid_argument:
                    return "invalid_argument";
                case RendererErrorType::invalid_mesh:
                    return "invalid_mesh";
                case RendererErrorType::invalid_model:
                    return "invalid_model";
                case RendererErrorType::invalid_material:
                    return "invalid_material";
                case RendererErrorType::unsupported_index_type:
                    return "unsupported_index_type";
                case RendererErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case RendererErrorType::size_overflow:
                    return "size_overflow";
                case RendererErrorType::model_load_error:
                    return "model_load_error";
                case RendererErrorType::geometry_error:
                    return "geometry_error";
                case RendererErrorType::material_error:
                    return "material_error";
                case RendererErrorType::image_error:
                    return "image_error";
                case RendererErrorType::forward_target_error:
                    return "forward_target_error";
                case RendererErrorType::device_error:
                    return "device_error";
                case RendererErrorType::pipeline_error:
                    return "pipeline_error";
                case RendererErrorType::compiler_error:
                    return "compiler_error";
                case RendererErrorType::invalid_pipeline:
                    return "invalid_pipeline";
                case RendererErrorType::pipeline_storage_error:
                    return "pipeline_storage_error";
                case RendererErrorType::gpu_resource_table_error:
                    return "gpu_resource_table_error";
                case RendererErrorType::pipeline_graph_error:
                    return "pipeline_graph_error";
            }

            return "unknown_renderer_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
