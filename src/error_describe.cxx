#include "error_describe.hxx"

#include <format>
#include <variant>

#include "device_error.hxx"
#include "forward_target.hxx"
#include "geometry_arena.hxx"
#include "gpu_resource_table.hxx"
#include "image.hxx"
#include "image_storage.hxx"
#include "load_model.hxx"
#include "material_storage.hxx"
#include "pipeline.hxx"
#include "pipeline_graph_repository.hxx"
#include "pipeline_storage.hxx"
#include "renderer.hxx"
#include "sampler_storage.hxx"
#include "slang_compiler.hxx"
#include "slang_library.hxx"

// Phase 0 of the error-context redesign: these describe() overloads read the
// *current* struct layouts (a `.type` tag plus always-present nested error
// fields, only meaningful when `.type` matches). They are written now, ahead
// of any layout change, so every log call site can stop printing bare `.type`
// immediately. Once a given aggregate is migrated to `{type; optional<ErrorCause>
// cause;}` (see error_context.hxx), its describe() collapses to the
// `cause ? describe(*cause) : head` shape shown for ErrorCause below.

auto describe(ErrorContext const &context) -> std::string {
    std::string text = context.message.empty() ? std::string{"(no message)"} : std::string{context.message.view()};

    if (context.vk_result.has_value()) {
        text += std::format(" [VkResult={}]", static_cast<int>(*context.vk_result));
    }

    if (context.slang_result.has_value()) {
        text += std::format(" [SlangResult={}]", static_cast<int>(*context.slang_result));
    }

    if (!context.diagnostics.empty()) {
        text += "\n";
        text += context.diagnostics;
    }

    text += std::format(" ({}:{})", context.location.file_name(), context.location.line());

    return text;
}

auto describe(ErrorCause const &cause) -> std::string {
    return std::visit(
            [](auto const &alternative) -> std::string {
                using T = std::decay_t<decltype(alternative)>;

                if constexpr (std::is_same_v<T, ErrorContext>) {
                    return describe(alternative);
                } else {
                    return describe(*alternative);
                }
            },
            cause);
}

auto describe(DeviceError const &error) -> std::string {
    auto const message = error.message.empty() ? std::string_view{"(no message)"} : error.message.view();

    return std::format("DeviceError({}): {} [VkResult={}] ({}:{})", error.type, message,
                       static_cast<int>(error.vk_result), error.location.file_name(), error.location.line());
}

auto describe(ImageError const &error) -> std::string {
    auto head = std::format("ImageError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(ImageStorageError const &error) -> std::string {
    auto head = std::format("ImageStorageError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(GeometryArenaError const &error) -> std::string {
    auto head = std::format("GeometryArenaError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(MaterialStorageError const &error) -> std::string {
    auto head = std::format("MaterialStorageError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(PipelineError const &error) -> std::string {
    auto head = std::format("PipelineError({})", error.type);

    if (error.context.has_value()) {
        return head + " -> " + describe(*error.context);
    }

    return head;
}

auto describe(PipelineStorageError const &error) -> std::string {
    auto head = std::format("PipelineStorageError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(PipelineGraphError const &error) -> std::string {
    auto head = std::format("PipelineGraphError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(ForwardTargetError const &error) -> std::string {
    auto head = std::format("ForwardTargetError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(GpuResourceTableError const &error) -> std::string {
    auto head = std::format("GpuResourceTableError({})", error.type);

    if (error.context.has_value()) {
        return head + " -> " + describe(*error.context);
    }

    return head;
}

auto describe(SamplerStorageError const &error) -> std::string {
    auto head = std::format("SamplerStorageError({})", error.type);

    if (error.context.has_value()) {
        return head + " -> " + describe(*error.context);
    }

    return head;
}

auto describe(ModelLoadError const &error) -> std::string {
    auto head = std::format("ModelLoadError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(SlangLibraryError const &error) -> std::string {
    auto head = std::format("SlangLibraryError({})", error.type);

    if (error.result != SLANG_OK) {
        head += std::format(" [SlangResult={}]", static_cast<int>(error.result));
    }

    if (!error.diagnostics.empty()) {
        head += "\n";
        head += error.diagnostics;
    }

    return head;
}

auto describe(renderer::ShaderCompileError const &error) -> std::string {
    auto head = std::format("ShaderCompileError({})", error.type);

    if (error.result != SLANG_OK) {
        head += std::format(" [SlangResult={}]", static_cast<int>(error.result));
    }

    if (!error.diagnostics.empty()) {
        head += "\n";
        head += error.diagnostics;
    }

    return head;
}

auto describe(RendererError const &error) -> std::string {
    auto head = std::format("RendererError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
