// describe() overloads for error types owned by the assets module. Declared
// in error_describe.hxx (via error_types.def), defined here so this
// module's error headers stay private to it -- see error_describe.cxx's
// file comment.
#include "core/error_describe.hxx"

#include <format>

#include "assets/geometry_arena.hxx"
#include "assets/load_model.hxx"
#include "assets/material_storage.hxx"
#include "assets/slang_compiler.hxx"
#include "assets/slang_library.hxx"
#include "assets/texture_pipeline.hxx"

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

auto describe(ModelLoadError const &error) -> std::string {
    auto head = std::format("ModelLoadError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}

auto describe(TexturePipelineError const &error) -> std::string {
    auto head = std::format("TexturePipelineError({})", error.type);

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
