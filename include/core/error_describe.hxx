#pragma once

#include <string>

#include "core/error_context.hxx"

// Renders the full cause chain of an error to a human-readable string, e.g.
// "RendererError(model_load_error) -> ModelLoadError(geometry_upload_failed)
// -> GeometryArenaError(out_of_memory) -> DeviceError(BufferCreation):
// vkCreateBuffer failed (VK_ERROR_OUT_OF_DEVICE_MEMORY) [buffer.cxx:42]".
//
// One overload per aggregate/leaf error type, declared here (so callers
// only need this one header) but each defined next to its own type in
// whichever module owns it -- e.g. describe(ImageError) lives in
// gpu_error_describe.cxx alongside image.hxx, not in a single file that
// would need to include every subsystem's error header. describe(ErrorContext)/
// describe(ErrorCause)/describe(RendererError) are the exception: they're
// generic/module-agnostic, and live in error_describe.cxx. Use these at
// every log call site instead of printing `.type` alone -- that drops
// everything a nested error captured (VkResult, Slang diagnostics,
// DeviceError's message/source_location).

struct RendererError;

auto describe(ErrorContext const &context) -> std::string;
auto describe(ErrorCause const &cause) -> std::string;

// Declarations generated from error_types.def -- add a new subsystem there,
// then implement its describe() overload next to that subsystem's error type.
#define X(T) auto describe(T const &error) -> std::string;
#define NX(ns, T) auto describe(ns::T const &error) -> std::string;
#include "core/error_types.def"
#undef X
#undef NX

auto describe(RendererError const &error) -> std::string;
