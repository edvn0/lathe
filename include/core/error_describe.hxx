#pragma once

#include <string>

#include "core/error_context.hxx"

// Renders an error's full cause chain, e.g. "RendererError(model_load_error) ->
// ModelLoadError(geometry_upload_failed) -> GeometryArenaError(out_of_memory) -> DeviceError(BufferCreation):
// vkCreateBuffer failed (VK_ERROR_OUT_OF_DEVICE_MEMORY) [buffer.cxx:42]". Log with this rather than `.type`.
//
// Each overload is defined next to its error type.

struct RendererError;

auto describe(ErrorContext const &context) -> std::string;
auto describe(ErrorCause const &cause) -> std::string;

// Generated from error_types.def.
#define X(T) auto describe(T const &error) -> std::string;
#define NX(ns, T) auto describe(ns::T const &error) -> std::string;
#include "core/error_types.def"
#undef X
#undef NX

auto describe(RendererError const &error) -> std::string;
