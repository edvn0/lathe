#pragma once

#include <string>

#include "error_context.hxx"

// Renders the full cause chain of an error to a human-readable string, e.g.
// "RendererError(model_load_error) -> ModelLoadError(geometry_upload_failed)
// -> GeometryArenaError(out_of_memory) -> DeviceError(BufferCreation):
// vkCreateBuffer failed (VK_ERROR_OUT_OF_DEVICE_MEMORY) [buffer.cxx:42]".
//
// One overload per aggregate/leaf error type, all implemented in
// error_describe.cxx (the one place that includes every error header, so
// each type is complete). Use these at every log call site instead of
// printing `.type` alone -- that drops everything a nested error captured
// (VkResult, Slang diagnostics, DeviceError's message/source_location).

struct RendererError;

auto describe(ErrorContext const &context) -> std::string;
auto describe(ErrorCause const &cause) -> std::string;

// Declarations generated from error_types.def -- add a new subsystem there,
// then implement its describe() overload in error_describe.cxx.
#define X(T) auto describe(T const &error) -> std::string;
#define NX(ns, T) auto describe(ns::T const &error) -> std::string;
#include "error_types.def"
#undef X
#undef NX

auto describe(RendererError const &error) -> std::string;
