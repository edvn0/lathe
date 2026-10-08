#pragma once

#include <string>

#include "core/error_context.hxx"

struct RendererError;

auto describe(ErrorContext const &context) -> std::string;
auto describe(ErrorCause const &cause) -> std::string;

#define X(T) auto describe(T const &error) -> std::string;
#define NX(ns, T) auto describe(ns::T const &error) -> std::string;
#include "core/error_types.def"
#undef X
#undef NX

auto describe(RendererError const &error) -> std::string;
