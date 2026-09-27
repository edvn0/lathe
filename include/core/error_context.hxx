#pragma once

#include <slang.h>
#include <volk.h>

#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <type_traits>
#include <variant>

#include "core/fly_string.hxx"

// The leaf of every error: message, VkResult and source location. `diagnostics` is a std::string because
// Slang logs are large and unique, and FlyString's intern pool is never freed.
struct ErrorContext {
    FlyString message;
    std::optional<VkResult> vk_result;
    std::optional<SlangResult> slang_result;
    std::string diagnostics;
    std::source_location location = std::source_location::current();
};

// Forward declarations generated from error_types.def; add new subsystems there.
#define X(T) struct T;
#define NX(ns, T)                                                                                                      \
    namespace ns {                                                                                                     \
        struct T;                                                                                                      \
    }
#include "core/error_types.def"
#undef X
#undef NX

// Copyable indirection for incomplete, mutually recursive error types. shared_ptr because a unique_ptr would
// make every *Error move-only.
template<class T>
struct Boxed {
    std::shared_ptr<const T> ptr;

    Boxed() = default;

    Boxed(T value) : ptr(std::make_shared<const T>(std::move(value))) {}

    [[nodiscard]]
    auto operator*() const noexcept -> T const & {
        return *ptr;
    }

    [[nodiscard]]
    auto operator->() const noexcept -> T const * {
        return ptr.get();
    }
};

// An error's cause: a leaf ErrorContext or a boxed error one layer down. Generated from error_types.def.
#define X(T) , Boxed<T>
#define NX(ns, T) , Boxed<ns::T>
using ErrorCause = std::variant<ErrorContext
#include "core/error_types.def"
                                >;
#undef X
#undef NX
