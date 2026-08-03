#pragma once

#include <slang.h>
#include <volk.h>

#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <type_traits>
#include <variant>

#include "fly_string.hxx"

// Shared error leaf, generalizing what DeviceError already captured
// (message + VkResult + source_location) so every subsystem can produce the
// same shape of "why did this actually fail" context. `diagnostics` stays a
// plain std::string (not FlyString) because Slang compile logs are large and
// effectively unique -- FlyString's intern pool is never freed, so interning
// those would leak for the life of the process.
struct ErrorContext {
    FlyString message;
    std::optional<VkResult> vk_result;
    std::optional<SlangResult> slang_result;
    std::string diagnostics;
    std::source_location location = std::source_location::current();
};

// Forward declarations only -- ErrorCause below must be able to reference
// these while they are still incomplete. Their `describe()` overloads (see
// error_describe.hxx) are the only place these need to be complete, and that
// lives in a single .cxx that includes every error header. Generated from
// error_types.def -- add a new subsystem there, not here.
#define X(T) struct T;
#define NX(ns, T) \
    namespace ns { \
        struct T; \
    }
#include "error_types.def"
#undef X
#undef NX

// Copyable heap indirection so ErrorCause can hold currently-incomplete /
// mutually-recursive aggregate error types. shared_ptr (not unique_ptr) is
// required: std::expected<T, E> copies E by value throughout this codebase,
// and a unique_ptr member would silently make every *Error type move-only.
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

// The cause of a subsystem error: either a leaf ErrorContext, or a boxed
// aggregate error one layer down. Aggregates convert to this via their
// `.cause` field (see e.g. RendererError) rather than holding N
// always-present nested structs, only one of which was ever meaningful.
// Alternatives generated from error_types.def -- add a new subsystem there.
#define X(T) , Boxed<T>
#define NX(ns, T) , Boxed<ns::T>
using ErrorCause = std::variant<ErrorContext
#include "error_types.def"
        >;
#undef X
#undef NX
