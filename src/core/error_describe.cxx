#include "core/error_describe.hxx"

#include <format>
#include <variant>

#include "core/renderer_error.hxx"

// Phase 0 of the error-context redesign: these describe() overloads read the
// *current* struct layouts (a `.type` tag plus always-present nested error
// fields, only meaningful when `.type` matches). They are written now, ahead
// of any layout change, so every log call site can stop printing bare `.type`
// immediately. Once a given aggregate is migrated to `{type; optional<ErrorCause>
// cause;}` (see error_context.hxx), its describe() collapses to the
// `cause ? describe(*cause) : head` shape shown for ErrorCause below.
//
// Every other describe() overload is defined next to its own error type
// (e.g. describe(ImageError) lives in gpu_error_describe.cxx) -- declared
// here (via error_types.def) but defined per-module, so no module needs to
// see every other module's error headers just to log an error. This file
// keeps only the describe() overloads for the two generic, module-agnostic
// error types (ErrorContext/ErrorCause) plus RendererError, which needs
// nothing beyond its own header and the generic ErrorCause describer.
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

auto describe(RendererError const &error) -> std::string {
    auto head = std::format("RendererError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
