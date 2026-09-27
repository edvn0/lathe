#include "core/error_describe.hxx"

#include <format>
#include <variant>

#include "core/renderer_error.hxx"

// The generic describe() overloads and RendererError's. Every other error type's lives next to it.
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
