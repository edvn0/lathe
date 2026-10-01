// describe() overload for the serialisation module's error type.
#include "core/error_describe.hxx"

#include <format>

#include "serialisation/lbf_error.hxx"

auto describe(LbfError const &error) -> std::string {
    auto head = std::format("LbfError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
