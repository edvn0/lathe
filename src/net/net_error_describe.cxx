#include "core/error_describe.hxx"

#include <format>

#include "net/http.hxx"

auto describe(HttpError const &error) -> std::string {
    auto head = error.status_code != 0 ? std::format("HttpError({}, status={})", error.type, error.status_code)
                                       : std::format("HttpError({})", error.type);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
