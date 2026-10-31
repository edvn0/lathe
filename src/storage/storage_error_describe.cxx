#include "core/error_describe.hxx"

#include <format>

#include "storage/cache_store.hxx"

auto describe(CacheStoreError const &error) -> std::string {
    auto head = error.detail.empty() ? std::format("CacheStoreError({})", error.type)
                                     : std::format("CacheStoreError({}: {})", error.type, error.detail);

    if (error.cause.has_value()) {
        return head + " -> " + describe(*error.cause);
    }

    return head;
}
