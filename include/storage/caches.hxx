#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "storage/cache_store.hxx"

// The process-wide CacheStore every cache service uses, so one set of limits and one budget cover them all.
namespace caches {

    // The store in Paths::user_cache_directory()/store, opened on first use; null (after one warning) if it cannot be
    // opened, in which case the services behave as if everything missed and nothing is stored.
    [[nodiscard]] auto store() -> std::shared_ptr<CacheStore>;

    // Replaces the process-wide store (tests, or a tool that wants its own location).
    auto set_store(std::shared_ptr<CacheStore> store) -> void;

    // Start-up housekeeping: drops entries whose source file is gone, deletes files nothing refers to, and evicts the
    // least recently used entries until the store fits in 'max_bytes' (0: no limit). Logs what is left of the layouts
    // earlier versions used.
    auto maintain(std::uint64_t max_bytes) -> void;

    // Empties every cache, including the ones the budget never evicts, and deletes the layouts earlier versions used
    // (the old texture directory, loose shader binaries and packs).
    auto clear_all() -> CacheReclaimed;

    // A multi-line, human-readable summary of what the store holds.
    [[nodiscard]] auto report() -> std::string;

}
