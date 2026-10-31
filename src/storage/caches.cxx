#include "storage/caches.hxx"

#include <chrono>
#include <format>
#include <mutex>
#include <system_error>

#include "core/error_describe.hxx"
#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"

namespace {
    // Where earlier versions kept their caches, which nothing reads now.
    auto legacy_paths() -> std::vector<std::filesystem::path> {
        auto const root = Paths::user_cache_directory();
        auto paths = std::vector<std::filesystem::path>{root / "shader_binaries", root.parent_path() / "ktx2"};

        std::error_code error;

        for (std::filesystem::directory_iterator it{root, error}, end; !error && it != end; it.increment(error)) {
            auto const name = it->path().filename().string();

            if (name.starts_with("shaders-") && name.ends_with(".lsp")) {
                paths.push_back(it->path());
            }
        }

        return paths;
    }

    std::mutex store_mutex;
    std::shared_ptr<CacheStore> current_store; // NOLINT: process-wide, guarded by store_mutex.
    bool attempted = false;                    // NOLINT: process-wide, guarded by store_mutex.
}

namespace caches {

    auto store() -> std::shared_ptr<CacheStore> {
        std::scoped_lock const lock{store_mutex};

        if (!attempted) {
            attempted = true;

            auto opened = CacheStore::open(Paths::user_cache_directory() / "store");

            if (opened) {
                current_store = *std::move(opened);
            } else {
                warn("Caching is off: {}", describe(opened.error()));
            }
        }

        return current_store;
    }

    auto set_store(std::shared_ptr<CacheStore> store) -> void {
        std::scoped_lock const lock{store_mutex};

        current_store = std::move(store);
        attempted = true;
    }

    auto maintain(std::uint64_t max_bytes) -> void {
        auto const store = caches::store();

        if (store == nullptr) {
            return;
        }

        auto stats = store->stats();

        if (stats) {
            for (auto const &entry: stats->namespaces) {
                if (entry.kind == CacheKind::file && entry.entries != 0) {
                    static_cast<void>(store->prune_missing_sources(entry.name));
                }
            }
        }

        static_cast<void>(store->remove_orphans());
        static_cast<void>(store->enforce_budget(max_bytes));

        std::error_code error;

        for (auto const &path: legacy_paths()) {
            if (std::filesystem::exists(path, error)) {
                info("'{}' belongs to an old cache layout and is no longer used; --clear-cache removes it",
                     path.string());
            }
        }
    }

    auto clear_all() -> CacheReclaimed {
        CacheReclaimed reclaimed;

        if (auto const store = caches::store()) {
            reclaimed = store->clear();
        }

        std::error_code error;

        for (auto const &path: legacy_paths()) {
            auto const count = std::filesystem::remove_all(path, error);
            reclaimed.entries += error ? 0 : count;
        }

        return reclaimed;
    }

    auto report() -> std::string {
        auto const store = caches::store();

        if (store == nullptr) {
            return "Caching is off.";
        }

        auto const stats = store->stats();

        if (!stats) {
            return describe(stats.error());
        }

        auto text = std::format("Cache store '{}': {} entries, {}\n", stats->directory.string(), stats->total_entries(),
                                human_readable_bytes(stats->total_bytes()));

        for (auto const &entry: stats->namespaces) {
            text += std::format("  {:<16} {:>6} entries  {:>10}  {}{}\n", entry.name, entry.entries,
                                human_readable_bytes(entry.bytes), entry.kind == CacheKind::file ? "files" : "inline",
                                entry.evictable ? "" : ", never evicted");
        }

        text += std::format("  {:<16} {:>6}          {:>10}\n", "index", "", human_readable_bytes(stats->database_bytes));

        return text;
    }

}
