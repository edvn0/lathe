#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"

enum class CacheStoreErrorType : std::uint8_t {
    invalid_argument,
    open_failed,
    sql_failed,
    io_failed,
    unknown_namespace,
    wrong_kind,
};

template<>
struct std::formatter<CacheStoreErrorType> : std::formatter<std::string_view> {
    auto format(CacheStoreErrorType type, std::format_context &context) const {
        auto const name = [type]() -> std::string_view {
            switch (type) {
                case CacheStoreErrorType::invalid_argument:
                    return "invalid_argument";
                case CacheStoreErrorType::open_failed:
                    return "open_failed";
                case CacheStoreErrorType::sql_failed:
                    return "sql_failed";
                case CacheStoreErrorType::io_failed:
                    return "io_failed";
                case CacheStoreErrorType::unknown_namespace:
                    return "unknown_namespace";
                case CacheStoreErrorType::wrong_kind:
                    return "wrong_kind";
            }

            return "unknown";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct CacheStoreError {
    CacheStoreErrorType type = CacheStoreErrorType::sql_failed;
    std::string detail;
    std::optional<ErrorCause> cause;
};

// How a namespace keeps its payloads.
enum class CacheKind : std::uint8_t {
    // Small payloads (up to a few hundred KiB) kept in the database itself: shader binaries.
    inline_blob,
    // Large payloads kept as files beside the database and handed out by path, because their readers (libktx, the shader
    // pack loader, the HTTP client) work on paths: textures, shader packs, downloads.
    file,
};

struct CacheNamespaceSpec {
    std::string name;
    // Bump when the payload format or what feeds the key changes; every entry written under an older version is dropped.
    std::uint32_t version = 1;
    CacheKind kind = CacheKind::inline_blob;
    // File namespaces: the extension the files get, with the dot (begin_file() can give one entry another).
    std::string extension;
    // False for data that is expensive or impossible to get again (downloads); the budget never evicts it.
    bool evictable = true;
};

// What an entry records beside its payload; all optional.
struct CacheEntryInfo {
    // The file this entry was derived from. prune_missing_sources() drops the entry once that file is gone.
    std::string source;
    // A readable stem for the payload file's name (file namespaces only).
    std::string label;
};

struct CacheNamespaceStats {
    std::string name;
    CacheKind kind = CacheKind::inline_blob;
    bool evictable = true;
    std::uint64_t entries = 0;
    std::uint64_t bytes = 0;
    // Last use, in seconds since the Unix epoch (to the minute: a use only updates it when the old stamp is a minute
    // old); 0 when empty.
    std::int64_t oldest_use = 0;
    std::int64_t newest_use = 0;
};

struct CacheStats {
    std::filesystem::path directory;
    std::uint64_t database_bytes = 0;
    std::vector<CacheNamespaceStats> namespaces;

    [[nodiscard]] auto total_entries() const noexcept -> std::uint64_t;
    [[nodiscard]] auto total_bytes() const noexcept -> std::uint64_t;
};

struct CacheReclaimed {
    std::uint64_t entries = 0;
    std::uint64_t bytes = 0;
};

// A file being produced for a file namespace. Write to path(), then CacheStore::commit() it; dropping it uncommitted
// deletes the partial file.
class PendingCacheFile {
public:
    PendingCacheFile() = default;
    ~PendingCacheFile();

    PendingCacheFile(PendingCacheFile const &) = delete;
    auto operator=(PendingCacheFile const &) -> PendingCacheFile & = delete;
    PendingCacheFile(PendingCacheFile &&other) noexcept;
    auto operator=(PendingCacheFile &&other) noexcept -> PendingCacheFile &;

    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return temporary_; }
    [[nodiscard]] explicit operator bool() const noexcept { return !temporary_.empty(); }

private:
    friend class CacheStore;

    std::string namespace_name_;
    std::string key_;
    std::filesystem::path temporary_;
    std::filesystem::path final_;
};

// Held while one process or thread produces the entry for a key, so the others wait for it instead of repeating the
// work; see CacheStore::lock(). Released when destroyed, or when the process dies.
class CacheLock {
public:
    CacheLock() = default;
    ~CacheLock();

    CacheLock(CacheLock const &) = delete;
    auto operator=(CacheLock const &) -> CacheLock & = delete;
    CacheLock(CacheLock &&other) noexcept;
    auto operator=(CacheLock &&other) noexcept -> CacheLock &;

    // False when the wait was given up on, in which case nothing is held and the caller just does the work.
    [[nodiscard]] explicit operator bool() const noexcept { return descriptor_ >= 0; }

private:
    friend class CacheStore;

    explicit CacheLock(int descriptor) : descriptor_(descriptor) {}

    int descriptor_ = -1;
};

// An on-disk cache shared by every Lathe process (editor, player, packager, benchmark) of a user: a SQLite database
// indexing the entries (key, version, size, source, last use) plus a blobs/ directory for the large payloads. Keys are
// whole strings, so there are no hash collisions to guard against. Entries carry their namespace's version, are evicted
// least-recently-used within a size budget, and can be pruned once their source file is gone. It is only a cache: a
// database that is corrupt or from a newer format is thrown away and rebuilt.
//
// Safe to share between threads and processes (WAL, busy timeout, atomic renames). Engine code should not use it
// directly but through the services next to what is cached (TextureCache, ShaderBinaryCache, ...).
class CacheStore {
public:
    ~CacheStore();

    CacheStore(CacheStore const &) = delete;
    auto operator=(CacheStore const &) -> CacheStore & = delete;
    CacheStore(CacheStore &&) = delete;
    auto operator=(CacheStore &&) -> CacheStore & = delete;

    // Opens (creating if needed) the store in 'directory'.
    [[nodiscard]] static auto open(std::filesystem::path const &directory)
            -> std::expected<std::shared_ptr<CacheStore>, CacheStoreError>;

    // Registers a namespace; call before using it. A namespace whose stored version differs has its entries dropped.
    [[nodiscard]] auto declare(CacheNamespaceSpec const &spec) -> std::expected<void, CacheStoreError>;

    // Replaces the clock the last-use stamps come from (seconds since the Unix epoch), so tests can age entries.
    auto set_clock(std::function<std::int64_t()> clock) -> void;

    [[nodiscard]] auto directory() const noexcept -> std::filesystem::path const & { return directory_; }

    // Inline namespaces.
    [[nodiscard]] auto get_blob(std::string_view ns, std::string_view key) -> std::optional<std::vector<std::byte>>;
    [[nodiscard]] auto put_blob(std::string_view ns, std::string_view key, std::span<std::byte const> payload,
                                CacheEntryInfo const &info = {}) -> std::expected<void, CacheStoreError>;

    // File namespaces. find_file() returns the payload's path if the entry and its file are intact.
    [[nodiscard]] auto find_file(std::string_view ns, std::string_view key) -> std::optional<std::filesystem::path>;
    [[nodiscard]] auto begin_file(std::string_view ns, std::string_view key, std::string_view label = {},
                                  std::string_view extension = {})
            -> std::expected<PendingCacheFile, CacheStoreError>;
    // Atomically installs the written file and records the entry; returns its final path.
    [[nodiscard]] auto commit(PendingCacheFile &&pending, CacheEntryInfo const &info = {})
            -> std::expected<std::filesystem::path, CacheStoreError>;

    // For a file whose writer installs it atomically itself (the HTTP client's downloads): records the entry, so it is
    // counted, listed and cleared, and returns the stable path to write to. The entry's size is read from the file when
    // it exists (on find_file() and stats()).
    [[nodiscard]] auto reserve_file(std::string_view ns, std::string_view key, std::string_view label = {},
                                    std::string_view extension = {})
            -> std::expected<std::filesystem::path, CacheStoreError>;

    // Waits until nobody else holds the lock for 'key', then takes it. Callers that missed take it, look again, and only
    // then produce the entry, so two processes cooking the same texture do it once. Gives up (returning an empty lock)
    // after 'timeout' or once 'give_up' returns true.
    [[nodiscard]] auto lock(std::string_view ns, std::string_view key, std::function<bool()> const &give_up = {},
                            std::chrono::seconds timeout = std::chrono::minutes{10}) -> CacheLock;

    auto erase(std::string_view ns, std::string_view key) -> void;

    [[nodiscard]] auto stats() -> std::expected<CacheStats, CacheStoreError>;

    // Drops entries whose source file no longer exists.
    auto prune_missing_sources(std::string_view ns) -> CacheReclaimed;

    // Deletes least-recently-used entries of evictable namespaces until the payloads fit in 'max_bytes' (the index
    // is small and not counted). 0 means unlimited.
    auto enforce_budget(std::uint64_t max_bytes) -> CacheReclaimed;

    // Deletes payload files nothing refers to (left by crashes) and stale temporary files.
    auto remove_orphans() -> CacheReclaimed;

    // Empties one namespace, or all of them when 'ns' is empty (including non-evictable ones).
    auto clear(std::string_view ns = {}) -> CacheReclaimed;

private:
    CacheStore() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::filesystem::path directory_;
};
