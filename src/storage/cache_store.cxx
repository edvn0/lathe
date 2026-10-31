#include "storage/cache_store.hxx"

#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <format>
#include <map>
#include <set>
#include <system_error>
#include <utility>

#include "core/logger.hxx"
#include "core/sha256.hxx"

namespace {

    constexpr int schema_version = 1;
    // Process-wide, so two stores on one directory in one process (same PID) never pick the same temporary name.
    std::atomic<std::uint64_t> temporary_counter{0};
    constexpr auto busy_timeout = std::chrono::seconds{15};
    // A use only rewrites the last-use stamp when the old one is this old, so a burst of hits is one write, not many.
    constexpr std::int64_t touch_interval_seconds = 60;
    // Unreferenced files younger than this may belong to a commit that is still in flight.
    constexpr auto orphan_grace = std::chrono::minutes{10};

    [[nodiscard]] auto now_seconds() -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
    }

    [[nodiscard]] auto make_error(CacheStoreErrorType type, std::string detail) -> CacheStoreError {
        return CacheStoreError{.type = type, .detail = std::move(detail)};
    }

    [[nodiscard]] auto sanitise_label(std::string_view label) -> std::string {
        std::string out;

        for (auto const c: label.substr(0, 48)) {
            auto const keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
                              c == '_';
            out += keep ? c : '_';
        }

        return out.empty() ? std::string{"entry"} : out;
    }

    [[nodiscard]] auto file_size_or_zero(std::filesystem::path const &path) -> std::uint64_t {
        std::error_code error;
        auto const size = std::filesystem::file_size(path, error);

        return error ? 0 : size;
    }

    auto remove_quietly(std::filesystem::path const &path) -> void {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    // One prepared statement; binds and columns are 1-based and 0-based like SQLite's.
    class Statement {
    public:
        Statement(sqlite3 *db, char const *sql) {
            if (sqlite3_prepare_v2(db, sql, -1, &statement_, nullptr) != SQLITE_OK) {
                statement_ = nullptr;
            }
        }

        ~Statement() { sqlite3_finalize(statement_); }

        Statement(Statement const &) = delete;
        auto operator=(Statement const &) -> Statement & = delete;

        [[nodiscard]] auto valid() const noexcept -> bool { return statement_ != nullptr; }

        auto text(int index, std::string_view value) -> Statement & {
            sqlite3_bind_text(statement_, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
            return *this;
        }

        auto integer(int index, std::int64_t value) -> Statement & {
            sqlite3_bind_int64(statement_, index, value);
            return *this;
        }

        auto blob(int index, std::span<std::byte const> value) -> Statement & {
            sqlite3_bind_blob64(statement_, index, value.data(), value.size(), SQLITE_TRANSIENT);
            return *this;
        }

        auto null(int index) -> Statement & {
            sqlite3_bind_null(statement_, index);
            return *this;
        }

        // True while a row is available.
        [[nodiscard]] auto step() -> bool { return statement_ != nullptr && sqlite3_step(statement_) == SQLITE_ROW; }

        // Runs a statement that returns no rows.
        [[nodiscard]] auto run() -> bool {
            if (statement_ == nullptr) {
                return false;
            }

            auto const result = sqlite3_step(statement_);

            return result == SQLITE_DONE || result == SQLITE_ROW;
        }

        [[nodiscard]] auto column_text(int index) const -> std::string {
            auto const *text = sqlite3_column_text(statement_, index);

            return text == nullptr ? std::string{}
                                   : std::string{reinterpret_cast<char const *>(text),
                                                 static_cast<std::size_t>(sqlite3_column_bytes(statement_, index))};
        }

        [[nodiscard]] auto column_int(int index) const -> std::int64_t { return sqlite3_column_int64(statement_, index); }

        [[nodiscard]] auto column_is_null(int index) const -> bool {
            return sqlite3_column_type(statement_, index) == SQLITE_NULL;
        }

        [[nodiscard]] auto column_blob(int index) const -> std::vector<std::byte> {
            auto const *data = static_cast<std::byte const *>(sqlite3_column_blob(statement_, index));
            auto const size = static_cast<std::size_t>(sqlite3_column_bytes(statement_, index));

            return data == nullptr ? std::vector<std::byte>{} : std::vector<std::byte>{data, data + size};
        }

    private:
        sqlite3_stmt *statement_ = nullptr;
    };

    struct Row {
        std::string ns;
        std::string key;
        std::string file;
        std::string source;
        std::uint64_t size = 0;
    };

}

struct CacheStore::Impl {
    sqlite3 *db = nullptr;
    std::mutex mutex;
    std::filesystem::path blobs;
    std::filesystem::path database_path;
    std::map<std::string, CacheNamespaceSpec, std::less<>> namespaces;
    std::function<std::int64_t()> clock = now_seconds;

    [[nodiscard]] auto now() const -> std::int64_t { return clock(); }

    ~Impl() {
        if (db != nullptr) {
            sqlite3_close_v2(db);
        }
    }

    [[nodiscard]] auto exec(char const *sql) -> bool { return sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK; }

    // Only an unusable database is reported as open_failed (the one case open() rebuilds from scratch); a busy or
    // failing one must not make a process delete a database others are using.
    [[nodiscard]] auto last_error(std::string_view what) const -> CacheStoreError {
        auto const code = sqlite3_errcode(db) & 0xFF;
        auto const unusable = code == SQLITE_NOTADB || code == SQLITE_CORRUPT;

        return make_error(unusable ? CacheStoreErrorType::open_failed : CacheStoreErrorType::sql_failed,
                          std::format("{}: {}", what, sqlite3_errmsg(db)));
    }

    [[nodiscard]] auto find_namespace(std::string_view name) const -> CacheNamespaceSpec const * {
        auto const found = namespaces.find(name);

        return found == namespaces.end() ? nullptr : &found->second;
    }

    [[nodiscard]] auto require(std::string_view name, CacheKind kind) const
            -> std::expected<CacheNamespaceSpec const *, CacheStoreError> {
        auto const *spec = find_namespace(name);

        if (spec == nullptr) {
            return std::unexpected(make_error(CacheStoreErrorType::unknown_namespace, std::string{name}));
        }

        if (spec->kind != kind) {
            return std::unexpected(make_error(CacheStoreErrorType::wrong_kind, std::string{name}));
        }

        return spec;
    }

    // Read from the database, so namespaces this process has not declared (yet) count too. One this build has never
    // heard of is left alone.
    [[nodiscard]] auto is_evictable(std::string_view name) -> bool {
        Statement query{db, "SELECT evictable FROM namespaces WHERE name = ?1"};
        query.text(1, name);

        return query.step() && query.column_int(0) != 0;
    }

    auto touch(std::string_view ns, std::string_view key, std::int64_t last_used) -> void {
        auto const current = now();

        if (current - last_used < touch_interval_seconds) {
            return;
        }

        Statement update{db, "UPDATE entries SET last_used = ?3 WHERE ns = ?1 AND key = ?2"};
        update.text(1, ns).text(2, key).integer(3, current);
        static_cast<void>(update.run());
    }

    // Deletes the rows and then their files. The caller holds the mutex.
    auto delete_rows(std::vector<Row> const &rows) -> CacheReclaimed {
        CacheReclaimed reclaimed;

        for (auto const &row: rows) {
            Statement remove{db, "DELETE FROM entries WHERE ns = ?1 AND key = ?2"};
            remove.text(1, row.ns).text(2, row.key);

            if (!remove.run()) {
                continue;
            }

            if (!row.file.empty()) {
                remove_quietly(blobs / row.file);
            }

            ++reclaimed.entries;
            reclaimed.bytes += row.size;
        }

        return reclaimed;
    }

    [[nodiscard]] auto database_bytes() const -> std::uint64_t {
        return file_size_or_zero(database_path) + file_size_or_zero(database_path.string() + "-wal");
    }

    [[nodiscard]] auto init_schema() -> std::expected<void, CacheStoreError> {
        if (!exec("BEGIN IMMEDIATE")) {
            return std::unexpected(last_error("begin"));
        }

        // A statement still holding a row would make the COMMIT fail, so the reads below live in their own scope.
        auto const current = [this] {
            Statement version{db, "PRAGMA user_version"};

            return version.step() ? static_cast<int>(version.column_int(0)) : -1;
        }();

        if (current != 0 && current != schema_version) {
            static_cast<void>(exec("ROLLBACK"));

            return std::unexpected(make_error(CacheStoreErrorType::open_failed,
                                              std::format("database has schema version {}, expected {}", current,
                                                          schema_version)));
        }

        auto const created = exec("CREATE TABLE IF NOT EXISTS namespaces("
                                  " name TEXT PRIMARY KEY, version INTEGER NOT NULL, kind INTEGER NOT NULL,"
                                  " evictable INTEGER NOT NULL) WITHOUT ROWID") &&
                             exec("CREATE TABLE IF NOT EXISTS entries("
                                  " ns TEXT NOT NULL, key TEXT NOT NULL, size INTEGER NOT NULL,"
                                  " source TEXT NOT NULL DEFAULT '', file TEXT, data BLOB,"
                                  " created INTEGER NOT NULL, last_used INTEGER NOT NULL,"
                                  " PRIMARY KEY(ns, key)) WITHOUT ROWID") &&
                             exec("CREATE INDEX IF NOT EXISTS entries_last_used ON entries(last_used)") &&
                             exec(std::format("PRAGMA user_version = {}", schema_version).c_str());

        if (!created) {
            auto error = last_error("create schema");
            static_cast<void>(exec("ROLLBACK"));

            return std::unexpected(std::move(error));
        }

        if (!exec("COMMIT")) {
            return std::unexpected(last_error("commit"));
        }

        return {};
    }
};

auto CacheStats::total_entries() const noexcept -> std::uint64_t {
    std::uint64_t total = 0;

    for (auto const &entry: namespaces) {
        total += entry.entries;
    }

    return total;
}

auto CacheStats::total_bytes() const noexcept -> std::uint64_t {
    auto total = database_bytes;

    for (auto const &entry: namespaces) {
        total += entry.bytes;
    }

    return total;
}

PendingCacheFile::~PendingCacheFile() {
    if (!temporary_.empty()) {
        remove_quietly(temporary_);
    }
}

PendingCacheFile::PendingCacheFile(PendingCacheFile &&other) noexcept :
    namespace_name_(std::move(other.namespace_name_)), key_(std::move(other.key_)),
    temporary_(std::exchange(other.temporary_, {})), final_(std::move(other.final_)) {}

auto PendingCacheFile::operator=(PendingCacheFile &&other) noexcept -> PendingCacheFile & {
    if (this != &other) {
        if (!temporary_.empty()) {
            remove_quietly(temporary_);
        }

        namespace_name_ = std::move(other.namespace_name_);
        key_ = std::move(other.key_);
        temporary_ = std::exchange(other.temporary_, {});
        final_ = std::move(other.final_);
    }

    return *this;
}

CacheStore::~CacheStore() = default;

auto CacheStore::set_clock(std::function<std::int64_t()> clock) -> void {
    std::scoped_lock const lock{impl_->mutex};

    impl_->clock = std::move(clock);
}

auto CacheStore::open(std::filesystem::path const &directory)
        -> std::expected<std::shared_ptr<CacheStore>, CacheStoreError> {
    if (directory.empty()) {
        return std::unexpected(make_error(CacheStoreErrorType::invalid_argument, "empty directory"));
    }

    std::error_code error;
    std::filesystem::create_directories(directory / "blobs", error);

    if (error) {
        return std::unexpected(make_error(CacheStoreErrorType::io_failed,
                                          std::format("{}: {}", directory.string(), error.message())));
    }

    auto const attempt = [&]() -> std::expected<std::shared_ptr<CacheStore>, CacheStoreError> {
        std::shared_ptr<CacheStore> store{new CacheStore};
        store->directory_ = directory;
        store->impl_ = std::make_unique<Impl>();

        auto &impl = *store->impl_;
        impl.blobs = directory / "blobs";
        impl.database_path = directory / "cache.db";

        auto const result = sqlite3_open_v2(impl.database_path.c_str(), &impl.db,
                                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);

        if (result != SQLITE_OK) {
            auto const unusable = (result & 0xFF) == SQLITE_NOTADB || (result & 0xFF) == SQLITE_CORRUPT;

            return std::unexpected(make_error(unusable ? CacheStoreErrorType::open_failed : CacheStoreErrorType::sql_failed,
                                              std::format("{}: {}", impl.database_path.string(),
                                                          impl.db != nullptr ? sqlite3_errmsg(impl.db)
                                                                             : sqlite3_errstr(result))));
        }

        sqlite3_busy_timeout(impl.db, static_cast<int>(std::chrono::milliseconds{busy_timeout}.count()));

        // journal_mode reads back as a row, so it goes through a statement. WAL lets other processes keep reading
        // while one writes.
        {
            Statement journal{impl.db, "PRAGMA journal_mode = WAL"};
            static_cast<void>(journal.step());
        }

        if (!impl.exec("PRAGMA synchronous = NORMAL")) {
            return std::unexpected(impl.last_error("pragma"));
        }

        if (auto schema = impl.init_schema(); !schema) {
            return std::unexpected(std::move(schema.error()));
        }

        return store;
    };

    auto store = attempt();

    if (!store && store.error().type == CacheStoreErrorType::open_failed) {
        // Corrupt, not a database, or from a newer format: it is only a cache, so start over.
        warn("CacheStore: rebuilding '{}': {}", directory.string(), store.error().detail);

        for (auto const *suffix: {"cache.db", "cache.db-wal", "cache.db-shm"}) {
            remove_quietly(directory / suffix);
        }

        // The payloads are indexed by the database we just dropped.
        std::filesystem::remove_all(directory / "blobs", error);
        std::filesystem::create_directories(directory / "blobs", error);

        store = attempt();
    }

    return store;
}

auto CacheStore::declare(CacheNamespaceSpec const &spec) -> std::expected<void, CacheStoreError> {
    if (spec.name.empty()) {
        return std::unexpected(make_error(CacheStoreErrorType::invalid_argument, spec.name));
    }

    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    if (!impl.exec("BEGIN IMMEDIATE")) {
        return std::unexpected(impl.last_error("begin"));
    }

    auto const previous = [&]() -> std::optional<std::int64_t> {
        Statement stored{impl.db, "SELECT version FROM namespaces WHERE name = ?1"};
        stored.text(1, spec.name);

        return stored.step() ? std::optional{stored.column_int(0)} : std::nullopt;
    }();

    if (previous.has_value() && *previous != spec.version) {
        info("CacheStore: '{}' is version {}, was {}: dropping its entries", spec.name, spec.version, *previous);

        std::vector<Row> rows;
        {
            Statement list{impl.db, "SELECT key, file, size FROM entries WHERE ns = ?1"};
            list.text(1, spec.name);

            while (list.step()) {
                rows.push_back(Row{.ns = spec.name,
                                   .key = list.column_text(0),
                                   .file = list.column_text(1),
                                   .size = static_cast<std::uint64_t>(list.column_int(2))});
            }
        }

        static_cast<void>(impl.delete_rows(rows));
    }

    Statement upsert{impl.db, "INSERT INTO namespaces(name, version, kind, evictable) VALUES(?1, ?2, ?3, ?4)"
                              " ON CONFLICT(name) DO UPDATE SET version = excluded.version, kind = excluded.kind,"
                              " evictable = excluded.evictable"};
    upsert.text(1, spec.name)
            .integer(2, spec.version)
            .integer(3, std::to_underlying(spec.kind))
            .integer(4, spec.evictable ? 1 : 0);

    if (!upsert.run()) {
        auto error = impl.last_error("declare namespace");
        static_cast<void>(impl.exec("ROLLBACK"));

        return std::unexpected(std::move(error));
    }

    if (!impl.exec("COMMIT")) {
        return std::unexpected(impl.last_error("commit"));
    }

    impl.namespaces.insert_or_assign(spec.name, spec);

    return {};
}

auto CacheStore::get_blob(std::string_view ns, std::string_view key) -> std::optional<std::vector<std::byte>> {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    if (!impl.require(ns, CacheKind::inline_blob)) {
        return std::nullopt;
    }

    Statement find{impl.db, "SELECT data, last_used FROM entries WHERE ns = ?1 AND key = ?2 AND data IS NOT NULL"};
    find.text(1, ns).text(2, key);

    if (!find.step()) {
        return std::nullopt;
    }

    auto data = find.column_blob(0);
    impl.touch(ns, key, find.column_int(1));

    return data;
}

auto CacheStore::put_blob(std::string_view ns, std::string_view key, std::span<std::byte const> payload,
                          CacheEntryInfo const &info) -> std::expected<void, CacheStoreError> {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    if (auto spec = impl.require(ns, CacheKind::inline_blob); !spec) {
        return std::unexpected(std::move(spec.error()));
    }

    auto const now = impl.now();
    Statement put{impl.db, "INSERT INTO entries(ns, key, size, source, file, data, created, last_used)"
                           " VALUES(?1, ?2, ?3, ?4, NULL, ?5, ?6, ?6)"
                           " ON CONFLICT(ns, key) DO UPDATE SET size = excluded.size, source = excluded.source,"
                           " file = NULL, data = excluded.data, last_used = excluded.last_used"};
    put.text(1, ns)
            .text(2, key)
            .integer(3, static_cast<std::int64_t>(payload.size()))
            .text(4, info.source)
            .blob(5, payload)
            .integer(6, now);

    if (!put.run()) {
        return std::unexpected(impl.last_error("put"));
    }

    return {};
}

auto CacheStore::find_file(std::string_view ns, std::string_view key) -> std::optional<std::filesystem::path> {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    if (!impl.require(ns, CacheKind::file)) {
        return std::nullopt;
    }

    Statement find{impl.db, "SELECT file, last_used FROM entries WHERE ns = ?1 AND key = ?2 AND file IS NOT NULL"};
    find.text(1, ns).text(2, key);

    if (!find.step()) {
        return std::nullopt;
    }

    auto const path = impl.blobs / find.column_text(0);
    auto const last_used = find.column_int(1);

    std::error_code error;

    if (!std::filesystem::exists(path, error)) {
        // Deleted behind our back (a manual clean-up, another process evicting): forget the entry.
        Statement drop{impl.db, "DELETE FROM entries WHERE ns = ?1 AND key = ?2"};
        drop.text(1, ns).text(2, key);
        static_cast<void>(drop.run());

        return std::nullopt;
    }

    impl.touch(ns, key, last_used);

    return path;
}

auto CacheStore::begin_file(std::string_view ns, std::string_view key, std::string_view label,
                            std::string_view extension)
        -> std::expected<PendingCacheFile, CacheStoreError> {
    auto &impl = *impl_;

    std::filesystem::path final_path;
    {
        std::scoped_lock const lock{impl.mutex};

        auto spec = impl.require(ns, CacheKind::file);

        if (!spec) {
            return std::unexpected(std::move(spec.error()));
        }

        auto const digest = sha256_hex(std::as_bytes(std::span{key.data(), key.size()}));
        final_path = impl.blobs / std::string{ns} /
                     std::format("{}.{}{}", sanitise_label(label), std::string_view{digest}.substr(0, 16),
                                 extension.empty() ? std::string_view{(*spec)->extension} : extension);
    }

    std::error_code error;
    std::filesystem::create_directories(final_path.parent_path(), error);

    if (error) {
        return std::unexpected(make_error(CacheStoreErrorType::io_failed,
                                          std::format("{}: {}", final_path.parent_path().string(), error.message())));
    }

    PendingCacheFile pending;
    pending.namespace_name_ = std::string{ns};
    pending.key_ = std::string{key};
    pending.final_ = final_path;
    pending.temporary_ = std::filesystem::path{final_path.string() +
                                               std::format(".tmp-{}-{}", getpid(), temporary_counter.fetch_add(1))};

    return pending;
}

auto CacheStore::commit(PendingCacheFile &&pending, CacheEntryInfo const &info)
        -> std::expected<std::filesystem::path, CacheStoreError> {
    auto &impl = *impl_;
    auto const temporary = std::exchange(pending.temporary_, {});

    if (temporary.empty()) {
        return std::unexpected(make_error(CacheStoreErrorType::invalid_argument, "empty pending file"));
    }

    std::error_code error;
    auto const size = std::filesystem::file_size(temporary, error);

    if (error) {
        return std::unexpected(make_error(CacheStoreErrorType::io_failed,
                                          std::format("{}: {}", temporary.string(), error.message())));
    }

    std::filesystem::rename(temporary, pending.final_, error);

    if (error) {
        remove_quietly(temporary);

        return std::unexpected(make_error(CacheStoreErrorType::io_failed,
                                          std::format("{}: {}", pending.final_.string(), error.message())));
    }

    std::scoped_lock const lock{impl.mutex};

    auto const relative = pending.final_.lexically_relative(impl.blobs).generic_string();
    auto const now = impl.now();
    Statement put{impl.db, "INSERT INTO entries(ns, key, size, source, file, data, created, last_used)"
                           " VALUES(?1, ?2, ?3, ?4, ?5, NULL, ?6, ?6)"
                           " ON CONFLICT(ns, key) DO UPDATE SET size = excluded.size, source = excluded.source,"
                           " file = excluded.file, data = NULL, last_used = excluded.last_used"};
    put.text(1, pending.namespace_name_)
            .text(2, pending.key_)
            .integer(3, static_cast<std::int64_t>(size))
            .text(4, info.source)
            .text(5, relative)
            .integer(6, now);

    if (!put.run()) {
        // The file is in place but unindexed; remove_orphans() would get it, but a miss is also fine, so clean up now.
        remove_quietly(pending.final_);

        return std::unexpected(impl.last_error("commit"));
    }

    return pending.final_;
}

auto CacheStore::erase(std::string_view ns, std::string_view key) -> void {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    Statement find{impl.db, "SELECT file, size FROM entries WHERE ns = ?1 AND key = ?2"};
    find.text(1, ns).text(2, key);

    if (!find.step()) {
        return;
    }

    static_cast<void>(impl.delete_rows({Row{.ns = std::string{ns},
                                            .key = std::string{key},
                                            .file = find.column_text(0),
                                            .size = static_cast<std::uint64_t>(find.column_int(1))}}));
}

auto CacheStore::stats() -> std::expected<CacheStats, CacheStoreError> {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    CacheStats result;
    result.directory = directory_;
    result.database_bytes = impl.database_bytes();

    std::map<std::string, CacheNamespaceStats, std::less<>> by_name;

    Statement known{impl.db, "SELECT name, kind, evictable FROM namespaces"};

    while (known.step()) {
        auto name = known.column_text(0);

        by_name[name] = CacheNamespaceStats{.name = name,
                                            .kind = static_cast<CacheKind>(known.column_int(1)),
                                            .evictable = known.column_int(2) != 0};
    }

    Statement query{impl.db, "SELECT ns, COUNT(*), COALESCE(SUM(size), 0), MIN(last_used), MAX(last_used)"
                             " FROM entries GROUP BY ns"};

    while (query.step()) {
        auto name = query.column_text(0);
        auto &entry = by_name[name];

        entry.name = name;
        entry.entries = static_cast<std::uint64_t>(query.column_int(1));
        entry.bytes = static_cast<std::uint64_t>(query.column_int(2));
        entry.oldest_use = query.column_int(3);
        entry.newest_use = query.column_int(4);
    }

    for (auto &[name, entry]: by_name) {
        result.namespaces.push_back(std::move(entry));
    }

    return result;
}

auto CacheStore::prune_missing_sources(std::string_view ns) -> CacheReclaimed {
    auto &impl = *impl_;
    std::vector<Row> candidates;

    {
        std::scoped_lock const lock{impl.mutex};

        Statement list{impl.db, "SELECT key, source, COALESCE(file, ''), size FROM entries"
                                " WHERE ns = ?1 AND source != ''"};
        list.text(1, ns);

        while (list.step()) {
            candidates.push_back(Row{.ns = std::string{ns},
                                     .key = list.column_text(0),
                                     .file = list.column_text(2),
                                     .source = list.column_text(1),
                                     .size = static_cast<std::uint64_t>(list.column_int(3))});
        }
    }

    // Probing the file system happens outside the lock.
    std::vector<Row> gone;

    for (auto &row: candidates) {
        std::error_code error;

        if (!std::filesystem::exists(row.source, error) && !error) {
            gone.push_back(std::move(row));
        }
    }

    std::scoped_lock const lock{impl.mutex};

    return impl.delete_rows(gone);
}

auto CacheStore::enforce_budget(std::uint64_t max_bytes) -> CacheReclaimed {
    if (max_bytes == 0) {
        return {};
    }

    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    Statement total{impl.db, "SELECT COALESCE(SUM(size), 0) FROM entries"};
    // The budget is for the payloads; the index is small and does not shrink when rows are deleted.
    auto used = total.step() ? static_cast<std::uint64_t>(total.column_int(0)) : std::uint64_t{0};

    if (used <= max_bytes) {
        return {};
    }

    std::vector<Row> victims;
    Statement oldest{impl.db, "SELECT ns, key, COALESCE(file, ''), size FROM entries ORDER BY last_used ASC, created ASC"};

    while (used > max_bytes && oldest.step()) {
        auto row = Row{.ns = oldest.column_text(0),
                       .key = oldest.column_text(1),
                       .file = oldest.column_text(2),
                       .size = static_cast<std::uint64_t>(oldest.column_int(3))};

        if (!impl.is_evictable(row.ns)) {
            continue;
        }

        used -= std::min(used, row.size);
        victims.push_back(std::move(row));
    }

    auto const reclaimed = impl.delete_rows(victims);

    if (reclaimed.entries != 0) {
        info("CacheStore: evicted {} entries ({} bytes) to fit the budget", reclaimed.entries, reclaimed.bytes);
    }

    return reclaimed;
}

auto CacheStore::remove_orphans() -> CacheReclaimed {
    auto &impl = *impl_;
    std::set<std::string> referenced;

    {
        std::scoped_lock const lock{impl.mutex};

        Statement list{impl.db, "SELECT file FROM entries WHERE file IS NOT NULL"};

        while (list.step()) {
            referenced.insert(list.column_text(0));
        }
    }

    CacheReclaimed reclaimed;
    std::error_code error;
    auto const cutoff = std::filesystem::file_time_type::clock::now() - orphan_grace;

    for (std::filesystem::recursive_directory_iterator it{impl.blobs, error}, end; !error && it != end;
         it.increment(error)) {
        if (!it->is_regular_file(error)) {
            continue;
        }

        auto const relative = it->path().lexically_relative(impl.blobs).generic_string();

        if (referenced.contains(relative)) {
            continue;
        }

        std::error_code time_error;
        auto const modified = std::filesystem::last_write_time(it->path(), time_error);

        if (time_error || modified > cutoff) {
            continue;
        }

        auto const size = file_size_or_zero(it->path());

        if (std::filesystem::remove(it->path(), time_error)) {
            ++reclaimed.entries;
            reclaimed.bytes += size;
        }
    }

    if (reclaimed.entries != 0) {
        info("CacheStore: removed {} orphaned files ({} bytes)", reclaimed.entries, reclaimed.bytes);
    }

    return reclaimed;
}

auto CacheStore::clear(std::string_view ns) -> CacheReclaimed {
    auto &impl = *impl_;
    std::scoped_lock const lock{impl.mutex};

    std::vector<Row> rows;
    Statement list{impl.db, ns.empty() ? "SELECT ns, key, COALESCE(file, ''), size FROM entries"
                                       : "SELECT ns, key, COALESCE(file, ''), size FROM entries WHERE ns = ?1"};

    if (!ns.empty()) {
        list.text(1, ns);
    }

    while (list.step()) {
        rows.push_back(Row{.ns = list.column_text(0),
                           .key = list.column_text(1),
                           .file = list.column_text(2),
                           .size = static_cast<std::uint64_t>(list.column_int(3))});
    }

    auto const reclaimed = impl.delete_rows(rows);

    // Give the freed space back to the file system.
    static_cast<void>(impl.exec("PRAGMA wal_checkpoint(TRUNCATE)"));

    if (ns.empty()) {
        static_cast<void>(impl.exec("VACUUM"));
    }

    return reclaimed;
}
