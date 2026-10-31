#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "assets/shader_pack_cache.hxx"
#include "core/error_describe.hxx"
#include "assets/texture_cache.hxx"
#include "gpu/shader_binary_cache.hxx"
#include "gpu/shader_stage.hxx"
#include "storage/cache_store.hxx"
#include "storage/download_cache.hxx"

namespace {

    auto make_temp_dir(std::string_view label) -> std::filesystem::path {
        auto const dir = std::filesystem::temp_directory_path() /
                         std::format("cache_store_test_{}_{}", label,
                                     std::chrono::steady_clock::now().time_since_epoch().count());

        std::filesystem::create_directories(dir);

        return dir;
    }

    auto bytes_of(std::string_view text) -> std::vector<std::byte> {
        auto const *begin = reinterpret_cast<std::byte const *>(text.data());

        return {begin, begin + text.size()};
    }

    auto text_of(std::vector<std::byte> const &bytes) -> std::string {
        return {reinterpret_cast<char const *>(bytes.data()), bytes.size()};
    }

    auto open_store(std::filesystem::path const &dir) -> std::shared_ptr<CacheStore> {
        auto store = CacheStore::open(dir);

        if (!store) {
            FAIL(describe(store.error()));
        }

        return *store;
    }

    auto declare_inline(CacheStore &store, std::string name = "blobs", std::uint32_t version = 1, bool evictable = true)
            -> void {
        REQUIRE(store.declare({.name = std::move(name), .version = version, .kind = CacheKind::inline_blob,
                               .evictable = evictable})
                        .has_value());
    }

    auto declare_files(CacheStore &store, std::string name = "files", std::uint32_t version = 1,
                       bool evictable = true) -> void {
        REQUIRE(store.declare({.name = std::move(name), .version = version, .kind = CacheKind::file,
                               .extension = ".bin", .evictable = evictable})
                        .has_value());
    }

    auto write_file(std::filesystem::path const &path, std::string_view content) -> void {
        std::ofstream{path, std::ios::binary} << content;
    }

    // Writes 'content' as a committed entry and returns its path.
    auto put_file(CacheStore &store, std::string_view ns, std::string_view key, std::string_view content,
                  std::string_view source = {}) -> std::filesystem::path {
        auto pending = store.begin_file(ns, key, "entry");
        REQUIRE(pending.has_value());
        write_file(pending->path(), content);

        auto committed = store.commit(std::move(*pending), {.source = std::string{source}});
        REQUIRE(committed.has_value());

        return *committed;
    }

}

TEST_SUITE("unit") {

    TEST_CASE("CacheStore: an inline entry round-trips, overwrites and survives a reopen") {
        auto const dir = make_temp_dir("inline");

        {
            auto store = open_store(dir);
            declare_inline(*store);

            CHECK_FALSE(store->get_blob("blobs", "a").has_value());

            auto const first = bytes_of("first");
            REQUIRE(store->put_blob("blobs", "a", first).has_value());
            CHECK(text_of(*store->get_blob("blobs", "a")) == "first");

            auto const second = bytes_of("second");
            REQUIRE(store->put_blob("blobs", "a", second).has_value());
            CHECK(text_of(*store->get_blob("blobs", "a")) == "second");
        }

        auto store = open_store(dir);
        declare_inline(*store);
        CHECK(text_of(*store->get_blob("blobs", "a")) == "second");

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: the key is the whole string, not a hash of it") {
        auto const dir = make_temp_dir("keys");
        auto store = open_store(dir);
        declare_inline(*store);

        auto const one = bytes_of("one");
        auto const two = bytes_of("two");
        REQUIRE(store->put_blob("blobs", "k|1", one).has_value());
        REQUIRE(store->put_blob("blobs", "k|2", two).has_value());

        CHECK(text_of(*store->get_blob("blobs", "k|1")) == "one");
        CHECK(text_of(*store->get_blob("blobs", "k|2")) == "two");
        CHECK_FALSE(store->get_blob("blobs", "k|").has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: using a namespace that was not declared, or the wrong kind, fails") {
        auto const dir = make_temp_dir("kinds");
        auto store = open_store(dir);
        declare_inline(*store);

        auto const payload = bytes_of("x");
        auto const undeclared = store->put_blob("nope", "a", payload);
        REQUIRE_FALSE(undeclared.has_value());
        CHECK(undeclared.error().type == CacheStoreErrorType::unknown_namespace);

        CHECK_FALSE(store->begin_file("blobs", "a").has_value());
        CHECK_FALSE(store->find_file("blobs", "a").has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: a new namespace version drops the old entries and their files") {
        auto const dir = make_temp_dir("version");
        std::filesystem::path file;

        {
            auto store = open_store(dir);
            declare_inline(*store, "blobs", 1);
            declare_files(*store, "files", 1);

            auto const payload = bytes_of("old");
            REQUIRE(store->put_blob("blobs", "a", payload).has_value());
            file = put_file(*store, "files", "f", "old");
            REQUIRE(std::filesystem::exists(file));
        }

        auto store = open_store(dir);
        declare_inline(*store, "blobs", 2);
        declare_files(*store, "files", 1);

        CHECK_FALSE(store->get_blob("blobs", "a").has_value());
        CHECK(store->find_file("files", "f").has_value());

        declare_files(*store, "files", 2);

        CHECK_FALSE(store->find_file("files", "f").has_value());
        CHECK_FALSE(std::filesystem::exists(file));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: a committed file is found by key and keeps its content") {
        auto const dir = make_temp_dir("file");
        auto store = open_store(dir);
        declare_files(*store);

        CHECK_FALSE(store->find_file("files", "k").has_value());

        auto const path = put_file(*store, "files", "k", "payload");

        REQUIRE(store->find_file("files", "k").has_value());
        CHECK(*store->find_file("files", "k") == path);
        CHECK(path.extension() == ".bin");
        CHECK(std::filesystem::file_size(path) == 7);

        // Committing the same key again replaces the content in place.
        auto const replaced = put_file(*store, "files", "k", "other payload");
        CHECK(replaced == path);
        CHECK(std::filesystem::file_size(path) == 13);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: an entry can choose its own file extension") {
        auto const dir = make_temp_dir("extension");
        auto store = open_store(dir);
        declare_files(*store);

        auto pending = store->begin_file("files", "model", "snake", ".glb");
        REQUIRE(pending.has_value());
        write_file(pending->path(), "glb");
        auto const path = store->commit(std::move(*pending));

        REQUIRE(path.has_value());
        CHECK(path->extension() == ".glb");
        CHECK(path->filename().string().starts_with("snake."));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: a pending file that is dropped leaves nothing behind") {
        auto const dir = make_temp_dir("pending");
        auto store = open_store(dir);
        declare_files(*store);

        std::filesystem::path temporary;
        {
            auto pending = store->begin_file("files", "k", "entry");
            REQUIRE(pending.has_value());
            temporary = pending->path();
            write_file(temporary, "half");
            CHECK(std::filesystem::exists(temporary));
        }

        CHECK_FALSE(std::filesystem::exists(temporary));
        CHECK_FALSE(store->find_file("files", "k").has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: an entry whose file vanished is a miss and is forgotten") {
        auto const dir = make_temp_dir("vanished");
        auto store = open_store(dir);
        declare_files(*store);

        auto const path = put_file(*store, "files", "k", "payload");
        std::filesystem::remove(path);

        CHECK_FALSE(store->find_file("files", "k").has_value());

        auto const stats = store->stats();
        REQUIRE(stats.has_value());
        CHECK(stats->total_entries() == 0);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: pruning drops entries whose source file is gone") {
        auto const dir = make_temp_dir("prune");
        auto store = open_store(dir);
        declare_files(*store);

        auto const source = dir / "source.png";
        write_file(source, "pixels");

        auto const kept = put_file(*store, "files", "has-source", "a", source.string());
        auto const dropped_source = dir / "deleted.png";
        write_file(dropped_source, "pixels");
        auto const dropped = put_file(*store, "files", "lost-source", "b", dropped_source.string());
        auto const sourceless = put_file(*store, "files", "no-source", "c");

        std::filesystem::remove(dropped_source);

        auto const reclaimed = store->prune_missing_sources("files");

        CHECK(reclaimed.entries == 1);
        CHECK(std::filesystem::exists(kept));
        CHECK_FALSE(std::filesystem::exists(dropped));
        CHECK(std::filesystem::exists(sourceless));
        CHECK(store->find_file("files", "has-source").has_value());
        CHECK_FALSE(store->find_file("files", "lost-source").has_value());
        CHECK(store->find_file("files", "no-source").has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: the budget evicts the least recently used entries first") {
        auto const dir = make_temp_dir("budget");
        auto store = open_store(dir);
        declare_files(*store);

        std::int64_t now = 1'000'000;
        store->set_clock([&now] { return now; });

        auto const first = put_file(*store, "files", "oldest", std::string(1000, 'a'));
        now += 1000;
        auto const second = put_file(*store, "files", "middle", std::string(1000, 'b'));
        now += 1000;
        auto const third = put_file(*store, "files", "newest", std::string(1000, 'c'));

        // Using the oldest entry makes it the most recent.
        now += 1000;
        REQUIRE(store->find_file("files", "oldest").has_value());

        auto const stats_before = store->stats();
        auto const used = stats_before->total_bytes() - stats_before->database_bytes;
        auto const reclaimed = store->enforce_budget(used - 500);

        CHECK(reclaimed.entries == 1);
        CHECK(reclaimed.bytes == 1000);
        CHECK(std::filesystem::exists(first));
        CHECK_FALSE(std::filesystem::exists(second));
        CHECK(std::filesystem::exists(third));

        CHECK(store->enforce_budget(0).entries == 0);
        CHECK(store->enforce_budget(1'000'000).entries == 0);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: eviction counts exactly the entries it removes, across namespaces") {
        auto const dir = make_temp_dir("counting");
        auto store = open_store(dir);
        declare_inline(*store);
        declare_files(*store);

        std::int64_t now = 1'000'000;
        store->set_clock([&now] { return now; });

        auto const payload = bytes_of(std::string(100, 'b'));

        // Many entries share a timestamp, as a burst of shader binaries does.
        for (int index = 0; index < 60; ++index) {
            REQUIRE(store->put_blob("blobs", std::format("blob-{}", index), payload).has_value());
        }

        now += 500;

        for (int index = 0; index < 20; ++index) {
            static_cast<void>(put_file(*store, "files", std::format("file-{}", index), std::string(1000, 'f')));
        }

        auto const before = store->stats();
        REQUIRE(before.has_value());
        CHECK(before->total_entries() == 80);

        auto const target = before->total_bytes() - before->database_bytes - 12'000;
        auto const reclaimed = store->enforce_budget(target);

        auto const after = store->stats();
        REQUIRE(after.has_value());
        CHECK(reclaimed.entries == before->total_entries() - after->total_entries());
        CHECK(after->total_bytes() - after->database_bytes <= target);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: the budget never evicts a namespace that opted out") {
        auto const dir = make_temp_dir("pinned");
        auto store = open_store(dir);
        declare_files(*store, "precious", 1, false);
        declare_files(*store, "files");

        std::int64_t now = 1'000'000;
        store->set_clock([&now] { return now; });

        auto const precious = put_file(*store, "precious", "p", std::string(5000, 'p'));
        now += 1000;
        auto const disposable = put_file(*store, "files", "d", std::string(1000, 'd'));

        static_cast<void>(store->enforce_budget(1));

        CHECK(std::filesystem::exists(precious));
        CHECK_FALSE(std::filesystem::exists(disposable));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: the budget applies to namespaces declared by another process") {
        auto const dir = make_temp_dir("foreign");
        std::filesystem::path file;

        {
            auto writer = open_store(dir);
            declare_files(*writer, "elsewhere");
            file = put_file(*writer, "elsewhere", "k", std::string(2000, 'x'));
        }

        // This one never declared the namespace, as a start-up pass before any service runs wouldn't have.
        auto store = open_store(dir);
        CHECK(store->enforce_budget(1).entries == 1);
        CHECK_FALSE(std::filesystem::exists(file));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: orphaned files are removed once old enough") {
        auto const dir = make_temp_dir("orphans");
        auto store = open_store(dir);
        declare_files(*store);

        auto const kept = put_file(*store, "files", "k", "kept");
        auto const stale = kept.parent_path() / "stale.bin.tmp-1-1";
        auto const recent = kept.parent_path() / "recent.bin.tmp-1-2";
        write_file(stale, "partial");
        write_file(recent, "partial");
        std::filesystem::last_write_time(stale, std::filesystem::file_time_type::clock::now() - std::chrono::hours{2});

        auto const reclaimed = store->remove_orphans();

        CHECK(reclaimed.entries == 1);
        CHECK_FALSE(std::filesystem::exists(stale));
        CHECK(std::filesystem::exists(recent));
        CHECK(std::filesystem::exists(kept));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: stats count entries and bytes per namespace, and clear empties them") {
        auto const dir = make_temp_dir("stats");
        auto store = open_store(dir);
        declare_inline(*store);
        declare_files(*store);

        auto const payload = bytes_of("12345");
        REQUIRE(store->put_blob("blobs", "a", payload).has_value());
        REQUIRE(store->put_blob("blobs", "b", payload).has_value());
        auto const file = put_file(*store, "files", "f", "1234567890");

        auto stats = store->stats();
        REQUIRE(stats.has_value());
        REQUIRE(stats->namespaces.size() == 2);

        for (auto const &entry: stats->namespaces) {
            if (entry.name == "blobs") {
                CHECK(entry.entries == 2);
                CHECK(entry.bytes == 10);
                CHECK(entry.kind == CacheKind::inline_blob);
            } else {
                CHECK(entry.name == "files");
                CHECK(entry.entries == 1);
                CHECK(entry.bytes == 10);
                CHECK(entry.kind == CacheKind::file);
            }
        }

        CHECK(store->clear("blobs").entries == 2);
        CHECK(std::filesystem::exists(file));

        CHECK(store->clear().entries == 1);
        CHECK_FALSE(std::filesystem::exists(file));
        CHECK(store->stats()->total_entries() == 0);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: a database that is not one is replaced, not fatal") {
        auto const dir = make_temp_dir("corrupt");

        {
            auto store = open_store(dir);
            declare_inline(*store);
            auto const payload = bytes_of("x");
            REQUIRE(store->put_blob("blobs", "a", payload).has_value());
        }

        write_file(dir / "cache.db", std::string(4096, 'z'));
        std::filesystem::remove(dir / "cache.db-wal");
        std::filesystem::remove(dir / "cache.db-shm");

        auto store = open_store(dir);
        declare_inline(*store);
        CHECK_FALSE(store->get_blob("blobs", "a").has_value());

        auto const payload = bytes_of("y");
        CHECK(store->put_blob("blobs", "a", payload).has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: many threads and independent connections can write the same store at once") {
        auto const dir = make_temp_dir("concurrent");
        constexpr int connections = 3;
        constexpr int per_connection = 40;

        std::vector<std::jthread> workers;
        std::atomic<int> failures{0};
        std::mutex first_failure_mutex;
        std::string first_failure;
        auto const note = [&](std::string what) {
            std::scoped_lock const lock{first_failure_mutex};

            if (first_failure.empty()) {
                first_failure = std::move(what);
            }
        };

        for (int connection = 0; connection < connections; ++connection) {
            workers.emplace_back([&, connection] {
                auto store = CacheStore::open(dir);

                if (!store) {
                    note(describe(store.error()));
                    ++failures;
                    return;
                }

                auto const blobs = (*store)->declare({.name = "blobs", .kind = CacheKind::inline_blob});
                auto const files = (*store)->declare({.name = "files", .kind = CacheKind::file, .extension = ".bin"});

                if (!blobs || !files) {
                    note(describe(blobs ? files.error() : blobs.error()));
                    ++failures;
                    return;
                }

                for (int index = 0; index < per_connection; ++index) {
                    // Half the keys are shared between connections, as two processes cooking the same texture.
                    auto const key = index % 2 == 0 ? std::format("shared-{}", index)
                                                    : std::format("own-{}-{}", connection, index);
                    auto const payload = bytes_of(key);

                    if (auto const put = (*store)->put_blob("blobs", key, payload); !put) {
                        note(describe(put.error()));
                        ++failures;
                    }

                    auto pending = (*store)->begin_file("files", key, "entry");

                    if (!pending) {
                        ++failures;
                        continue;
                    }

                    write_file(pending->path(), key);

                    if (auto const committed = (*store)->commit(std::move(*pending)); !committed) {
                        note(describe(committed.error()));
                        ++failures;
                    }
                }
            });
        }

        workers.clear();
        CHECK_MESSAGE(failures.load() == 0, first_failure);

        auto store = open_store(dir);
        declare_inline(*store);
        declare_files(*store);

        for (int index = 0; index < per_connection; ++index) {
            for (int connection = 0; connection < connections; ++connection) {
                auto const key = index % 2 == 0 ? std::format("shared-{}", index)
                                                : std::format("own-{}-{}", connection, index);
                auto const blob = store->get_blob("blobs", key);
                REQUIRE(blob.has_value());
                CHECK(text_of(*blob) == key);

                auto const path = store->find_file("files", key);
                REQUIRE(path.has_value());
                CHECK(std::filesystem::file_size(*path) == key.size());
            }
        }

        CHECK(store->remove_orphans().entries == 0);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("ShaderBinaryCache: a binary is only found for the same driver, stage and SPIR-V") {
        auto const dir = make_temp_dir("shader_binaries");
        auto store = open_store(dir);

        std::array<std::uint8_t, VK_UUID_SIZE> driver_a{};
        driver_a.fill(0xAA);
        std::array<std::uint8_t, VK_UUID_SIZE> driver_b{};
        driver_b.fill(0xBB);

        auto cache_a = ShaderBinaryCache::create(store, driver_a, 1);
        auto cache_b = ShaderBinaryCache::create(store, driver_b, 1);
        auto cache_a_v2 = ShaderBinaryCache::create(store, driver_a, 2);
        REQUIRE(cache_a.has_value());
        REQUIRE(cache_b.has_value());
        REQUIRE(cache_a_v2.has_value());

        std::array<std::uint32_t, 4> spirv{1, 2, 3, 4};
        std::array<std::uint32_t, 4> edited{1, 2, 3, 5};
        auto const binary = bytes_of("compiled");

        CHECK_FALSE(cache_a->find("lit.slang|main|0", spirv).has_value());

        cache_a->store("lit.slang|main|0", spirv, binary);

        REQUIRE(cache_a->find("lit.slang|main|0", spirv).has_value());
        CHECK(text_of(*cache_a->find("lit.slang|main|0", spirv)) == "compiled");

        // Another stage, an edited shader, another driver or a newer driver version each start from nothing.
        CHECK_FALSE(cache_a->find("lit.slang|main|1", spirv).has_value());
        CHECK_FALSE(cache_a->find("lit.slang|main|0", edited).has_value());
        CHECK_FALSE(cache_b->find("lit.slang|main|0", spirv).has_value());
        CHECK_FALSE(cache_a_v2->find("lit.slang|main|0", spirv).has_value());

        cache_a->discard("lit.slang|main|0", spirv);
        CHECK_FALSE(cache_a->find("lit.slang|main|0", spirv).has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("ShaderStageInfo: the cache key outlives the string it was built from") {
        // The pipeline repository builds the key on the spot; a view would dangle and key the binary by garbage.
        auto const make = [](int stage) {
            return ShaderStageInfo{.cache_key = std::format("lit.slang|main|{}", stage)};
        };

        auto const info = make(7);
        std::string const scratch(64, 'x');

        CHECK(info.cache_key == "lit.slang|main|7");
        CHECK_FALSE(scratch.empty());
    }

    TEST_CASE("ShaderBinaryCache: needs a store") {
        std::array<std::uint8_t, VK_UUID_SIZE> driver{};

        CHECK_FALSE(ShaderBinaryCache::create(nullptr, driver, 1).has_value());
    }

    TEST_CASE("TextureCache: files are found by key and pruned with their source") {
        auto const dir = make_temp_dir("textures");
        TextureCache cache{dir};
        REQUIRE(cache.enabled());

        auto const source = dir / "albedo.png";
        write_file(source, "png");

        CHECK_FALSE(cache.find("albedo|0").has_value());

        auto pending = cache.begin("albedo|0", "albedo");
        REQUIRE(pending.has_value());
        write_file(pending->path(), "ktx2");
        REQUIRE(cache.commit(std::move(*pending), source.string()));

        auto const found = cache.find("albedo|0");
        REQUIRE(found.has_value());
        CHECK(found->extension() == ".ktx2");

        CHECK(cache.prune_missing_sources().entries == 0);
        std::filesystem::remove(source);
        CHECK(cache.prune_missing_sources().entries == 1);
        CHECK_FALSE(cache.find("albedo|0").has_value());

        cache.discard("never-existed");

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("TextureCache: without a store everything misses and nothing is kept") {
        TextureCache cache;

        CHECK_FALSE(cache.enabled());
        CHECK_FALSE(cache.find("k").has_value());
        CHECK_FALSE(cache.begin("k", "k").has_value());
        CHECK(cache.prune_missing_sources().entries == 0);
    }

    TEST_CASE("ShaderPackCache: a recorded pack is found for its checkout until the checkout is gone") {
        auto const dir = make_temp_dir("packs");
        ShaderPackCache cache{open_store(dir)};

        auto const shaders = dir / "shaders";
        std::filesystem::create_directories(shaders);

        CHECK_FALSE(cache.find("/checkout").has_value());

        auto pending = cache.begin("/checkout");
        REQUIRE(pending.has_value());
        write_file(pending->path(), "pack");
        REQUIRE(cache.commit(std::move(*pending), shaders.string()));

        auto const found = cache.find("/checkout");
        REQUIRE(found.has_value());
        CHECK(found->extension() == ".lsp");
        CHECK_FALSE(cache.find("/another-checkout").has_value());

        std::filesystem::remove_all(shaders);

        CHECK_FALSE(cache.find("/checkout").has_value());

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("CacheStore: a reserved file is listed, sized once written, and survives orphan removal") {
        auto const dir = make_temp_dir("reserve");
        auto store = open_store(dir);
        declare_files(*store);

        auto const path = store->reserve_file("files", "map", "earth", ".png");
        REQUIRE(path.has_value());
        CHECK(path->extension() == ".png");
        CHECK(std::filesystem::exists(path->parent_path()));

        // Reserved but not written: counted, but a miss.
        CHECK(store->stats()->total_entries() == 1);
        CHECK_FALSE(store->find_file("files", "map").has_value());

        // Reserving again after the writer finished keeps the file and gives the same path.
        REQUIRE(store->reserve_file("files", "map", "earth", ".png").value() == *path);
        write_file(*path, "pixels");
        std::filesystem::last_write_time(*path, std::filesystem::file_time_type::clock::now() - std::chrono::hours{2});

        CHECK(store->remove_orphans().entries == 0);
        CHECK(store->find_file("files", "map") == path);
        CHECK(store->stats()->namespaces.front().bytes == 6);

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("DownloadCache: hands out a stable path, adopts the old location and is never evicted") {
        auto const dir = make_temp_dir("downloads");
        auto store = open_store(dir);
        DownloadCache downloads{store};
        REQUIRE(downloads.enabled());

        auto const legacy = dir / "old" / "model.glb";
        std::filesystem::create_directories(legacy.parent_path());
        write_file(legacy, "glb bytes");

        // Nothing is recorded for something never downloaded.
        CHECK_FALSE(downloads.find("gallery/Missing", "Missing", ".glb", dir / "old" / "nothing.glb").has_value());
        CHECK(store->stats()->total_entries() == 0);

        auto const found = downloads.find("gallery/Model", "Model", ".glb", legacy);
        REQUIRE(found.has_value());
        CHECK(found->extension() == ".glb");
        CHECK_FALSE(std::filesystem::exists(legacy));
        CHECK(std::filesystem::file_size(*found) == 9);

        CHECK(downloads.path("gallery/Model", "Model", ".glb") == *found);
        CHECK(downloads.find("gallery/Model", "Model", ".glb") == found);

        CHECK(store->enforce_budget(1).entries == 0);
        CHECK(std::filesystem::exists(*found));

        std::filesystem::remove_all(dir);
    }

    TEST_CASE("DownloadCache: without a store it hands out nothing") {
        DownloadCache downloads;

        CHECK_FALSE(downloads.enabled());
        CHECK(downloads.path("k", "k", ".bin").empty());
        CHECK_FALSE(downloads.find("k", "k", ".bin").has_value());
    }

    TEST_CASE("CacheStore: a key's lock excludes other holders until it is released") {
        auto const dir = make_temp_dir("locks");
        auto store = open_store(dir);

        auto held = store->lock("files", "k");
        REQUIRE(held);

        // A different key is free; the same key blocks until the holder lets go (or the waiter is told to give up).
        CHECK(store->lock("files", "other"));
        CHECK_FALSE(store->lock("files", "k", {}, std::chrono::seconds{0}));
        CHECK_FALSE(store->lock("files", "k", [] { return true; }));

        std::atomic<bool> acquired{false};
        std::jthread waiter{[&] {
            auto lock = store->lock("files", "k");
            acquired = static_cast<bool>(lock);
        }};

        std::this_thread::sleep_for(std::chrono::milliseconds{200});
        CHECK_FALSE(acquired.load());

        held = CacheLock{};
        waiter.join();
        CHECK(acquired.load());

        std::filesystem::remove_all(dir);
    }

}
