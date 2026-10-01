#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <entt/entity/fwd.hpp>

#include "scripting/script_error.hxx"

struct ScriptEngineSettings {
    std::chrono::milliseconds timeout{50};
    std::size_t memory_limit_bytes = 64ULL * 1024 * 1024;
    std::size_t error_handler_headroom_bytes = 256ULL * 1024;
    // VM instructions between deadline checks.
    std::uint32_t hook_instruction_interval = 1000;
    std::size_t chunk_cache_capacity = 32;
    std::uint32_t max_print_lines_per_run = 256;
    // make_random_engine() stream for Vec3.random.
    std::uint32_t random_stream = 0x5C1F7U;
};

// What a run reads and writes. Scene-independent so tests can use a bare registry; Application passes the editor
// scene's registry and hierarchy_revision().
struct ScriptWorld {
    entt::registry *registry = nullptr;
    std::uint64_t hierarchy_revision = 0;
};

struct ScriptRunReport {
    std::chrono::microseconds duration{};
    // Writes made before an error are kept; there's no rollback.
    std::uint32_t transforms_written = 0;
    bool chunk_cache_hit = false;
    std::size_t peak_bytes = 0;
    std::optional<ScriptError> error;

    [[nodiscard]] auto ok() const noexcept -> bool { return !error.has_value(); }
};

struct ScriptEngineStats {
    std::uint64_t runs = 0;
    std::uint64_t chunk_cache_hits = 0;
    std::uint64_t chunk_cache_misses = 0;
    std::uint64_t entity_index_rebuilds = 0;
    std::size_t lua_bytes_in_use = 0;
    std::size_t lua_peak_bytes = 0;
};

// Sandboxed Lua 5.4 state: base (minus dofile/loadfile/load/collectgarbage, print -> logger), string, table, math;
// text chunks only; per-run wall-clock deadline and memory cap. Main thread only.
//
// Script API: scene.get_entity(name), entity.get_children_or_empty(), entity.get_transform(),
// transform.translation (read: a Vec3 copy, so `t.translation.x = 1` doesn't write; write: a whole Vec3), and Vec3
// (Vec3(x, y, z), Vec3.new, Vec3.random(lo, hi), x/y/z, + - * and unary -). Duplicate entity names resolve to the
// lowest entity id.
class ScriptEngine {
public:
    [[nodiscard]] static auto
    create(ScriptEngineSettings const &settings = {}) -> std::expected<ScriptEngine, ScriptError>;

    ScriptEngine(ScriptEngine const &) = delete;
    auto operator=(ScriptEngine const &) -> ScriptEngine & = delete;
    ScriptEngine(ScriptEngine &&) noexcept;
    auto operator=(ScriptEngine &&) noexcept -> ScriptEngine &;
    ~ScriptEngine();

    // A call made while another run is in flight (re-entered from a registry signal) returns `busy` immediately.
    [[nodiscard]] auto run(std::string_view source, ScriptWorld world) -> ScriptRunReport;

    [[nodiscard]] auto is_running() const noexcept -> bool;

    auto set_timeout(std::chrono::milliseconds timeout) noexcept -> void;
    auto set_memory_limit(std::size_t bytes) noexcept -> void;

    [[nodiscard]] auto settings() const noexcept -> ScriptEngineSettings const &;
    [[nodiscard]] auto stats() const noexcept -> ScriptEngineStats;

    // Sorted global, library and bound member names; built once at create(), used for "did you mean".
    [[nodiscard]] auto api_names() const noexcept -> std::span<std::string const>;

private:
    struct Impl;
    explicit ScriptEngine(std::unique_ptr<Impl> impl) noexcept;

    // Heap-stable: the allocator's user data and lua_getextraspace() point into it.
    std::unique_ptr<Impl> impl_;
};
