#pragma once

#include <chrono>
#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "scripting/lua_api.hxx"

// A persistent, sandboxed Lua 5.4 state for game code, as opposed to ScriptEngine's one-shot editor runs.
//
// A game's entry script returns a table of callbacks (on_update, on_ui, ...). The host calls them by name; a callback
// that raises an error is reported once and then disabled until reset_errors(), so one bad frame doesn't flood the log
// or leave the game half-updated every frame.
//
// Sandbox: base (minus dofile, loadfile, load, collectgarbage, warn), coroutine, table, string, math and utf8. No io,
// os or package; require() resolves modules through the source loader and the registered native modules. Every call
// has a wall-clock budget and the whole state a memory cap. Main thread only.
class LuaRuntime {
public:
    struct Settings {
        std::size_t memory_limit_bytes = 128ULL * 1024 * 1024;
        // Per call into Lua (a callback, or loading the entry script, which gets 10x).
        std::chrono::milliseconds call_budget{20};
    };

    // "a/b" for require("a.b"): the module's source, or nothing if there is no such module.
    using SourceLoader = std::function<std::optional<std::string>(std::string const &module_path)>;

    struct CallResult {
        bool ok = true;
        std::string error;
    };

    [[nodiscard]] static auto create(Settings const &settings) -> std::expected<LuaRuntime, std::string>;
    [[nodiscard]] static auto create() -> std::expected<LuaRuntime, std::string>;

    LuaRuntime(LuaRuntime const &) = delete;
    auto operator=(LuaRuntime const &) -> LuaRuntime & = delete;
    LuaRuntime(LuaRuntime &&) noexcept;
    auto operator=(LuaRuntime &&) noexcept -> LuaRuntime &;
    ~LuaRuntime();

    auto set_source_loader(SourceLoader loader) -> void;

    // require("name") returns what `opener` leaves on top of the stack. Register before run_main().
    auto register_native_module(std::string name, lua_CFunction opener) -> void;

    // Bindings find their host through host(state).
    auto set_host(void *host) noexcept -> void;
    [[nodiscard]] static auto host(lua_State *state) noexcept -> void *;

    // Runs the entry script, which must return the callback table.
    [[nodiscard]] auto run_main(std::string_view chunk_name, std::string_view source) -> std::expected<void, std::string>;

    [[nodiscard]] auto has_callback(char const *name) const -> bool;

    // Whether the callback table's `name` field is the boolean true.
    [[nodiscard]] auto callback_flag(char const *name) const -> bool;

    // Calls callbacks[name](numbers...), leaving `result_count` results on the stack on success (read them through
    // state(), then pop them). A missing or disabled callback returns ok with nothing pushed only if result_count is
    // 0; otherwise it returns an error.
    [[nodiscard]] auto call(char const *name, std::span<double const> numbers = {}, int result_count = 0) -> CallResult;

    // Pops what call() left.
    auto pop(int count) -> void;

    [[nodiscard]] auto state() const noexcept -> lua_State *;

    // The first error since the last reset, for an on-screen report.
    [[nodiscard]] auto last_error() const noexcept -> std::string const &;
    auto reset_errors() -> void;

    [[nodiscard]] auto memory_in_use() const noexcept -> std::size_t;

private:
    struct Impl;
    explicit LuaRuntime(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

// Table field readers for results left by call(). `index` is an absolute stack index.
[[nodiscard]] auto lua_field_number(lua_State *state, int index, char const *key, double fallback) -> double;

// {x, y, z} or {x=, y=, z=} sub-table `key` of the table at `index`; `fallback` if absent or malformed.
struct LuaVec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};
[[nodiscard]] auto lua_field_vec3(lua_State *state, int index, char const *key, LuaVec3 fallback) -> LuaVec3;
