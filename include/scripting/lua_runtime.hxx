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

class LuaRuntime {
public:
    struct Settings {
        std::size_t memory_limit_bytes = 128ULL * 1024 * 1024;
        std::chrono::milliseconds call_budget{20};
    };

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

    auto register_native_module(std::string name, lua_CFunction opener) -> void;

    auto set_host(void *host) noexcept -> void;
    [[nodiscard]] static auto host(lua_State *state) noexcept -> void *;

    [[nodiscard]] auto run_main(std::string_view chunk_name, std::string_view source) -> std::expected<void, std::string>;

    [[nodiscard]] auto has_callback(char const *name) const -> bool;

    [[nodiscard]] auto callback_flag(char const *name) const -> bool;

    [[nodiscard]] auto call(char const *name, std::span<double const> numbers = {}, int result_count = 0) -> CallResult;

    // Calls the function on the stack below its `argument_count` arguments under the call budget (`loading` allows
    // the longer budget of a chunk's top level). On success the results are left on the stack; on a failure the
    // function and arguments are removed and the message (with a traceback) is returned. Unlike call(), nothing is
    // disabled or remembered as the runtime's error: the caller decides what a failure means.
    [[nodiscard]] auto protected_call(int argument_count, int result_count, bool loading = false) -> CallResult;

    auto pop(int count) -> void;

    [[nodiscard]] auto state() const noexcept -> lua_State *;

    [[nodiscard]] auto last_error() const noexcept -> std::string const &;
    auto reset_errors() -> void;

    [[nodiscard]] auto memory_in_use() const noexcept -> std::size_t;

private:
    struct Impl;
    explicit LuaRuntime(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] auto lua_field_number(lua_State *state, int index, char const *key, double fallback) -> double;

struct LuaVec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};
[[nodiscard]] auto lua_field_vec3(lua_State *state, int index, char const *key, LuaVec3 fallback) -> LuaVec3;
