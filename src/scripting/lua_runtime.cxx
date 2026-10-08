#include "scripting/lua_runtime.hxx"

#include <algorithm>
#include <format>
#include <set>
#include <unordered_map>
#include <utility>

#include "core/logger.hxx"
#include "scripting/lua_allocator.hxx"

namespace {
    constexpr char const *callbacks_key = "lathe.callbacks";
    constexpr char const *loaded_key = "lathe.loaded";

    constexpr std::size_t max_module_name_length = 128;
    constexpr int load_budget_multiplier = 10;
}

struct LuaRuntimeContext {
    LuaMemoryBudget budget;
    std::chrono::steady_clock::time_point deadline{};
    std::chrono::milliseconds active_budget{0};
    void *host = nullptr;
    LuaRuntime::SourceLoader loader;
    std::unordered_map<std::string, lua_CFunction> native_modules;
};

struct LuaRuntime::Impl {
    struct StateDeleter {
        auto operator()(lua_State *state) const noexcept -> void { lua_close(state); }
    };

    LuaRuntimeContext context;
    std::unique_ptr<lua_State, StateDeleter> state;
    Settings settings;
    std::set<std::string, std::less<>> disabled;
    std::string last_error;
};

namespace {
    auto context_of(lua_State *state) noexcept -> LuaRuntimeContext & {
        return **static_cast<LuaRuntimeContext **>(lua_getextraspace(state));
    }

    auto lua_panic(lua_State *state) -> int {
        auto const *message = lua_tostring(state, -1);

        error("[lua] panic: {}", message != nullptr ? message : "(no message)");

        return 0;
    }

    auto deadline_hook(lua_State *state, lua_Debug * ) -> void {
        auto &context = context_of(state);

        if (std::chrono::steady_clock::now() < context.deadline) {
            return;
        }

        lua_sethook(state, nullptr, 0, 0);
        luaL_error(state, "script exceeded its time budget of %d ms", static_cast<int>(context.active_budget.count()));
    }

    auto message_handler(lua_State *state) -> int {
        auto const *message = lua_tostring(state, 1);

        if (message == nullptr) {
            if (luaL_callmeta(state, 1, "__tostring") != 0 && lua_type(state, -1) == LUA_TSTRING) {
                return 1;
            }

            message = lua_pushfstring(state, "(error object is a %s value)", luaL_typename(state, 1));
        }

        luaL_traceback(state, state, message, 1);

        return 1;
    }

    auto lua_print(lua_State *state) -> int {
        auto const count = lua_gettop(state);

        luaL_Buffer buffer;
        luaL_buffinit(state, &buffer);

        for (int index = 1; index <= count; ++index) {
            std::size_t length = 0;
            auto const *text = luaL_tolstring(state, index, &length);

            if (index > 1) {
                luaL_addchar(&buffer, '\t');
            }

            luaL_addlstring(&buffer, text, length);
            lua_pop(state, 1);
        }

        luaL_pushresult(&buffer);

        info("[lua] {}", std::string_view{lua_tostring(state, -1)});

        return 0;
    }

    auto valid_module_name(std::string_view name) noexcept -> bool {
        if (name.empty() || name.size() > max_module_name_length || name.front() == '.' || name.back() == '.' ||
            name.find("..") != std::string_view::npos) {
            return false;
        }

        return std::ranges::all_of(name, [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.';
        });
    }

    auto lua_require(lua_State *state) -> int {
        std::size_t length = 0;
        auto const *raw = luaL_checklstring(state, 1, &length);

        lua_getfield(state, LUA_REGISTRYINDEX, loaded_key);
        auto const loaded_index = lua_gettop(state);

        lua_pushvalue(state, 1);
        lua_rawget(state, loaded_index);

        if (!lua_isnil(state, -1)) {
            return 1;
        }

        lua_pop(state, 1);

        lua_CFunction native = nullptr;
        bool found_source = false;
        bool bad_name = false;
        bool load_failed = false;

        {
            std::string const name{raw, length};

            if (!valid_module_name(name)) {
                bad_name = true;
            } else if (auto const &natives = context_of(state).native_modules; natives.contains(name)) {
                native = natives.at(name);
            } else if (auto const &loader = context_of(state).loader) {
                std::string path = name;

                std::ranges::replace(path, '.', '/');

                if (auto source = loader(path)) {
                    found_source = true;

                    auto const chunk = std::format("@{}.lua", path);

                    load_failed = luaL_loadbufferx(state, source->data(), source->size(), chunk.c_str(), "t") != LUA_OK;
                }
            }
        }

        if (bad_name) {
            return luaL_error(state, "invalid module name '%s'", raw);
        }

        if (native == nullptr && !found_source) {
            return luaL_error(state, "module '%s' not found", raw);
        }

        if (load_failed) {
            return lua_error(state);
        }

        if (native != nullptr) {
            lua_pushcfunction(state, native);
        }

        lua_pushvalue(state, 1);
        lua_call(state, 1, 1);

        if (lua_isnil(state, -1)) {
            lua_pop(state, 1);
            lua_pushboolean(state, 1);
        }

        lua_pushvalue(state, 1);
        lua_pushvalue(state, -2);
        lua_rawset(state, loaded_index);

        return 1;
    }

    auto open_sandbox(lua_State *state) -> void {
        luaL_requiref(state, "_G", luaopen_base, 1);
        luaL_requiref(state, LUA_COLIBNAME, luaopen_coroutine, 1);
        luaL_requiref(state, LUA_TABLIBNAME, luaopen_table, 1);
        luaL_requiref(state, LUA_STRLIBNAME, luaopen_string, 1);
        luaL_requiref(state, LUA_MATHLIBNAME, luaopen_math, 1);
        luaL_requiref(state, LUA_UTF8LIBNAME, luaopen_utf8, 1);
        lua_pop(state, 6);

        for (auto const *name: {"dofile", "loadfile", "load", "collectgarbage", "warn"}) {
            lua_pushnil(state);
            lua_setglobal(state, name);
        }

        lua_pushcfunction(state, &lua_print);
        lua_setglobal(state, "print");

        lua_pushcfunction(state, &lua_require);
        lua_setglobal(state, "require");

        lua_newtable(state);
        lua_setfield(state, LUA_REGISTRYINDEX, loaded_key);
    }

    struct CallGuard {
        LuaRuntimeContext &context;
        lua_State *state;
        std::chrono::milliseconds budget;

        CallGuard(LuaRuntimeContext &c, lua_State *s, std::chrono::milliseconds b) : context(c), state(s), budget(b) {
            context.active_budget = budget;
            context.deadline = std::chrono::steady_clock::now() + budget;
            context.budget.reset_run_flags();

            lua_sethook(state, &deadline_hook, LUA_MASKCOUNT, 1000);
        }

        ~CallGuard() { lua_sethook(state, nullptr, 0, 0); }

        CallGuard(CallGuard const &) = delete;
        auto operator=(CallGuard const &) -> CallGuard & = delete;
    };
}

auto LuaRuntime::create() -> std::expected<LuaRuntime, std::string> { return create(Settings{}); }

auto LuaRuntime::create(Settings const &settings) -> std::expected<LuaRuntime, std::string> {
    auto impl = std::make_unique<Impl>();

    impl->settings = settings;
    impl->context.budget.limit_bytes = settings.memory_limit_bytes;

    auto *const state = lua_newstate(&lua_budget_alloc, &impl->context.budget);

    if (state == nullptr) {
        return std::unexpected{"could not create the Lua state"};
    }

    impl->state.reset(state);

    auto *context = &impl->context;

    *static_cast<LuaRuntimeContext **>(lua_getextraspace(state)) = context;

    lua_atpanic(state, &lua_panic);

    open_sandbox(state);

    lua_settop(state, 0);

    return LuaRuntime{std::move(impl)};
}

LuaRuntime::LuaRuntime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LuaRuntime::LuaRuntime(LuaRuntime &&) noexcept = default;

auto LuaRuntime::operator=(LuaRuntime &&) noexcept -> LuaRuntime & = default;

LuaRuntime::~LuaRuntime() = default;

auto LuaRuntime::set_source_loader(SourceLoader loader) -> void { impl_->context.loader = std::move(loader); }

auto LuaRuntime::register_native_module(std::string name, lua_CFunction opener) -> void {
    impl_->context.native_modules.insert_or_assign(std::move(name), opener);
}

auto LuaRuntime::set_host(void *host) noexcept -> void { impl_->context.host = host; }

auto LuaRuntime::host(lua_State *state) noexcept -> void * { return context_of(state).host; }

auto LuaRuntime::state() const noexcept -> lua_State * { return impl_->state.get(); }

auto LuaRuntime::last_error() const noexcept -> std::string const & { return impl_->last_error; }

auto LuaRuntime::reset_errors() -> void {
    impl_->disabled.clear();
    impl_->last_error.clear();
}

auto LuaRuntime::memory_in_use() const noexcept -> std::size_t { return impl_->context.budget.used_bytes; }

auto LuaRuntime::pop(int count) -> void { lua_pop(impl_->state.get(), count); }

auto LuaRuntime::run_main(std::string_view chunk_name, std::string_view source) -> std::expected<void, std::string> {
    auto *const state = impl_->state.get();
    auto const chunk = std::format("@{}", chunk_name);

    if (luaL_loadbufferx(state, source.data(), source.size(), chunk.c_str(), "t") != LUA_OK) {
        std::string message = lua_tostring(state, -1);

        lua_pop(state, 1);

        return std::unexpected{std::move(message)};
    }

    lua_pushcfunction(state, &message_handler);
    lua_insert(state, -2);

    int status = LUA_OK;

    {
        CallGuard const guard{impl_->context, state, impl_->settings.call_budget * load_budget_multiplier};

        status = lua_pcall(state, 0, 1, -2);
    }

    if (status != LUA_OK) {
        std::string message = lua_tostring(state, -1) != nullptr ? lua_tostring(state, -1) : "(unknown error)";

        lua_settop(state, 0);

        return std::unexpected{std::move(message)};
    }

    if (!lua_istable(state, -1)) {
        lua_settop(state, 0);

        return std::unexpected{"the entry script must return a table of callbacks"};
    }

    lua_setfield(state, LUA_REGISTRYINDEX, callbacks_key);
    lua_settop(state, 0);

    return {};
}

auto LuaRuntime::has_callback(char const *name) const -> bool {
    auto *const state = impl_->state.get();

    lua_getfield(state, LUA_REGISTRYINDEX, callbacks_key);

    if (!lua_istable(state, -1)) {
        lua_pop(state, 1);

        return false;
    }

    lua_getfield(state, -1, name);

    auto const present = lua_isfunction(state, -1) != 0;

    lua_pop(state, 2);

    return present && !impl_->disabled.contains(std::string_view{name});
}

auto LuaRuntime::callback_flag(char const *name) const -> bool {
    auto *const state = impl_->state.get();

    lua_getfield(state, LUA_REGISTRYINDEX, callbacks_key);

    if (!lua_istable(state, -1)) {
        lua_pop(state, 1);

        return false;
    }

    lua_getfield(state, -1, name);

    auto const flag = lua_isboolean(state, -1) != 0 && lua_toboolean(state, -1) != 0;

    lua_pop(state, 2);

    return flag;
}

auto LuaRuntime::call(char const *name, std::span<double const> numbers, int result_count) -> CallResult {
    auto *const state = impl_->state.get();

    if (!has_callback(name)) {
        if (result_count == 0) {
            return {};
        }

        return {.ok = false, .error = std::format("callback '{}' is missing or disabled", name)};
    }

    lua_pushcfunction(state, &message_handler);
    auto const handler_index = lua_gettop(state);

    lua_getfield(state, LUA_REGISTRYINDEX, callbacks_key);
    lua_getfield(state, -1, name);
    lua_remove(state, -2);

    for (auto const number: numbers) {
        lua_pushnumber(state, number);
    }

    int status = LUA_OK;

    {
        CallGuard const guard{impl_->context, state, impl_->settings.call_budget};

        status = lua_pcall(state, static_cast<int>(numbers.size()), result_count, handler_index);
    }

    if (status != LUA_OK) {
        std::string message = lua_tostring(state, -1) != nullptr ? lua_tostring(state, -1) : "(unknown error)";

        lua_settop(state, handler_index - 1);

        impl_->disabled.insert(name);

        if (impl_->last_error.empty()) {
            impl_->last_error = std::format("{}: {}", name, message);
        }

        error("[lua] {}: {}", name, message);

        return {.ok = false, .error = std::move(message)};
    }

    lua_remove(state, handler_index);

    return {};
}

auto lua_field_number(lua_State *state, int index, char const *key, double fallback) -> double {
    lua_getfield(state, index, key);

    auto const value = lua_isnumber(state, -1) != 0 ? lua_tonumber(state, -1) : fallback;

    lua_pop(state, 1);

    return value;
}

auto lua_field_vec3(lua_State *state, int index, char const *key, LuaVec3 fallback) -> LuaVec3 {
    lua_getfield(state, index, key);

    LuaVec3 result = fallback;

    if (lua_istable(state, -1)) {
        auto const table = lua_gettop(state);
        auto const read = [&](char const *name, lua_Integer position, float current) {
            lua_getfield(state, table, name);

            if (lua_isnil(state, -1)) {
                lua_pop(state, 1);
                lua_geti(state, table, position);
            }

            auto const value = lua_isnumber(state, -1) != 0 ? static_cast<float>(lua_tonumber(state, -1)) : current;

            lua_pop(state, 1);

            return value;
        };

        result.x = read("x", 1, fallback.x);
        result.y = read("y", 2, fallback.y);
        result.z = read("z", 3, fallback.z);
    }

    lua_pop(state, 1);

    return result;
}
