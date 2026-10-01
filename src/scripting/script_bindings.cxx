// The Lua-facing API. Lua is compiled as C, so a raised error longjmps past every C++ frame up to the protected
// call. Every lua_CFunction here therefore:
//   - reads and checks its arguments with luaL_check*/luaL_test* before creating any C++ object,
//   - does its C++ work (map lookups, std::format, registry.patch) inside a {} scope that ends before any call
//     that can raise,
//   - carries error text out of that scope in a std::array<char, N>,
//   - raises only through luaL_error.
// sol2 is used for the Vec3 usertype only.

#include "scripting/lua_api.hxx"

#include <sol/sol.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <new>
#include <random>
#include <span>
#include <string_view>
#include <utility>

#include <glm/vec3.hpp>

#include "core/logger.hxx"
#include "core/transform.hxx"
#include "rendering/entity.hxx"
#include "scripting/script_error.hxx"
#include "scripting/script_runtime.hxx"

namespace scripting::detail {
    namespace {
        constexpr char const *entity_metatable = "lathe.Entity";
        constexpr char const *transform_metatable = "lathe.Transform";

        // For luaL_error's "%s"; the same text as invalid_entity_tag.
        constexpr char const *invalid_entity_tag_cstr = "[invalid_entity] ";
        static_assert(std::string_view{invalid_entity_tag_cstr} == invalid_entity_tag);

        // print() output past this many bytes per line is cut off.
        constexpr std::size_t max_print_line_bytes = 4ULL * 1024;

        // Base-library functions scripts keep. Left out: dofile, loadfile, load (binary chunks), collectgarbage
        // ("stop" defeats the memory cap), warn and _G (the real globals also hold sol2's internals).
        constexpr std::array<char const *, 18> copied_base_names{
                "assert",       "error",    "getmetatable", "ipairs", "next",   "pairs",
                "pcall",        "rawequal", "rawget",       "rawlen", "rawset", "select",
                "setmetatable", "tonumber", "tostring",     "type",   "xpcall", "_VERSION",
        };

        constexpr std::array<char const *, 3> read_only_libraries{LUA_STRLIBNAME, LUA_TABLIBNAME, LUA_MATHLIBNAME};

        constexpr std::array<std::string_view, 9> member_names{
                "get_entity", "get_children_or_empty", "get_transform", "translation", "x", "y", "z", "random", "new",
        };

        // Full userdata; the generation catches refs that outlive a registry change.
        struct LuaEntityRef {
            entt::entity id;
            std::uint32_t generation;
        };

        struct LuaTransformRef {
            entt::entity id;
            std::uint32_t generation;
        };

        // Returns the live entity behind the ref at `index`, or raises "[invalid_entity] entity <id> no longer
        // exists".
        template<typename Ref>
        auto check_ref(lua_State *state, int index, char const *metatable) -> entt::entity {
            auto const *const ref = static_cast<Ref const *>(luaL_checkudata(state, index, metatable));
            auto const &context = run_context(state);
            bool const alive = ref->generation == context.world_generation && context.world.registry != nullptr &&
                               context.world.registry->valid(ref->id);
            if (!alive) {
                luaL_error(state, "%sentity %I no longer exists", invalid_entity_tag_cstr,
                           static_cast<lua_Integer>(entt::to_integral(ref->id)));
                return entt::null;
            }
            return ref->id;
        }

        auto check_entity(lua_State *state, int index) -> entt::entity {
            return check_ref<LuaEntityRef>(state, index, entity_metatable);
        }

        // Also requires the Transform to still be there.
        auto check_transform(lua_State *state, int index) -> entt::entity {
            auto const entity = check_ref<LuaTransformRef>(state, index, transform_metatable);
            if (!run_context(state).world.registry->all_of<Components::Transform>(entity)) {
                luaL_error(state, "entity %I has no Transform", static_cast<lua_Integer>(entt::to_integral(entity)));
                return entt::null;
            }
            return entity;
        }

        template<typename Ref>
        auto push_ref(lua_State *state, entt::entity entity, char const *metatable) -> void {
            void *const memory = lua_newuserdatauv(state, sizeof(Ref), 0);
            ::new (memory) Ref{.id = entity, .generation = run_context(state).world_generation};
            luaL_setmetatable(state, metatable);
        }

        auto push_entity(lua_State *state, entt::entity entity) -> void {
            push_ref<LuaEntityRef>(state, entity, entity_metatable);
        }

        auto refresh_entity_index(RunContext &context) -> void {
            if (context.entity_index->refresh(context.world)) {
                ++*context.entity_index_rebuilds;
            }
        }

        // Accepts scene.get_entity(name) and scene:get_entity(name).
        auto scene_get_entity(lua_State *state) -> int {
            int const name_index = lua_gettop(state) >= 2 && lua_isstring(state, 1) == 0 ? 2 : 1;
            std::size_t length = 0;
            char const *const raw_name = luaL_checklstring(state, name_index, &length);

            std::array<char, 256> message{};
            entt::entity found = entt::null;
            {
                auto &context = run_context(state);
                std::string_view const name{raw_name, length};
                refresh_entity_index(context);
                found = context.entity_index->find(name);
                if (found == entt::null || !context.world.registry->valid(found)) {
                    auto const limit = static_cast<std::ptrdiff_t>(message.size() - 1);
                    // The array is zeroed and one byte is held back, so the text stays terminated.
                    if (auto const suggestion = closest_name(name, context.entity_index->names())) {
                        static_cast<void>(std::format_to_n(
                                message.data(), limit, "no entity named '{}' (did you mean '{}'?)", name, *suggestion));
                    } else {
                        static_cast<void>(std::format_to_n(message.data(), limit, "no entity named '{}'", name));
                    }
                }
            }

            if (message[0] != '\0') {
                return luaL_error(state, "%s%s", invalid_entity_tag_cstr, message.data());
            }

            push_entity(state, found);
            return 1;
        }

        auto entity_get_children_or_empty(lua_State *state) -> int {
            auto const entity = check_entity(state, lua_upvalueindex(1));

            // A view into the engine's index: nothing here owns memory, so a raise below leaks nothing. The index
            // can't be rebuilt mid-run, since a run's world (and so its revision) is fixed.
            std::span<entt::entity const> children;
            {
                auto &context = run_context(state);
                refresh_entity_index(context);
                children = context.entity_index->children(entity);
            }

            auto *const registry = run_context(state).world.registry;
            lua_createtable(state, static_cast<int>(children.size()), 0);
            lua_Integer count = 0;
            for (auto const child: children) {
                // Parent may have been edited in place since the index was built.
                auto const *const parent =
                        registry->valid(child) ? registry->try_get<Components::Parent>(child) : nullptr;
                if (parent == nullptr || parent->entity != entity) {
                    continue;
                }
                push_entity(state, child);
                lua_rawseti(state, -2, ++count);
            }
            return 1;
        }

        auto entity_get_transform(lua_State *state) -> int {
            auto const entity = check_entity(state, lua_upvalueindex(1));
            if (!run_context(state).world.registry->all_of<Components::Transform>(entity)) {
                return luaL_error(state, "entity %I has no Transform",
                                  static_cast<lua_Integer>(entt::to_integral(entity)));
            }
            push_ref<LuaTransformRef>(state, entity, transform_metatable);
            return 1;
        }

        // __index. Methods are closures over the entity, so both e.get_transform() (the documented form) and
        // e:get_transform() work.
        auto entity_index(lua_State *state) -> int {
            luaL_checkudata(state, 1, entity_metatable);

            lua_CFunction method = nullptr;
            if (lua_type(state, 2) == LUA_TSTRING) {
                std::size_t length = 0;
                char const *const raw_key = lua_tolstring(state, 2, &length);
                std::string_view const key{raw_key, length};
                if (key == "get_children_or_empty") {
                    method = &entity_get_children_or_empty;
                } else if (key == "get_transform") {
                    method = &entity_get_transform;
                }
            }

            // Unknown keys are nil, so calling one reads "attempt to call a nil value (field 'x')" plus a
            // suggestion.
            if (method == nullptr) {
                lua_pushnil(state);
                return 1;
            }

            lua_pushvalue(state, 1);
            lua_pushcclosure(state, method, 1);
            return 1;
        }

        // "Entity(42, 'Helmets')"; never raises, so a stale ref can still be printed.
        auto entity_to_string(lua_State *state) -> int {
            auto const *const ref = static_cast<LuaEntityRef const *>(luaL_checkudata(state, 1, entity_metatable));

            std::array<char, 256> text{};
            std::size_t size = 0;
            {
                auto const &context = run_context(state);
                auto const *const registry = context.world.registry;
                auto const id = entt::to_integral(ref->id);
                auto const limit = static_cast<std::ptrdiff_t>(text.size());

                std::string_view name;
                bool const alive =
                        ref->generation == context.world_generation && registry != nullptr && registry->valid(ref->id);
                if (alive) {
                    if (auto const *generated = registry->try_get<Components::GeneratedMeta>(ref->id)) {
                        name = generated->name;
                    } else if (auto const *meta = registry->try_get<Components::Meta>(ref->id)) {
                        name = meta->name.view();
                    }
                }

                auto const result = !alive         ? std::format_to_n(text.data(), limit, "Entity({}, destroyed)", id)
                                    : name.empty() ? std::format_to_n(text.data(), limit, "Entity({})", id)
                                                   : std::format_to_n(text.data(), limit, "Entity({}, '{}')", id, name);
                size = static_cast<std::size_t>(result.out - text.data());
            }

            lua_pushlstring(state, text.data(), size);
            return 1;
        }

        auto entity_equals(lua_State *state) -> int {
            auto const *const left = static_cast<LuaEntityRef const *>(luaL_testudata(state, 1, entity_metatable));
            auto const *const right = static_cast<LuaEntityRef const *>(luaL_testudata(state, 2, entity_metatable));
            bool const equal = left != nullptr && right != nullptr && left->id == right->id &&
                               left->generation == right->generation;
            lua_pushboolean(state, equal ? 1 : 0);
            return 1;
        }

        // "translation" -> a Vec3 copy of the position.
        auto transform_index(lua_State *state) -> int {
            auto const entity = check_transform(state, 1);

            bool is_translation = false;
            if (lua_type(state, 2) == LUA_TSTRING) {
                std::size_t length = 0;
                char const *const raw_key = lua_tolstring(state, 2, &length);
                is_translation = std::string_view{raw_key, length} == "translation";
            }
            if (!is_translation) {
                lua_pushnil(state);
                return 1;
            }

            glm::vec3 const position = run_context(state).world.registry->get<Components::Transform>(entity).position;
            sol::stack::push(state, position);
            return 1;
        }

        // "translation" = Vec3 -> registry.patch, which fires on_update<Transform> (and so
        // Scene::on_transform_changed).
        auto transform_new_index(lua_State *state) -> int {
            std::size_t length = 0;
            char const *const raw_key = luaL_checklstring(state, 2, &length);
            if (std::string_view{raw_key, length} != "translation") {
                return luaL_error(state, "Transform has no writable field '%s'", raw_key);
            }

            glm::vec3 value{};
            bool is_vec3 = false;
            {
                auto const maybe_vec3 = sol::stack::check_get<glm::vec3 *>(state, 3);
                if (maybe_vec3.has_value() && *maybe_vec3 != nullptr) {
                    value = **maybe_vec3;
                    is_vec3 = true;
                }
            }
            if (!is_vec3) {
                return luaL_error(state, "translation expects a Vec3, got %s", luaL_typename(state, 3));
            }

            auto const entity = check_transform(state, 1);
            {
                auto &context = run_context(state);
                context.world.registry->patch<Components::Transform>(
                        entity, [&value](Components::Transform &transform) { transform.position = value; });
                ++context.transforms_written;
            }
            return 0;
        }

        // Vec3.random(lo, hi): each component uniform in [lo, hi).
        auto vec3_random(lua_State *state) -> int {
            auto const first = luaL_checknumber(state, 1);
            auto const second = luaL_checknumber(state, 2);

            auto low = static_cast<float>(first);
            auto high = static_cast<float>(second);
            if (low > high) {
                std::swap(low, high);
            }
            if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(high - low)) {
                return luaL_error(state, "Vec3.random expects finite bounds");
            }

            glm::vec3 value{};
            {
                auto &random = *run_context(state).random;
                std::uniform_real_distribution<float> distribution{low, high};
                value.x = distribution(random);
                value.y = distribution(random);
                value.z = distribution(random);
            }

            sol::stack::push(state, value);
            return 1;
        }

        auto vec3_to_string(lua_State *state) -> int {
            std::array<char, 160> text{};
            std::size_t size = 0;
            {
                auto const maybe_vec3 = sol::stack::check_get<glm::vec3 *>(state, 1);
                if (maybe_vec3.has_value() && *maybe_vec3 != nullptr) {
                    auto const &vector = **maybe_vec3;
                    auto const result = std::format_to_n(text.data(), static_cast<std::ptrdiff_t>(text.size()),
                                                         "Vec3({:g}, {:g}, {:g})", vector.x, vector.y, vector.z);
                    size = static_cast<std::size_t>(result.out - text.data());
                }
            }
            if (size == 0) {
                return luaL_error(state, "Vec3 expected, got %s", luaL_typename(state, 1));
            }

            lua_pushlstring(state, text.data(), size);
            return 1;
        }

        // print(...) -> logger::info("[script] ..."), tab-separated like the standard print. Capped per run.
        auto sandbox_print(lua_State *state) -> int {
            int const count = lua_gettop(state);

            // Built in Lua memory, so a raise from a __tostring here leaks nothing.
            luaL_Buffer buffer;
            luaL_buffinit(state, &buffer);
            for (int index = 1; index <= count; ++index) {
                if (index > 1) {
                    luaL_addchar(&buffer, '\t');
                }
                luaL_tolstring(state, index, nullptr);
                luaL_addvalue(&buffer);
            }
            luaL_pushresult(&buffer);

            std::size_t length = 0;
            char const *const line = lua_tolstring(state, -1, &length);
            {
                auto &context = run_context(state);
                if (context.printed_lines < context.max_printed_lines) {
                    logger::info("[script] {}", std::string_view{line, std::min(length, max_print_line_bytes)});
                } else if (context.printed_lines == context.max_printed_lines) {
                    logger::warn("[script] print limit of {} lines reached; further output dropped",
                                 context.max_printed_lines);
                }
                if (context.printed_lines <= context.max_printed_lines) {
                    ++context.printed_lines;
                }
            }
            return 0;
        }

        // __newindex on a read-only proxy; upvalue 1 is the library name.
        auto read_only_new_index(lua_State *state) -> int {
            return luaL_error(state, "'%s' is read-only", lua_tostring(state, lua_upvalueindex(1)));
        }

        // __call on a proxy: calls the proxied object (upvalue 1) with the same arguments, as if it had been called
        // directly. Calling a table runs its own __call with the table prepended, which is what sol2's
        // call_constructor expects.
        auto forward_call(lua_State *state) -> int {
            // [proxy, args...] -> [target, args...]
            lua_pushvalue(state, lua_upvalueindex(1));
            lua_replace(state, 1);
            lua_call(state, lua_gettop(state) - 1, LUA_MULTRET);
            return lua_gettop(state);
        }

        // __pairs on a proxy: iterates the proxied table (upvalue 1).
        auto proxy_pairs(lua_State *state) -> int {
            lua_getglobal(state, "next");
            lua_pushvalue(state, lua_upvalueindex(1));
            lua_pushnil(state);
            return 3;
        }

        // Pushes a read-only proxy for the table at `table_index`. The proxy is a zero-size full userdata rather than
        // a table, so rawset() can't plant fields on it either. Proxies persist across runs, so this is what keeps
        // one script from changing the libraries the next one sees.
        auto make_read_only(lua_State *state, int table_index, char const *name, bool forward_calls) -> void {
            table_index = lua_absindex(state, table_index);

            lua_newuserdatauv(state, 0, 0);
            lua_createtable(state, 0, 5);

            lua_pushvalue(state, table_index);
            lua_setfield(state, -2, "__index");

            lua_pushstring(state, name);
            lua_pushcclosure(state, &read_only_new_index, 1);
            lua_setfield(state, -2, "__newindex");

            lua_pushvalue(state, table_index);
            lua_pushcclosure(state, &proxy_pairs, 1);
            lua_setfield(state, -2, "__pairs");

            if (forward_calls) {
                lua_pushvalue(state, table_index);
                lua_pushcclosure(state, &forward_call, 1);
                lua_setfield(state, -2, "__call");
            }

            lua_pushboolean(state, 0);
            lua_setfield(state, -2, "__metatable");

            lua_setmetatable(state, -2);
        }

        // Pushes the Vec3 usertype table (registered through a throwaway staging table, never a global).
        auto push_vec3_usertype(lua_State *state) -> void {
            sol::state_view lua{state};
            sol::table staging = lua.create_table();

            staging.new_usertype<glm::vec3>(
                    "Vec3", sol::call_constructor, sol::constructors<glm::vec3(), glm::vec3(float, float, float)>(),
                    "new", sol::constructors<glm::vec3(), glm::vec3(float, float, float)>(), "x", &glm::vec3::x, "y",
                    &glm::vec3::y, "z", &glm::vec3::z, "random", &vec3_random, sol::meta_function::addition,
                    [](glm::vec3 const &a, glm::vec3 const &b) { return a + b; }, sol::meta_function::subtraction,
                    [](glm::vec3 const &a, glm::vec3 const &b) { return a - b; }, sol::meta_function::multiplication,
                    sol::overload([](glm::vec3 const &v, float s) { return v * s; },
                                  [](float s, glm::vec3 const &v) { return s * v; },
                                  [](glm::vec3 const &a, glm::vec3 const &b) { return a * b; }),
                    sol::meta_function::unary_minus, [](glm::vec3 const &v) { return -v; },
                    sol::meta_function::equal_to, [](glm::vec3 const &a, glm::vec3 const &b) { return a == b; },
                    sol::meta_function::to_string, &vec3_to_string);

            sol::table const vec3 = staging["Vec3"];
            static_cast<void>(vec3.push(state));
        }

        auto register_metatables(lua_State *state) -> void {
            luaL_newmetatable(state, entity_metatable);
            lua_pushcfunction(state, &entity_index);
            lua_setfield(state, -2, "__index");
            lua_pushcfunction(state, &entity_to_string);
            lua_setfield(state, -2, "__tostring");
            lua_pushcfunction(state, &entity_equals);
            lua_setfield(state, -2, "__eq");
            lua_pushboolean(state, 0);
            lua_setfield(state, -2, "__metatable");
            lua_pop(state, 1);

            luaL_newmetatable(state, transform_metatable);
            lua_pushcfunction(state, &transform_index);
            lua_setfield(state, -2, "__index");
            lua_pushcfunction(state, &transform_new_index);
            lua_setfield(state, -2, "__newindex");
            lua_pushboolean(state, 0);
            lua_setfield(state, -2, "__metatable");
            lua_pop(state, 1);
        }
    } // namespace

    auto run_context(lua_State *state) noexcept -> RunContext & {
        return **static_cast<RunContext **>(lua_getextraspace(state));
    }

    auto build_sandbox(lua_State *state, RunContext & /*context*/) -> void {
        luaL_requiref(state, LUA_GNAME, &luaopen_base, 1);
        luaL_requiref(state, LUA_STRLIBNAME, &luaopen_string, 1);
        luaL_requiref(state, LUA_TABLIBNAME, &luaopen_table, 1);
        luaL_requiref(state, LUA_MATHLIBNAME, &luaopen_math, 1);
        lua_pop(state, 4);

        // Nothing else may reach the string metatable through getmetatable("").
        lua_pushliteral(state, "");
        if (lua_getmetatable(state, -1) != 0) {
            lua_pushboolean(state, 0);
            lua_setfield(state, -2, "__metatable");
            lua_pop(state, 1);
        }
        lua_pop(state, 1);

        register_metatables(state);

        lua_createtable(state, 0, 32);
        int const sandbox = lua_gettop(state);

        lua_pushglobaltable(state);
        int const globals = lua_gettop(state);
        for (auto const *const name: copied_base_names) {
            lua_getfield(state, globals, name);
            lua_setfield(state, sandbox, name);
        }
        lua_pop(state, 1);

        lua_pushcfunction(state, &sandbox_print);
        lua_setfield(state, sandbox, "print");

        for (auto const *const name: read_only_libraries) {
            lua_getglobal(state, name);
            make_read_only(state, -1, name, false);
            lua_setfield(state, sandbox, name);
            lua_pop(state, 1);
        }

        push_vec3_usertype(state);
        make_read_only(state, -1, "Vec3", true);
        lua_setfield(state, sandbox, "Vec3");
        lua_pop(state, 1);

        lua_createtable(state, 0, 1);
        lua_pushcfunction(state, &scene_get_entity);
        lua_setfield(state, -2, "get_entity");
        make_read_only(state, -1, "scene", false);
        lua_setfield(state, sandbox, "scene");
        lua_pop(state, 1);
    }

    auto deadline_hook(lua_State *state, lua_Debug *activation) -> void {
        auto &context = run_context(state);
        if (std::chrono::steady_clock::now() < context.deadline) {
            return;
        }
        if (!context.timed_out) {
            context.timed_out = true;
            if (lua_getinfo(state, "Sl", activation) != 0 && activation->currentline > 0) {
                context.timeout_line = activation->currentline;
            }
            // From now on every instruction raises, so `while true do pcall(f) end` can't swallow the timeout: the
            // loop's own instructions between pcalls raise outside any pcall.
            lua_sethook(state, &deadline_hook, LUA_MASKCOUNT, 1);
        }
        // The text is unused; classify() keys off context.timed_out.
        luaL_error(state, "[timeout]");
    }

    // No traceback, which keeps the allocation small; sol2's default handler is replaced because it holds a
    // std::string across luaL_traceback.
    auto message_handler(lua_State *state) -> int {
        auto &context = run_context(state);
        if (context.budget != nullptr) {
            context.budget->in_error_handler = true;
        }

        if (lua_isstring(state, 1) != 0) {
            lua_pushvalue(state, 1);
            lua_tostring(state, -1);
        } else if (luaL_callmeta(state, 1, "__tostring") == 0 || lua_type(state, -1) != LUA_TSTRING) {
            lua_pushfstring(state, "(error object is a %s value)", luaL_typename(state, 1));
        }

        if (context.budget != nullptr) {
            context.budget->in_error_handler = false;
        }
        return 1;
    }

    // Runs inside the protected call: gives the chunk at index 1 a fresh _ENV that reads through to the sandbox,
    // so globals one run assigns are gone in the next, and calls it.
    auto run_chunk_entry(lua_State *state) -> int {
        luaL_checktype(state, 1, LUA_TFUNCTION);

        lua_createtable(state, 0, 0);
        lua_createtable(state, 0, 2);
        lua_rawgeti(state, LUA_REGISTRYINDEX, run_context(state).sandbox_ref);
        lua_setfield(state, -2, "__index");
        lua_pushboolean(state, 0);
        lua_setfield(state, -2, "__metatable");
        lua_setmetatable(state, -2);

        // Upvalue 1 of a main chunk is _ENV.
        static_cast<void>(lua_setupvalue(state, 1, 1));

        lua_settop(state, 1);
        lua_call(state, 0, 0);
        return 0;
    }

    auto bound_member_names() noexcept -> std::span<std::string_view const> { return member_names; }
} // namespace scripting::detail
