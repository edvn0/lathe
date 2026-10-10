#include <doctest/doctest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

#include "app/lua_game.hxx"
#include "app/lua_game_api.hxx"
#include "scripting/lua_runtime.hxx"

// The LuaLS stubs in lua-api/ must name exactly what open_lua_game_api registers.

namespace {
    constexpr auto libraries = std::array{"scene", "assets", "camera", "ui", "game", "entity", "compute", "key", "mouse"};

    // Methods live in a metatable's __index; the stubs call the classes by these names.
    struct Class {
        char const *metatable;
        char const *stub_name;
    };
    constexpr auto classes = std::array{Class{"lathe.Entity", "Entity"}, Class{"lathe.Effect", "Effect"}};

    void collect_keys(lua_State *state, int table, std::string const &owner, std::set<std::string> &into) {
        table = lua_absindex(state, table);
        lua_pushnil(state);
        while (lua_next(state, table) != 0) {
            if (lua_type(state, -2) == LUA_TSTRING) {
                into.insert(owner + "." + lua_tostring(state, -2));
            }
            lua_pop(state, 1);
        }
    }

    auto registered_names() -> std::set<std::string> {
        auto created = LuaRuntime::create();
        REQUIRE(created.has_value());
        auto host = LuaGameHost{};
        created->set_host(&host);
        open_lua_game_api(*created);

        auto *const state = created->state();
        auto names = std::set<std::string>{};
        for (auto const *library: libraries) {
            lua_getglobal(state, library);
            REQUIRE(lua_istable(state, -1));
            collect_keys(state, -1, library, names);
            lua_pop(state, 1);
        }
        for (auto const &[metatable, stub_name]: classes) {
            REQUIRE(luaL_getmetatable(state, metatable) == LUA_TTABLE);
            lua_getfield(state, -1, "__index");
            REQUIRE(lua_istable(state, -1));
            collect_keys(state, -1, stub_name, names);
            lua_pop(state, 2);
        }
        return names;
    }

    // `function owner.name(`, `function Owner:name(` and `owner.NAME = value` lines, as "owner.name". Only owners the
    // runtime has (libraries and the two classes) count; other stub names (the LatheGame callbacks) are documentation.
    auto stub_names() -> std::set<std::string> {
        auto owners = std::set<std::string>{libraries.begin(), libraries.end()};
        for (auto const &entry: classes) {
            owners.insert(entry.stub_name);
        }

        auto names = std::set<std::string>{};
        auto const directory = std::filesystem::path{TEST_ASSETS_DIR} / "lua-api";
        REQUIRE(std::filesystem::is_directory(directory));

        for (auto const &file: std::filesystem::directory_iterator{directory}) {
            auto input = std::ifstream{file.path()};
            for (auto line = std::string{}; std::getline(input, line);) {
                auto declaration = std::string_view{line};
                auto const is_function = declaration.starts_with("function ");
                if (is_function) {
                    declaration.remove_prefix(9);
                }
                auto const separator = declaration.find_first_of(is_function ? ".:" : ".");
                auto const end = declaration.find_first_of(is_function ? "(" : " =");
                if (separator == std::string_view::npos || end == std::string_view::npos || end < separator) {
                    continue;
                }
                auto const owner = std::string{declaration.substr(0, separator)};
                auto const name = std::string{declaration.substr(separator + 1, end - separator - 1)};
                if (owners.contains(owner) && !name.empty() && name.find_first_of(" .") == std::string::npos) {
                    names.insert(owner + "." + name);
                }
            }
        }
        return names;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("every function and constant the runtime registers is in a lua-api stub, and the stubs name nothing else") {
        auto const registered = registered_names();
        auto const documented = stub_names();
        CHECK(registered.size() > 40);

        for (auto const &name: registered) {
            INFO("registered but not in lua-api/: " << name);
            CHECK(documented.contains(name));
        }
        for (auto const &name: documented) {
            INFO("in lua-api/ but not registered: " << name);
            CHECK(registered.contains(name));
        }
    }
}
