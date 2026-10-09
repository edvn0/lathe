#include "net_lua.hxx"

#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/json.hxx"
#include "gameclient/game_client.hxx"
#include "gameclient/ix_websocket_client_transport.hxx"

namespace {
    constexpr char const *client_meta = "lathe.GameClient";

    constexpr int max_depth = 16;

    auto client_of(lua_State *state, int index) -> gameclient::GameClient & {
        return *static_cast<gameclient::GameClient *>(luaL_checkudata(state, index, client_meta));
    }

    // Lua -> JSON ----------------------------------------------------------------------------------------------------

    // A table is an array when its keys are exactly 1..n.
    auto is_sequence(lua_State *state, int index) -> bool {
        auto const length = static_cast<lua_Integer>(lua_rawlen(state, index));

        if (length == 0) {
            return false;
        }

        lua_Integer count = 0;

        lua_pushnil(state);

        while (lua_next(state, index) != 0) {
            ++count;

            if (!lua_isinteger(state, -2)) {
                lua_pop(state, 2);
                return false;
            }

            lua_pop(state, 1);
        }

        return count == length;
    }

    auto write_value(lua_State *state, int index, JsonWriter &writer, std::string_view key, int depth) -> bool {
        index = lua_absindex(state, index);

        switch (lua_type(state, index)) {
            case LUA_TBOOLEAN:
                writer.value(key, lua_toboolean(state, index) != 0);
                return true;
            case LUA_TNUMBER:
                if (lua_isinteger(state, index)) {
                    writer.value(key, static_cast<std::int64_t>(lua_tointeger(state, index)));
                } else {
                    writer.value(key, static_cast<double>(lua_tonumber(state, index)), 6);
                }
                return true;
            case LUA_TSTRING: {
                std::size_t length = 0;

                auto const *const text = lua_tolstring(state, index, &length);

                writer.value(key, std::string_view{text, length});
                return true;
            }
            case LUA_TTABLE:
                break;
            default:
                return false;
        }

        if (depth >= max_depth) {
            return false;
        }

        auto const sequence = is_sequence(state, index);

        if (sequence) {
            writer.begin_array(key, true);
        } else {
            writer.begin_object(key, true);
        }

        lua_pushnil(state);

        while (lua_next(state, index) != 0) {
            bool ok = false;

            if (sequence) {
                ok = write_value(state, -1, writer, {}, depth + 1);
            } else if (lua_type(state, -2) == LUA_TSTRING) {
                ok = write_value(state, -1, writer, lua_tostring(state, -2), depth + 1);
            }

            if (!ok) {
                lua_pop(state, 2);
                return false;
            }

            lua_pop(state, 1);
        }

        if (sequence) {
            writer.end_array();
        } else {
            writer.end_object();
        }

        return true;
    }

    auto to_json(lua_State *state, int index) -> std::optional<std::string> {
        JsonWriter writer;

        if (!write_value(state, index, writer, {}, 0)) {
            return std::nullopt;
        }

        return writer.str();
    }

    // JSON -> Lua ----------------------------------------------------------------------------------------------------

    auto push_json(lua_State *state, JsonValue const &value) -> void {
        if (value.is_bool()) {
            lua_pushboolean(state, value.as_bool() ? 1 : 0);
        } else if (value.is_number()) {
            auto const number = value.as_number();

            if (std::floor(number) == number && std::abs(number) < 9.0e15) {
                lua_pushinteger(state, static_cast<lua_Integer>(number));
            } else {
                lua_pushnumber(state, number);
            }
        } else if (value.is_string()) {
            auto const text = value.as_string();

            lua_pushlstring(state, text.data(), text.size());
        } else if (value.is_array()) {
            auto const items = value.items();

            lua_createtable(state, static_cast<int>(items.size()), 0);

            lua_Integer position = 1;

            for (auto const &item: items) {
                push_json(state, item);
                lua_rawseti(state, -2, position++);
            }
        } else if (value.is_object()) {
            auto const members = value.members();

            lua_createtable(state, 0, static_cast<int>(members.size()));

            for (auto const &member: members) {
                push_json(state, member.value);
                lua_setfield(state, -2, member.key.c_str());
            }
        } else {
            lua_pushnil(state);
        }
    }

    // Methods --------------------------------------------------------------------------------------------------------

    auto client_new(lua_State *state) -> int {
        auto *const memory = lua_newuserdatauv(state, sizeof(gameclient::GameClient), 0);

        new (memory) gameclient::GameClient{&gameclient::make_ix_websocket_client_transport};

        luaL_setmetatable(state, client_meta);

        return 1;
    }

    auto client_gc(lua_State *state) -> int {
        static_cast<gameclient::GameClient *>(lua_touserdata(state, 1))->~GameClient();

        return 0;
    }

    auto client_connect(lua_State *state) -> int {
        client_of(state, 1).connect(luaL_checkstring(state, 2));

        return 0;
    }

    auto client_reconnect(lua_State *state) -> int {
        client_of(state, 1).reconnect();

        return 0;
    }

    auto client_resume(lua_State *state) -> int {
        lua_pushboolean(state, client_of(state, 1).resume() ? 1 : 0);

        return 1;
    }

    auto client_send(lua_State *state) -> int {
        auto &client = client_of(state, 1);

        luaL_checktype(state, 2, LUA_TTABLE);

        auto text = to_json(state, 2);

        if (!text) {
            return luaL_error(state, "send: only tables of strings, numbers, booleans and tables convert to JSON");
        }

        lua_pushboolean(state, client.send(std::move(*text)) ? 1 : 0);

        return 1;
    }

    auto client_forget_session(lua_State *state) -> int {
        client_of(state, 1).forget_session();

        return 0;
    }

    auto client_close(lua_State *state) -> int {
        client_of(state, 1).close();

        return 0;
    }

    auto client_is_open(lua_State *state) -> int {
        lua_pushboolean(state, client_of(state, 1).is_open() ? 1 : 0);

        return 1;
    }

    auto client_token(lua_State *state) -> int {
        auto const token = client_of(state, 1).token();

        lua_pushlstring(state, token.data(), token.size());

        return 1;
    }

    auto client_poll(lua_State *state) -> int {
        auto const events = client_of(state, 1).drain();

        lua_createtable(state, static_cast<int>(events.size()), 0);

        lua_Integer position = 1;

        for (auto const &event: events) {
            lua_createtable(state, 0, 2);

            switch (event.kind) {
                case gameclient::ClientEvent::Kind::opened:
                    lua_pushstring(state, "open");
                    lua_setfield(state, -2, "kind");
                    break;
                case gameclient::ClientEvent::Kind::message: {
                    lua_pushstring(state, "message");
                    lua_setfield(state, -2, "kind");

                    if (auto const parsed = parse_json(event.text); parsed) {
                        push_json(state, *parsed);
                        lua_setfield(state, -2, "message");
                    }

                    break;
                }
                case gameclient::ClientEvent::Kind::closed:
                    lua_pushstring(state, "close");
                    lua_setfield(state, -2, "kind");
                    lua_pushlstring(state, event.text.data(), event.text.size());
                    lua_setfield(state, -2, "reason");
                    break;
            }

            lua_rawseti(state, -2, position++);
        }

        return 1;
    }

    constexpr std::array client_methods{
            luaL_Reg{"connect", &client_connect},
            luaL_Reg{"reconnect", &client_reconnect},
            luaL_Reg{"resume", &client_resume},
            luaL_Reg{"send", &client_send},
            luaL_Reg{"forget_session", &client_forget_session},
            luaL_Reg{"close", &client_close},
            luaL_Reg{"is_open", &client_is_open},
            luaL_Reg{"token", &client_token},
            luaL_Reg{"poll", &client_poll},
            luaL_Reg{nullptr, nullptr},
    };
}

auto luaopen_lathe_net(lua_State *state) -> int {
    luaL_newmetatable(state, client_meta);
    lua_createtable(state, 0, static_cast<int>(client_methods.size()));
    luaL_setfuncs(state, client_methods.data(), 0);
    lua_setfield(state, -2, "__index");
    lua_pushcfunction(state, &client_gc);
    lua_setfield(state, -2, "__gc");
    lua_pushstring(state, client_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    lua_createtable(state, 0, 1);
    lua_pushcfunction(state, &client_new);
    lua_setfield(state, -2, "new");

    return 1;
}
