#include <string>
#include <utility>
#include <vector>

#include "chess_lua.hxx"
#include "net_lua.hxx"
#include "runtime/engine_main.hxx"

// The native Lua modules every script of this executable can require; kept apart from the game registry so a tool that
// runs scripts without the games (lathe-lua) offers the same ones.
auto native_script_modules() -> std::vector<std::pair<std::string, lua_CFunction>> {
    return {
            {"native.chess", &luaopen_lathe_chess},
            {"native.net", &luaopen_lathe_net},
    };
}
