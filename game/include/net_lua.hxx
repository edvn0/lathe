#pragma once

#include "scripting/lua_api.hxx"

// Lua module `native.net`: a GameClient for lathe-server, polled from the script's on_update.
//
//   local client = require("native.net").new()
//   client:connect("ws://127.0.0.1:9002")      new session, async
//   client:send({ type = "create_room", game = "chess" })   tables become JSON; false if not open
//   for _, event in ipairs(client:poll()) do
//       -- event.kind is "open", "message" (event.message is the decoded table) or "close" (event.reason)
//   end
//   client:reconnect(); client:resume()        after a close: redial, then take the old seat back
//   client:forget_session(); client:close(); client:is_open(); client:token()
auto luaopen_lathe_net(lua_State *state) -> int;
