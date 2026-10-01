#pragma once

// Lua is compiled as C (see the root CMakeLists.txt), so its errors are setjmp/longjmp.
extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}
