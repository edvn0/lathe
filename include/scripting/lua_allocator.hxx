#pragma once

#include <cstddef>

struct LuaMemoryBudget {
    std::size_t limit_bytes = 64ULL * 1024 * 1024;
    // Extra room while the message handler runs, so reporting an out-of-memory error can still allocate.
    std::size_t error_handler_headroom_bytes = 256ULL * 1024;
    std::size_t used_bytes = 0;
    std::size_t peak_bytes = 0;
    bool in_error_handler = false;
    // A growth request was refused since the last reset_run_flags().
    bool limit_hit = false;

    auto reset_run_flags() noexcept -> void {
        in_error_handler = false;
        limit_hit = false;
        peak_bytes = used_bytes;
    }
};

// lua_Alloc. `user_data` is a LuaMemoryBudget*. Frees and shrinks never fail (Lua 5.4 assumes osize >= nsize can't).
[[nodiscard]] auto lua_budget_alloc(void *user_data, void *block, std::size_t old_size, std::size_t new_size) noexcept
        -> void *;
