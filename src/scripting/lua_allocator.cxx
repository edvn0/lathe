#include "scripting/lua_allocator.hxx"

#include <algorithm>
#include <cstdlib>

auto lua_budget_alloc(void *user_data, void *block, std::size_t old_size, std::size_t new_size) noexcept -> void * {
    auto &budget = *static_cast<LuaMemoryBudget *>(user_data);
    // block == nullptr: old_size is an object type tag (LUA_TSTRING, LUA_TTABLE, ...), not a size.
    std::size_t const current = block != nullptr ? old_size : 0;

    if (new_size == 0) {
        std::free(block);
        budget.used_bytes -= current;
        return nullptr;
    }

    // Shrinking must not fail.
    if (new_size <= current) {
        void *const shrunk = std::realloc(block, new_size);
        if (shrunk == nullptr) {
            // The old block is still valid and big enough.
            return block;
        }
        budget.used_bytes -= current - new_size;
        return shrunk;
    }

    std::size_t const growth = new_size - current;
    std::size_t const cap = budget.limit_bytes + (budget.in_error_handler ? budget.error_handler_headroom_bytes : 0);
    // Written so neither side can overflow.
    if (budget.used_bytes > cap || growth > cap - budget.used_bytes) {
        budget.limit_hit = true;
        // Lua runs an emergency full GC and retries once (lmem.c tryagain), then raises LUA_ERRMEM.
        return nullptr;
    }

    // A separate variable, not block = realloc(block, ...): bugprone-suspicious-realloc-usage.
    void *const grown = std::realloc(block, new_size);
    if (grown == nullptr) {
        // Real out-of-memory; Lua keeps `block`.
        return nullptr;
    }
    budget.used_bytes += growth;
    budget.peak_bytes = std::max(budget.peak_bytes, budget.used_bytes);
    return grown;
}
