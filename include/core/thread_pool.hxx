#pragma once

#include <BS_thread_pool.hpp>

[[nodiscard]]
auto thread_pool() noexcept -> BS::priority_thread_pool &;

// Called when the process starts shutting down. Queued work that has not begun checks thread_pool_shutting_down() and
// returns at once, so the pool drains in the time of the tasks already running instead of the whole backlog (a scene
// that queued hundreds of textures would otherwise hold the exit until every one of them was decoded and encoded).
auto request_thread_pool_shutdown() noexcept -> void;

[[nodiscard]]
auto thread_pool_shutting_down() noexcept -> bool;

// True while any task is queued or running.
[[nodiscard]]
auto thread_pool_busy() noexcept -> bool;

// For the end of main: returns `code`, unless pool workers are still busy, in which case it flushes the log and ends the
// process on the spot. Their results would be thrown away, and the pool's destructor would otherwise join them first (a
// texture encode already under way takes seconds). Only call it once nothing those tasks touch is still needed.
auto exit_promptly(int code) noexcept -> int;
