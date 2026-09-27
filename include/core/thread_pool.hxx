#pragma once

#include <BS_thread_pool.hpp>

// Process-wide pool for CPU-heavy background work (model loading, textures, terrain generation).
[[nodiscard]]
auto thread_pool() noexcept -> BS::priority_thread_pool &;
