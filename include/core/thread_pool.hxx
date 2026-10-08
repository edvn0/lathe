#pragma once

#include <BS_thread_pool.hpp>

[[nodiscard]]
auto thread_pool() noexcept -> BS::priority_thread_pool &;
