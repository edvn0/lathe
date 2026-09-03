#pragma once

#include <BS_thread_pool.hpp>

// Process-wide worker pool for CPU-heavy background work (glTF parsing/LOD
// generation, texture transcoding, terrain chunk generation, ...). A free
// function rather than a method on whichever subsystem used to own it
// (formerly Renderer::thread_pool()), so every subsystem can submit work to
// it without depending on that subsystem's type.
[[nodiscard]]
auto thread_pool() noexcept -> BS::priority_thread_pool &;
