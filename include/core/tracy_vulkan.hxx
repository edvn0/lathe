#pragma once

// Tracy's Vulkan header, with the one GCC warning it triggers silenced here instead of for the whole build: GCC reports
// `tgpu` in VkCtx::Collect as maybe-uninitialized wherever it is inlined, system header or not.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <tracy/TracyVulkan.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
