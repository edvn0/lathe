#pragma once

#include "app/benchmark.hxx"

// Machine state for benchmark results. Linux-specific where it reads /proc and /sys; elsewhere those fields stay
// empty. Nothing here touches Vulkan: the device fields are filled by the caller.

// CPU model and thread count, OS, CPU frequency governor, UTC time, build type and source revision.
auto probe_host_environment(BenchmarkEnvironment &environment) -> void;

// GPU and CPU temperature, power and clocks from hwmon / cpufreq, where the drivers expose them (amdgpu, i915/xe and
// nouveau do; NVIDIA's proprietary driver doesn't). Cheap enough to call between runs, not every frame.
[[nodiscard]] auto sample_thermals() -> ThermalSample;

// Appends warnings for setups that make results less trustworthy: a governor other than "performance", vsync /
// FIFO presentation, a software rasterizer, a Debug build, memory tracking on.
auto add_environment_warnings(BenchmarkEnvironment &environment) -> void;
