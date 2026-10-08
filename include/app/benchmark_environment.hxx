#pragma once

#include "app/benchmark.hxx"

auto probe_host_environment(BenchmarkEnvironment &environment) -> void;

[[nodiscard]] auto sample_thermals() -> ThermalSample;

auto add_environment_warnings(BenchmarkEnvironment &environment) -> void;
