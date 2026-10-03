#include "app/benchmark_environment.hxx"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

#ifndef LATHE_BUILD_TYPE
#define LATHE_BUILD_TYPE ""
#endif

#ifndef LATHE_SOURCE_DIR
#define LATHE_SOURCE_DIR ""
#endif

namespace {

    [[nodiscard]] auto read_first_line(std::filesystem::path const &path) -> std::string {
        std::ifstream file{path};
        std::string line;
        if (file) {
            std::getline(file, line);
        }
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        return line;
    }

    [[nodiscard]] auto read_number(std::filesystem::path const &path) -> std::optional<double> {
        auto const line = read_first_line(path);
        if (line.empty()) {
            return std::nullopt;
        }
        auto value = 0.0;
        auto const [end, error] = std::from_chars(line.data(), line.data() + line.size(), value);
        if (error != std::errc{}) {
            return std::nullopt;
        }
        return value;
    }

    [[nodiscard]] auto cpu_model_name() -> std::string {
        std::ifstream file{"/proc/cpuinfo"};
        std::string line;
        while (std::getline(file, line)) {
            if (line.starts_with("model name")) {
                auto const colon = line.find(':');
                if (colon != std::string::npos) {
                    auto name = line.substr(colon + 1);
                    name.erase(0, name.find_first_not_of(' '));
                    return name;
                }
            }
        }
        return {};
    }

    [[nodiscard]] auto operating_system() -> std::string {
#if defined(__linux__)
        utsname name{};
        if (uname(&name) == 0) {
            return std::format("{} {} {}", name.sysname, name.release, name.machine);
        }
#endif
        return {};
    }

    // "<branch>@<sha>" of the source tree the binary was configured from, read from its .git directory now. It is
    // the checkout the user rebuilt from, as long as they did rebuild.
    [[nodiscard]] auto source_revision() -> std::string {
        std::filesystem::path const source_dir{LATHE_SOURCE_DIR};
        if (source_dir.empty()) {
            return {};
        }

        auto const git_dir = source_dir / ".git";
        auto const head = read_first_line(git_dir / "HEAD");
        if (head.empty()) {
            return {};
        }

        constexpr std::string_view ref_prefix = "ref: ";
        if (!head.starts_with(ref_prefix)) {
            return head.substr(0, 12); // detached
        }

        auto const ref = head.substr(ref_prefix.size());
        auto const branch = ref.starts_with("refs/heads/") ? ref.substr(11) : ref;

        auto sha = read_first_line(git_dir / ref);
        if (sha.empty()) {
            std::ifstream packed{git_dir / "packed-refs"};
            std::string line;
            while (std::getline(packed, line)) {
                if (line.ends_with(" " + ref)) {
                    sha = line.substr(0, line.find(' '));
                    break;
                }
            }
        }

        return sha.empty() ? branch : std::format("{}@{}", branch, sha.substr(0, 12));
    }

    [[nodiscard]] auto timestamp_utc() -> std::string {
        auto const now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        return std::format("{:%Y-%m-%dT%H:%M:%SZ}", now);
    }

} // namespace

auto probe_host_environment(BenchmarkEnvironment &environment) -> void {
    environment.cpu_name = cpu_model_name();
    environment.cpu_threads = std::thread::hardware_concurrency();
    environment.cpu_governor = read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    environment.os = operating_system();
    environment.timestamp_utc = timestamp_utc();
    environment.build_type = LATHE_BUILD_TYPE;
    environment.source_revision = source_revision();
#if defined(LATHE_TRACK_MEMORY) && LATHE_TRACK_MEMORY
    environment.memory_tracking = true;
#else
    environment.memory_tracking = false;
#endif
}

auto sample_thermals() -> ThermalSample {
    ThermalSample sample;

    std::error_code error;
    for (auto const &entry: std::filesystem::directory_iterator{"/sys/class/hwmon", error}) {
        auto const &directory = entry.path();
        auto const name = read_first_line(directory / "name");

        auto const is_gpu = name == "amdgpu" || name == "radeon" || name == "nouveau" || name == "i915" || name == "xe";
        auto const is_cpu = name == "k10temp" || name == "coretemp" || name == "zenpower" || name == "cpu_thermal";

        if (is_gpu && !sample.gpu_temperature_c) {
            if (auto const millidegrees = read_number(directory / "temp1_input")) {
                sample.gpu_temperature_c = static_cast<float>(*millidegrees / 1000.0);
            }
            auto microwatts = read_number(directory / "power1_average");
            if (!microwatts) {
                microwatts = read_number(directory / "power1_input");
            }
            if (microwatts) {
                sample.gpu_power_w = static_cast<float>(*microwatts / 1'000'000.0);
            }
            if (auto const hertz = read_number(directory / "freq1_input")) {
                sample.gpu_clock_mhz = static_cast<float>(*hertz / 1'000'000.0);
            }
        }

        if (is_cpu && !sample.cpu_temperature_c) {
            if (auto const millidegrees = read_number(directory / "temp1_input")) {
                sample.cpu_temperature_c = static_cast<float>(*millidegrees / 1000.0);
            }
        }
    }

    // Mean current frequency over all CPUs.
    auto total_khz = 0.0;
    auto cpus = 0;
    for (auto const &entry: std::filesystem::directory_iterator{"/sys/devices/system/cpu", error}) {
        auto const file_name = entry.path().filename().string();
        if (!file_name.starts_with("cpu") || file_name.size() <= 3 ||
            !std::all_of(file_name.begin() + 3, file_name.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            continue;
        }
        if (auto const khz = read_number(entry.path() / "cpufreq" / "scaling_cur_freq")) {
            total_khz += *khz;
            ++cpus;
        }
    }
    if (cpus > 0) {
        sample.cpu_clock_mhz = static_cast<float>(total_khz / cpus / 1000.0);
    }

    return sample;
}

auto add_environment_warnings(BenchmarkEnvironment &environment) -> void {
    auto &warnings = environment.warnings;

    if (!environment.cpu_governor.empty() && environment.cpu_governor != "performance") {
        warnings.push_back(std::format("CPU governor is '{}', not 'performance': clocks ramp with load, which adds "
                                       "noise and skews CPU timings",
                                       environment.cpu_governor));
    }

    if (!environment.requested_present_mode.empty() && environment.requested_present_mode != environment.present_mode) {
        warnings.push_back(
                std::format("present mode '{}' was requested but the surface doesn't offer it; ran with '{}'",
                            environment.requested_present_mode, environment.present_mode));
    }

    if (environment.present_mode == "fifo" || environment.present_mode == "fifo_relaxed") {
        warnings.emplace_back("presentation is vsynced (FIFO): displayed intervals are capped at the refresh rate, so "
                              "they show whether frames make it, not how fast they could be");
    }

    if (environment.device_name.find("llvmpipe") != std::string::npos || environment.device_type == "cpu") {
        warnings.emplace_back("rendering on a software rasterizer (lavapipe): timings don't reflect a GPU");
    }

    if (environment.build_type == "Debug") {
        warnings.emplace_back("Debug build: CPU timings are far from an optimised build's");
    }

    if (environment.memory_tracking) {
        warnings.emplace_back("memory tracking is on (non-Release build): every allocation is counted, which costs a "
                              "little CPU time");
    }
}
