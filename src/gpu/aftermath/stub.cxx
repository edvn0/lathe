#include "gpu/gpu_crash_dumps.hxx"

#include "core/logger.hxx"

// Built without the Nsight Aftermath SDK (AFTERMATH_SDK_DIR was not set or does not hold it): nothing is collected.
namespace gpu_crash_dumps {

    auto start(Settings settings) noexcept -> void {
        if (settings.mode != Mode::off) {
            debug("GPU crash dumps: this build has no Nsight Aftermath SDK (see docs/gpu-crash-dumps.md)");
        }
    }

    auto register_shader(void const *, std::size_t, std::string_view) noexcept -> void {}

    auto prefers_spirv() noexcept -> bool { return false; }

    auto device_flags() noexcept -> std::uint32_t { return 0; }

    auto active() noexcept -> bool { return false; }

    auto collect() noexcept -> void {}

    auto stop() noexcept -> void {}

}
