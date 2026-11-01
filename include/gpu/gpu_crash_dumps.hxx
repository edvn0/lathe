#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

// GPU crash dumps through the NVIDIA Nsight Aftermath SDK (docs/gpu-crash-dumps.md): when the GPU faults or hangs, the
// driver writes a dump (the shader that was running, the page fault, the checkpoint each queue had reached) that Nsight
// Graphics opens. Compiled in only where the SDK was found at configure time (HAS_AFTERMATH); everywhere else, and on
// devices without the extensions, these do nothing.
namespace gpu_crash_dumps {

    enum class Mode : std::uint8_t {
        // Nothing is collected.
        off,
        // The crash dump and the driver's resource tracking: cheap enough to leave on.
        standard,
        // Adds shader debug information (line tables), automatic checkpoints with call stacks and extra shader error
        // reporting. Costs CPU time in command recording and shader compilation, so it is for chasing a crash: the
        // engine turns it on by itself for the run after a device-lost restart.
        full,
    };

    struct Settings {
        Mode mode = Mode::standard;
        // Where the dumps (.nv-gpudmp) and shader debug information (.nvdbg) go.
        std::filesystem::path directory;
        std::string application = "lathe";
        std::string version;
    };

    // "off", "standard", "full" or "auto" (standard, and full after a device-lost restart); nothing for anything else.
    [[nodiscard]] auto mode_from(std::string_view text, bool after_device_loss) -> std::optional<Mode>;

    // Before the device is created: loads the library and turns collection on. Without the SDK, with the mode off, or
    // when the library cannot be loaded this leaves everything off.
    auto start(Settings settings) noexcept -> void;

    // Tells the crash dump which shader is which: a dump names shaders by a hash of their SPIR-V, and this keeps the name
    // for it. Call it for every shader object made from SPIR-V; cheap, and nothing when collection is off.
    auto register_shader(void const *spirv, std::size_t bytes, std::string_view name) noexcept -> void;

    // True when shaders should be built from SPIR-V rather than a cached binary: the driver only produces debug
    // information (line tables) for shaders it compiles itself, which the full mode asks for.
    [[nodiscard]] auto prefers_spirv() noexcept -> bool;

    // The VkDeviceDiagnosticsConfigFlagsNV the device is to be created with (0: do not ask for the extension).
    [[nodiscard]] auto device_flags() noexcept -> std::uint32_t;

    [[nodiscard]] auto active() noexcept -> bool;

    // After the device was lost: gives the driver time to finish the dump (it is written from another thread), then
    // logs where it went.
    auto collect() noexcept -> void;

    auto stop() noexcept -> void;

}
