#include "gpu/gpu_crash_dumps.hxx"

#include <dlfcn.h>
#include <volk.h>
#include <vulkan/vulkan.h> // GFSDK_Aftermath_GetShaderHashSpirv is declared only once this is included

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <mutex>
#include <iterator>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include "core/logger.hxx"
#include "gpu/device_checkpoints.hxx"

// The Nsight Aftermath library is loaded when the process starts it (dlopen), not linked: a build with the SDK still runs
// on a machine that does not have the library, it just collects nothing there.
namespace {

    // The flags of VkDeviceDiagnosticsConfigFlagsNV (VK_NV_device_diagnostics_config).
    constexpr std::uint32_t flag_shader_debug_info = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV;
    constexpr std::uint32_t flag_resource_tracking = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV;
    constexpr std::uint32_t flag_automatic_checkpoints = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV;
    constexpr std::uint32_t flag_shader_error_reporting = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;

    struct State {
        void *library = nullptr;
        PFN_GFSDK_Aftermath_EnableGpuCrashDumps enable = nullptr;
        PFN_GFSDK_Aftermath_DisableGpuCrashDumps disable = nullptr;
        PFN_GFSDK_Aftermath_GetCrashDumpStatus status = nullptr;
        PFN_GFSDK_Aftermath_GetShaderDebugInfoIdentifier identifier = nullptr;
        PFN_GFSDK_Aftermath_GpuCrashDump_CreateDecoder create_decoder = nullptr;
        PFN_GFSDK_Aftermath_GpuCrashDump_DestroyDecoder destroy_decoder = nullptr;
        PFN_GFSDK_Aftermath_GpuCrashDump_GenerateJSON generate_json = nullptr;
        PFN_GFSDK_Aftermath_GpuCrashDump_GetJSON get_json = nullptr;
        PFN_GFSDK_Aftermath_GetShaderHashSpirv shader_hash = nullptr;
        std::mutex names_mutex;
        std::unordered_map<std::uint64_t, std::string> shader_names; // SPIR-V hash to the engine's name for the shader
        gpu_crash_dumps::Settings settings;
        std::uint32_t flags = 0;
        std::atomic_bool active{false};
        std::atomic_uint32_t dumps{0};
        std::mutex write_mutex;
    };

    State state; // NOLINT: process-wide, set once before the device exists; the callbacks run on driver threads.

    auto load_library() -> void * {
        // The path the SDK had at build time first (the build container mounts it where the host has it), then the loader's.
#ifdef LATHE_AFTERMATH_LIB
        if (void *library = dlopen(LATHE_AFTERMATH_LIB, RTLD_NOW | RTLD_LOCAL)) {
            return library;
        }
#endif
        return dlopen("libGFSDK_Aftermath_Lib.x64.so", RTLD_NOW | RTLD_LOCAL);
    }

    template<typename F>
    auto symbol(void *library, char const *name, F &target) -> bool {
        target = reinterpret_cast<F>(dlsym(library, name));
        return target != nullptr;
    }

    auto stamp() -> std::string {
        auto const now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        return std::format("{:%Y%m%d-%H%M%S}", now);
    }

    auto write_file(std::filesystem::path const &path, void const *data, std::size_t size) -> bool {
        std::ofstream out{path, std::ios::binary};
        out.write(static_cast<char const *>(data), static_cast<std::streamsize>(size));
        return static_cast<bool>(out);
    }

    // Shader debug information the callbacks stored earlier, by the identifier the decoder asks for.
    void GFSDK_AFTERMATH_CALL lookup_shader_debug_info(GFSDK_Aftermath_ShaderDebugInfoIdentifier const *id,
                                                        PFN_GFSDK_Aftermath_SetData set, void *) {
        auto const path = state.settings.directory / std::format("shader-{:016x}{:016x}.nvdbg", id->id[0], id->id[1]);
        std::ifstream in{path, std::ios::binary};
        if (!in) {
            return;
        }
        std::vector<char> data{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
        set(data.data(), static_cast<std::uint32_t>(data.size()));
    }

    void GFSDK_AFTERMATH_CALL lookup_nothing_binary(GFSDK_Aftermath_ShaderBinaryHash const *, PFN_GFSDK_Aftermath_SetData,
                                                    void *) {}

    void GFSDK_AFTERMATH_CALL lookup_nothing_source(GFSDK_Aftermath_ShaderDebugName const *, PFN_GFSDK_Aftermath_SetData,
                                                    void *) {}

    // The text after `"key": ` in the JSON, from `from` on: a number, or a string without its quotes.
    auto value_after(std::string const &json, std::string_view key, std::size_t &from) -> std::optional<std::string> {
        auto const needle = std::format("\"{}\": ", key);
        auto const at = json.find(needle, from);
        if (at == std::string::npos) {
            return std::nullopt;
        }
        auto begin = at + needle.size();
        auto const quoted = begin < json.size() && json[begin] == '"';
        if (quoted) {
            ++begin;
        }
        auto end = begin;
        while (end < json.size() && (quoted ? json[end] != '"' : json[end] != ',' && json[end] != '\n' && json[end] != '}')) {
            ++end;
        }
        from = end;
        return json.substr(begin, end - begin);
    }

    auto shader_name_of(std::string_view hash) -> std::string {
        std::uint64_t value = 0;
        std::from_chars(hash.data(), hash.data() + hash.size(), value);
        std::scoped_lock const lock{state.names_mutex};
        auto const found = state.shader_names.find(value);
        return found != state.shader_names.end() ? found->second : std::string{"unknown"};
    }

    // What matters from a decoded dump, in the log: the fault, the address, and the shaders by the engine's names.
    auto log_summary(std::string const &json) -> void {
        auto at = std::size_t{0};
        if (auto const access = value_after(json, "Access Type", at)) {
            auto const fault = value_after(json, "Fault Type", at);
            auto const address = value_after(json, "GPU virtual address", at);
            error("GPU crash: page fault ({} access): {}, address {}", *access, fault.value_or("?"), address.value_or("?"));
        }
        // "Shader infos": the hash and the name the dump gives each shader that was active.
        std::unordered_map<std::string, std::string> engine_name_of_dump_name;
        at = 0;
        while (true) {
            auto const hash = value_after(json, "Shader hash", at);
            if (!hash) {
                break;
            }
            auto const dump_name = value_after(json, "Shader name", at);
            if (dump_name) {
                auto const name = shader_name_of(*hash);
                engine_name_of_dump_name[*dump_name] = name;
                error("GPU crash: active shader {} is '{}'", *dump_name, name);
            }
        }
        at = 0;
        while (true) {
            auto const fault = value_after(json, "Fault Name", at);
            if (!fault) {
                break;
            }
            auto const where = value_after(json, "Shader GPU PC Address", at);
            auto shader = where.value_or("?");
            if (auto const space = shader.find(' '); space != std::string::npos) {
                if (auto const found = engine_name_of_dump_name.find(shader.substr(0, space));
                    found != engine_name_of_dump_name.end()) {
                    shader = std::format("{} ({})", shader, found->second);
                }
            }
            error("GPU crash: {} in {}", *fault, shader);
        }
    }

    // The dump as JSON next to it: the faulting shader, the page fault, the checkpoints, the warps' state.
    auto write_json(std::filesystem::path const &dump_path, void const *dump, std::uint32_t size) -> void {
        if (state.create_decoder == nullptr || state.generate_json == nullptr || state.get_json == nullptr) {
            return;
        }
        GFSDK_Aftermath_GpuCrashDump_Decoder decoder{};
        if (!GFSDK_Aftermath_SUCCEED(state.create_decoder(GFSDK_Aftermath_Version_API, dump, size, &decoder))) {
            warn("GPU crash dump: it could not be decoded");
            return;
        }
        std::uint32_t json_size = 0;
        auto const generated = state.generate_json(
                decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO, GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT,
                lookup_shader_debug_info, lookup_nothing_binary, lookup_nothing_source, nullptr, &json_size);
        if (GFSDK_Aftermath_SUCCEED(generated) && json_size > 0) {
            std::vector<char> json(json_size);
            if (GFSDK_Aftermath_SUCCEED(state.get_json(decoder, json_size, json.data()))) {
                log_summary(std::string{json.data(), json_size > 0 ? json_size - 1 : 0});
                auto json_path = dump_path;
                json_path.replace_extension(".json");
                if (write_file(json_path, json.data(), json_size > 0 ? json_size - 1 : 0)) {
                    error("GPU crash dump decoded to {}", json_path.string());
                }
            }
        }
        state.destroy_decoder(decoder);
    }

    // Called by the driver's crash handling thread once the dump is complete.
    void GFSDK_AFTERMATH_CALL on_crash_dump(void const *dump, std::uint32_t const size, void *) {
        std::scoped_lock const lock{state.write_mutex};
        auto const index = state.dumps.fetch_add(1);
        auto const path = state.settings.directory / std::format("{}-{}-{}.nv-gpudmp", state.settings.application,
                                                                   stamp(), index);
        if (write_file(path, dump, size)) {
            error("GPU crash dump written to {} ({} bytes); open it with Nsight Graphics", path.string(), size);
            write_json(path, dump, size);
        } else {
            error("GPU crash dump could not be written to {}", path.string());
        }
    }

    void GFSDK_AFTERMATH_CALL on_shader_debug_info(void const *info, std::uint32_t const size, void *) {
        GFSDK_Aftermath_ShaderDebugInfoIdentifier id{};
        if (state.identifier == nullptr ||
            !GFSDK_Aftermath_SUCCEED(state.identifier(GFSDK_Aftermath_Version_API, info, size, &id))) {
            return;
        }
        std::scoped_lock const lock{state.write_mutex};
        auto const path = state.settings.directory / std::format("shader-{:016x}{:016x}.nvdbg", id.id[0], id.id[1]);
        if (!write_file(path, info, size)) {
            warn("Shader debug information could not be written to {}", path.string());
        }
    }

    void GFSDK_AFTERMATH_CALL on_description(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add, void *) {
        add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, state.settings.application.c_str());
        add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, state.settings.version.c_str());
        add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined,
            state.settings.mode == gpu_crash_dumps::Mode::full ? "diagnostics: full" : "diagnostics: standard");
    }

    // A checkpoint is a pointer to the name of a pass (device_checkpoints): turn it back into the name.
    void GFSDK_AFTERMATH_CALL on_resolve_marker(void const *marker, std::uint32_t const size, void *,
                                                PFN_GFSDK_Aftermath_ResolveMarker resolve) {
        if (size != 0) {
            return; // the driver's own checkpoints carry their payload
        }
        if (auto const name = device_checkpoints::resolve(marker)) {
            resolve(name->data(), static_cast<std::uint32_t>(name->size() + 1));
        }
    }

}

namespace gpu_crash_dumps {

    auto start(Settings settings) noexcept -> void {
        if (settings.mode == Mode::off || state.active.load()) {
            return;
        }
        state.library = load_library();
        if (state.library == nullptr) {
            info("GPU crash dumps: the Nsight Aftermath library was not found ({})", dlerror());
            return;
        }
        if (!symbol(state.library, "GFSDK_Aftermath_EnableGpuCrashDumps", state.enable) ||
            !symbol(state.library, "GFSDK_Aftermath_DisableGpuCrashDumps", state.disable) ||
            !symbol(state.library, "GFSDK_Aftermath_GetCrashDumpStatus", state.status) ||
            !symbol(state.library, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier", state.identifier)) {
            warn("GPU crash dumps: the Nsight Aftermath library is not the expected version");
            return;
        }

        symbol(state.library, "GFSDK_Aftermath_GetShaderHashSpirv", state.shader_hash);
        // Decoding is optional: a library without it still collects dumps.
        symbol(state.library, "GFSDK_Aftermath_GpuCrashDump_CreateDecoder", state.create_decoder);
        symbol(state.library, "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder", state.destroy_decoder);
        symbol(state.library, "GFSDK_Aftermath_GpuCrashDump_GenerateJSON", state.generate_json);
        symbol(state.library, "GFSDK_Aftermath_GpuCrashDump_GetJSON", state.get_json);
        if (state.destroy_decoder == nullptr) {
            state.create_decoder = nullptr;
        }

        std::error_code ignored;
        std::filesystem::create_directories(settings.directory, ignored);
        state.settings = std::move(settings);

        auto const full = state.settings.mode == Mode::full;
        state.flags = flag_resource_tracking;
        if (full) {
            state.flags |= flag_shader_debug_info | flag_automatic_checkpoints | flag_shader_error_reporting;
        }

        // Shader debug information only exists when the device asks for it; the library keeps it until a crash and
        // hands over the part that matters then.
        auto const result = state.enable(GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
                                         full ? GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks
                                              : GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default,
                                         on_crash_dump, full ? on_shader_debug_info : nullptr, on_description,
                                         on_resolve_marker, nullptr);
        if (!GFSDK_Aftermath_SUCCEED(result)) {
            warn("GPU crash dumps: the Nsight Aftermath library refused to start (result {:#x}); NV_AFTERMATH_FORCE_OFF set?",
                 static_cast<std::uint32_t>(result));
            state.flags = 0;
            return;
        }
        state.active.store(true);
        info("GPU crash dumps on ({}), written to {}", full ? "full: shader debug information and automatic checkpoints"
                                                          : "standard",
             state.settings.directory.string());
    }

    auto register_shader(void const *spirv, std::size_t bytes, std::string_view name) noexcept -> void {
        if (!state.active.load() || state.shader_hash == nullptr) {
            return;
        }
        GFSDK_Aftermath_SpirvCode const code{spirv, static_cast<std::uint32_t>(bytes)};
        GFSDK_Aftermath_ShaderBinaryHash hash{};
        if (GFSDK_Aftermath_SUCCEED(state.shader_hash(GFSDK_Aftermath_Version_API, &code, &hash))) {
            std::scoped_lock const lock{state.names_mutex};
            state.shader_names[hash.hash] = std::string{name};
        }
    }

    auto prefers_spirv() noexcept -> bool { return state.active.load() && state.settings.mode == Mode::full; }

    auto device_flags() noexcept -> std::uint32_t { return state.active.load() ? state.flags : 0U; }

    auto active() noexcept -> bool { return state.active.load(); }

    auto collect() noexcept -> void {
        if (!state.active.load() || state.status == nullptr) {
            return;
        }
        using namespace std::chrono;
        auto const start = steady_clock::now();
        auto status = GFSDK_Aftermath_CrashDump_Status_Unknown;
        while (steady_clock::now() - start < seconds{8}) {
            if (!GFSDK_Aftermath_SUCCEED(state.status(&status))) {
                break;
            }
            if (status == GFSDK_Aftermath_CrashDump_Status_Finished ||
                status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed) {
                break;
            }
            // Still "not started" a while after the device was lost: the driver saw no GPU crash (a lost device can also
            // be a driver reset or a removed GPU).
            if (status == GFSDK_Aftermath_CrashDump_Status_NotStarted && steady_clock::now() - start > milliseconds{2500}) {
                break;
            }
            std::this_thread::sleep_for(milliseconds{50});
        }
        switch (status) {
            case GFSDK_Aftermath_CrashDump_Status_Finished:
                break; // on_crash_dump has logged where it went
            case GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed:
                error("GPU crash dump: the driver could not collect the data");
                break;
            case GFSDK_Aftermath_CrashDump_Status_NotStarted:
                info("GPU crash dump: the driver did not report a GPU crash for this device loss");
                break;
            default:
                warn("GPU crash dump: still being collected after {} s, giving up", 8);
                break;
        }
    }

    auto stop() noexcept -> void {
        if (state.active.exchange(false) && state.disable != nullptr) {
            state.disable();
        }
    }

}
