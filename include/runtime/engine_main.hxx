#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/forward.hxx"

class CommandLine;
class IGame;
struct CpuFrameTimes;

struct RenderExtent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// What an executable adds to the shared engine startup and frame loop (run_engine). The player uses the defaults; the
// editor supplies an Application and the benchmark runner.
class EngineProfile {
public:
    enum class LoopStatus : std::uint8_t {
        running,
        finished,
        failed,
    };

    virtual ~EngineProfile() = default;

    [[nodiscard]] virtual auto name() const -> std::string_view = 0;
    [[nodiscard]] virtual auto description() const -> std::string_view = 0;

    // True for an executable that is only ever an installed game.
    [[nodiscard]] virtual auto always_player() const -> bool { return false; }

    // True when the executable draws its own title bar, so a windowed window is created without the OS one.
    [[nodiscard]] virtual auto custom_title_bar() const -> bool { return false; }

    [[nodiscard]] virtual auto create_runtime(VulkanContext &context) -> std::unique_ptr<Runtime> = 0;

    // Command line: add this executable's own arguments, then (after parsing) stop early with an exit code, and
    // validate what was parsed.
    virtual auto add_arguments(CommandLine &cli) -> void { (void) cli; }
    [[nodiscard]] virtual auto early_exit() -> std::optional<int> { return std::nullopt; }
    [[nodiscard]] virtual auto prepare() -> std::expected<void, std::string> { return {}; }

    // True when frames are paced for measurement: vsync off unless asked for, immediate present, and renderer resizes
    // taking effect at once.
    [[nodiscard]] virtual auto deterministic() const -> bool { return false; }

    // Called once the runtime is started and its first events are queued, before the first frame.
    [[nodiscard]] virtual auto attach(Runtime &runtime, VulkanContext &context) -> std::expected<void, std::string> {
        (void) runtime;
        (void) context;
        return {};
    }

    // `--save-scene`: cook the opened scene. Runs on the render thread; only an editor can.
    virtual auto save_scene(Runtime &runtime, std::filesystem::path const &path) -> void;

    // Frame loop.
    [[nodiscard]] virtual auto fixed_timestep() const -> std::optional<float> { return std::nullopt; }
    virtual auto begin_frame(Runtime &runtime) -> void { (void) runtime; }
    [[nodiscard]] virtual auto end_frame(Runtime &runtime, VulkanContext const &context, CpuFrameTimes const &cpu,
                                         RenderExtent extent) -> LoopStatus {
        (void) runtime;
        (void) context;
        (void) cpu;
        (void) extent;
        return LoopStatus::running;
    }
    [[nodiscard]] virtual auto render_extent(RenderExtent desired) -> RenderExtent { return desired; }

    // True when the run ended before the profile's work finished.
    [[nodiscard]] virtual auto interrupted() const -> bool { return false; }
};

// The games an executable offers, and a factory for them. Defined by the executable's game sources
// (game/src/main_entry.cxx), so the engine libraries stay game-agnostic.
auto game_names() -> std::span<std::string_view const>;
auto create_game(std::string_view name) -> std::unique_ptr<IGame>;

// Parses the command line, brings up Vulkan and a Runtime from `profile`, and runs frames until the window closes.
// Returns the process exit code.
auto run_engine(int argc, char **argv, EngineProfile &profile) -> int;
