#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "assets/slang_compiler.hxx"
#include "core/game_manifest.hxx"

struct PackageOptions {
    // Written as data/game.toml; `name` is also the package directory and the executable's file name.
    GameManifest manifest;

    // The package is built at <output_dir>/<manifest.name> (plus <manifest.name>.tar when `archive` is set).
    std::filesystem::path output_dir;

    // The scene cooked by the editor (a .lbf); moved to data/<manifest.scene> and removed from its staging place.
    std::filesystem::path cooked_scene;

    // Where assets/ is read from, and the running executable (plus the shared libraries next to it) to ship.
    std::filesystem::path data_root;
    std::filesystem::path executable;

    // Bakes data/shaders.lsp; null or invalid falls back to the data root's own shaders.lsp.
    renderer::SlangCompiler const *compiler = nullptr;

    bool archive = false;
};

struct PackageResult {
    std::filesystem::path directory;
    std::optional<std::filesystem::path> archive;
    std::size_t files = 0;
    std::uintmax_t bytes = 0;
    double seconds = 0.0;
};

struct PackageProgress {
    float fraction = 0.0F;
    std::string step;
};

// Packages a game on a worker thread, entirely in-process (no subprocesses): bakes the shaders, copies assets/ and the
// runtime, writes game.toml and optionally a tar archive. The result is the "installed game" layout Paths::resolve
// looks for: <name>/<name> next to <name>/data/game.toml. It is built in <name>.partial and renamed into place, so a
// failed or cancelled run leaves any previous package alone.
class GamePackageJob {
public:
    [[nodiscard]] static auto start(PackageOptions options) -> GamePackageJob;

    GamePackageJob(GamePackageJob &&) noexcept = default;
    auto operator=(GamePackageJob &&) noexcept -> GamePackageJob & = default;
    GamePackageJob(GamePackageJob const &) = delete;
    auto operator=(GamePackageJob const &) -> GamePackageJob & = delete;
    ~GamePackageJob();

    [[nodiscard]] auto ready() const -> bool;
    [[nodiscard]] auto progress() const -> PackageProgress;
    auto cancel() -> void;
    [[nodiscard]] auto take() -> std::expected<PackageResult, std::string>;

private:
    struct Shared {
        mutable std::mutex mutex;
        std::string step;
        std::atomic<float> fraction{0.0F};
        std::atomic<bool> cancelled{false};
    };

    GamePackageJob() = default;

    std::shared_ptr<Shared> shared_;
    std::future<std::expected<PackageResult, std::string>> future_;
};

// Writes every regular file under `directory` to a ustar archive at `output`, paths relative to the directory's parent
// (so extracting gives <directory name>/...). Fails on names the format cannot hold and on files over 8 GiB.
[[nodiscard]] auto write_tar(std::filesystem::path const &directory, std::filesystem::path const &output)
        -> std::expected<void, std::string>;
