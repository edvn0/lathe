#pragma once

#include <filesystem>
#include <memory>
#include <span>

class ShaderChangeQueue;

namespace efsw {
    class FileWatcher;
}

class ShaderHotReloadWatcher {
public:
    ShaderHotReloadWatcher();
    ~ShaderHotReloadWatcher();

    ShaderHotReloadWatcher(ShaderHotReloadWatcher const &) = delete;
    auto operator=(ShaderHotReloadWatcher const &) -> ShaderHotReloadWatcher & = delete;

    ShaderHotReloadWatcher(ShaderHotReloadWatcher &&other) noexcept;
    auto operator=(ShaderHotReloadWatcher &&other) noexcept -> ShaderHotReloadWatcher &;

    // change_queue must outlive this watcher.
    [[nodiscard]]
    auto start(ShaderChangeQueue &change_queue, std::span<std::filesystem::path const> directories) -> bool;

    auto stop() noexcept -> void;

private:
    struct Listener;

    std::unique_ptr<efsw::FileWatcher> watcher_;
    std::unique_ptr<Listener> listener_;
};
