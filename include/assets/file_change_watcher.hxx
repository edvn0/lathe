#pragma once

#include <filesystem>
#include <memory>
#include <span>

class FileChangeQueue;

namespace efsw {
    class FileWatcher;
}

class FileChangeWatcher {
public:
    FileChangeWatcher();
    ~FileChangeWatcher();

    FileChangeWatcher(FileChangeWatcher const &) = delete;
    auto operator=(FileChangeWatcher const &) -> FileChangeWatcher & = delete;

    FileChangeWatcher(FileChangeWatcher &&other) noexcept;
    auto operator=(FileChangeWatcher &&other) noexcept -> FileChangeWatcher &;

    [[nodiscard]]
    auto start(FileChangeQueue &change_queue, std::span<std::filesystem::path const> directories) -> bool;

    auto stop() noexcept -> void;

private:
    struct Listener;

    std::unique_ptr<efsw::FileWatcher> watcher_;
    std::unique_ptr<Listener> listener_;
};
