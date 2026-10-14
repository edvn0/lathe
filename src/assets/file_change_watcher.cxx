#include "assets/file_change_watcher.hxx"

#include <utility>

#include <efsw/efsw.hpp>

#include "assets/file_change_queue.hxx"
#include "core/logger.hxx"

struct FileChangeWatcher::Listener final : efsw::FileWatchListener {
    explicit Listener(FileChangeQueue &queue) noexcept : change_queue(&queue) {}

    auto handleFileAction(efsw::WatchID , std::string const &directory, std::string const &filename,
                          efsw::Action action, std::string old_filename) -> void override {
        switch (action) {
            case efsw::Actions::Add:
            case efsw::Actions::Modified:
                change_queue->push(std::filesystem::path{directory} / filename);
                break;

            case efsw::Actions::Moved:
                change_queue->push(std::filesystem::path{directory} / old_filename);
                change_queue->push(std::filesystem::path{directory} / filename);
                break;

            default:
                break;
        }
    }

    FileChangeQueue *change_queue = nullptr;
};

FileChangeWatcher::FileChangeWatcher() = default;
FileChangeWatcher::~FileChangeWatcher() { stop(); }

FileChangeWatcher::FileChangeWatcher(FileChangeWatcher &&other) noexcept :
    watcher_(std::move(other.watcher_)), listener_(std::move(other.listener_)) {}

auto FileChangeWatcher::operator=(FileChangeWatcher &&other) noexcept -> FileChangeWatcher & {
    if (this == &other) {
        return *this;
    }

    stop();

    watcher_ = std::move(other.watcher_);
    listener_ = std::move(other.listener_);

    return *this;
}

auto FileChangeWatcher::start(FileChangeQueue &change_queue, std::span<std::filesystem::path const> directories)
        -> bool {
    stop();

    watcher_ = std::make_unique<efsw::FileWatcher>();
    listener_ = std::make_unique<Listener>(change_queue);

    auto any_watched = false;

    for (auto const &directory: directories) {
        std::error_code error_code;

        if (!std::filesystem::is_directory(directory, error_code) || error_code) {
            error("Shader hot-reload: not a directory, skipping: {}", directory.string());
            continue;
        }

        auto const watch_id = watcher_->addWatch(directory.string(), listener_.get(),  true);

        if (watch_id < 0) {
            error("Shader hot-reload: failed to watch directory: {}", directory.string());
            continue;
        }

        any_watched = true;
    }

    if (!any_watched) {
        watcher_.reset();
        listener_.reset();

        return false;
    }

    watcher_->watch();

    return true;
}

auto FileChangeWatcher::stop() noexcept -> void {
    watcher_.reset();
    listener_.reset();
}
