#include "shader_hot_reload_watcher.hxx"

#include <utility>

#include <efsw/efsw.hpp>

#include "logger.hxx"
#include "renderer.hxx"

struct ShaderHotReloadWatcher::Listener final : efsw::FileWatchListener {
    explicit Listener(Renderer &r) noexcept : renderer(&r) {}

    auto handleFileAction(efsw::WatchID /*watch_id*/, std::string const &directory, std::string const &filename,
                          efsw::Action action, std::string old_filename) -> void override {
        switch (action) {
            case efsw::Actions::Add:
            case efsw::Actions::Modified:
                renderer->notify_shader_file_changed(std::filesystem::path{directory} / filename);
                break;

            case efsw::Actions::Moved:
                renderer->notify_shader_file_changed(std::filesystem::path{directory} / old_filename);
                renderer->notify_shader_file_changed(std::filesystem::path{directory} / filename);
                break;

            case efsw::Actions::Delete:
                break;

            default:
                break;
        }
    }

    Renderer *renderer = nullptr;
};

ShaderHotReloadWatcher::ShaderHotReloadWatcher() = default;
ShaderHotReloadWatcher::~ShaderHotReloadWatcher() { stop(); }

ShaderHotReloadWatcher::ShaderHotReloadWatcher(ShaderHotReloadWatcher &&other) noexcept :
    watcher_(std::move(other.watcher_)), listener_(std::move(other.listener_)) {}

auto ShaderHotReloadWatcher::operator=(ShaderHotReloadWatcher &&other) noexcept -> ShaderHotReloadWatcher & {
    if (this == &other) {
        return *this;
    }

    stop();

    watcher_ = std::move(other.watcher_);
    listener_ = std::move(other.listener_);

    return *this;
}

auto ShaderHotReloadWatcher::start(Renderer &renderer, std::span<std::filesystem::path const> directories) -> bool {
    stop();

    watcher_ = std::make_unique<efsw::FileWatcher>();
    listener_ = std::make_unique<Listener>(renderer);

    auto any_watched = false;

    for (auto const &directory: directories) {
        std::error_code error_code;

        if (!std::filesystem::is_directory(directory, error_code) || error_code) {
            error("Shader hot-reload: not a directory, skipping: {}", directory.string());
            continue;
        }

        // recursive = true: catches shaders under nested subfolders
        // (e.g. assets/shaders/post/, assets/shaders/common/).
        auto const watch_id = watcher_->addWatch(directory.string(), listener_.get(), /*recursive*/ true);

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

auto ShaderHotReloadWatcher::stop() noexcept -> void {
    watcher_.reset();
    listener_.reset();
}
