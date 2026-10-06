#include "core/paths.hxx"

#include <cstdlib>
#include <system_error>
#include <utility>

namespace {
    auto executable_directory() -> std::filesystem::path {
        std::error_code error;
        auto const exe = std::filesystem::read_symlink("/proc/self/exe", error);

        return error ? std::filesystem::path{} : exe.parent_path();
    }

    auto environment_directory(char const *name) -> std::optional<std::filesystem::path> {
        auto const *value = std::getenv(name);

        if (value == nullptr || *value == '\0') {
            return std::nullopt;
        }

        return std::filesystem::path{value};
    }

    auto home_relative(char const *xdg_variable, char const *fallback) -> std::filesystem::path {
        if (auto const xdg = environment_directory(xdg_variable)) {
            return *xdg / "lathe";
        }

        if (auto const home = environment_directory("HOME")) {
            return *home / fallback / "lathe";
        }

        return std::filesystem::path{fallback} / "lathe";
    }

    auto holds_manifest(std::filesystem::path const &directory) -> bool {
        std::error_code error;

        return std::filesystem::exists(directory / "game.toml", error);
    }
} // namespace

namespace {
    auto current_storage() -> std::optional<Paths> & {
        static std::optional<Paths> storage;

        return storage;
    }
} // namespace

namespace paths_detail {
    auto sanitise(std::string_view relative) -> std::optional<std::filesystem::path> {
        auto path = std::filesystem::path{relative}.lexically_normal();

        if (path.is_absolute() || path.has_root_name()) {
            return std::nullopt;
        }

        if (!path.empty() && *path.begin() == "..") {
            return std::nullopt;
        }

        return path;
    }
} // namespace paths_detail

auto Paths::resolve(PathsOptions const &options) -> Paths {
    auto paths = Paths{};
    std::error_code error;

    if (options.data_dir) {
        paths.data_ = std::filesystem::absolute(*options.data_dir, error).lexically_normal();
        paths.installed_ = true;
    } else if (auto const exe = executable_directory(); !exe.empty()) {
        for (auto const &candidate: {exe / "data", exe / ".." / "share" / "lathe"}) {
            if (holds_manifest(candidate)) {
                paths.data_ = candidate.lexically_normal();
                paths.installed_ = true;
                break;
            }
        }
    }

    if (paths.installed_) {
        paths.cache_ = home_relative("XDG_CACHE_HOME", ".cache");
        paths.state_ = home_relative("XDG_STATE_HOME", ".local/state");
        paths.screenshots_ = paths.state_ / "screenshots";
    } else {
        auto const working_directory = std::filesystem::current_path(error);

        paths.data_ = working_directory;
        paths.cache_ = working_directory / "cache";
        paths.state_ = working_directory;
        paths.screenshots_ = working_directory / "screenshots";
    }

    return paths;
}

auto Paths::data(std::string_view relative) const -> std::optional<DataPath> {
    auto const path = paths_detail::sanitise(relative);

    return path ? std::optional{DataPath{data_, *path}} : std::nullopt;
}

auto Paths::cache(std::string_view relative) const -> std::optional<CachePath> {
    auto const path = paths_detail::sanitise(relative);

    return path ? std::optional{CachePath{cache_, *path}} : std::nullopt;
}

auto Paths::state(std::string_view relative) const -> std::optional<StatePath> {
    auto const path = paths_detail::sanitise(relative);

    return path ? std::optional{StatePath{state_, *path}} : std::nullopt;
}

auto Paths::screenshot(std::string_view relative) const -> std::optional<ScreenshotPath> {
    auto const path = paths_detail::sanitise(relative);

    return path ? std::optional{ScreenshotPath{screenshots_, *path}} : std::nullopt;
}

auto Paths::set_current(Paths paths) -> void { current_storage() = std::move(paths); }

auto Paths::current() -> Paths const & {
    auto &storage = current_storage();

    if (!storage) {
        storage = resolve({});
    }

    return *storage;
}

auto data_path(std::string_view relative) -> DataPath { return Paths::current().data(relative).value(); }

auto cache_path(std::string_view relative) -> CachePath { return Paths::current().cache(relative).value(); }

auto state_path(std::string_view relative) -> StatePath { return Paths::current().state(relative).value(); }

auto screenshot_path(std::string_view relative) -> ScreenshotPath {
    return Paths::current().screenshot(relative).value();
}
