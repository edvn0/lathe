#include "core/project_file.hxx"

#include <algorithm>
#include <format>
#include <fstream>
#include <system_error>

#include "core/paths.hxx"

auto ProjectFile::load(std::filesystem::path const &path) -> std::expected<ProjectFile, std::string> {
    std::error_code error;
    auto const file = std::filesystem::is_directory(path, error) ? path / file_name : path;
    auto manifest = GameManifest::load(file, false);

    if (!manifest) {
        return std::unexpected{std::move(manifest.error())};
    }

    // lexically_normal keeps a trailing separator on a path like "dir/.", which then names the project twice.
    auto root = std::filesystem::absolute(file, error).parent_path().lexically_normal();

    if (!root.has_filename()) {
        root = root.parent_path();
    }

    return ProjectFile{.manifest = std::move(*manifest), .root = std::move(root)};
}

auto ProjectFile::find_root(std::filesystem::path const &start) -> std::optional<std::filesystem::path> {
    std::error_code error;
    auto directory = std::filesystem::absolute(start, error).lexically_normal();

    while (!directory.empty()) {
        if (std::filesystem::exists(directory / file_name, error)) {
            return directory;
        }

        auto parent = directory.parent_path();

        if (parent == directory) {
            break;
        }

        directory = std::move(parent);
    }

    return std::nullopt;
}

auto ProjectFile::save() const -> std::expected<void, std::string> {
    auto const path = root / file_name;
    std::ofstream out{path, std::ios::trunc};

    if (!out) {
        return std::unexpected{std::format("could not write '{}'", path.string())};
    }

    out << manifest.to_text();

    return out ? std::expected<void, std::string>{} : std::unexpected{std::format("could not write '{}'", path.string())};
}

auto RecentProjects::user_default() -> RecentProjects {
    return RecentProjects{Paths::user_state_directory() / "recent_projects.txt"};
}

auto RecentProjects::list() const -> std::vector<std::filesystem::path> {
    std::vector<std::filesystem::path> roots;
    std::ifstream in{file_};
    std::string line;
    std::error_code error;

    while (std::getline(in, line)) {
        if (!line.empty() && std::filesystem::exists(std::filesystem::path{line} / ProjectFile::file_name, error)) {
            roots.emplace_back(line);
        }
    }

    return roots;
}

auto RecentProjects::remember(std::filesystem::path const &root) const -> void {
    auto roots = list();

    std::erase(roots, root);
    roots.insert(roots.begin(), root);
    roots.resize(std::min(roots.size(), capacity));

    std::error_code error;
    std::filesystem::create_directories(file_.parent_path(), error);

    std::ofstream out{file_, std::ios::trunc};

    for (auto const &entry: roots) {
        out << entry.generic_string() << '\n';
    }
}
