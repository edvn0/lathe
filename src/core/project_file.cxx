#include "core/project_file.hxx"

#include <format>
#include <fstream>
#include <system_error>

auto ProjectFile::load(std::filesystem::path const &path) -> std::expected<ProjectFile, std::string> {
    std::error_code error;
    auto const file = std::filesystem::is_directory(path, error) ? path / file_name : path;
    auto manifest = GameManifest::load(file, false);

    if (!manifest) {
        return std::unexpected{std::move(manifest.error())};
    }

    return ProjectFile{
            .manifest = std::move(*manifest),
            .root = std::filesystem::absolute(file, error).parent_path().lexically_normal(),
    };
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
