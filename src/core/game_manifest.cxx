#include "core/game_manifest.hxx"

#include <format>
#include <fstream>

namespace {
    auto trim(std::string_view text) -> std::string_view {
        while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
            text.remove_prefix(1);
        }

        while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
            text.remove_suffix(1);
        }

        return text;
    }
} // namespace

auto GameManifest::load(std::filesystem::path const &path) -> std::expected<GameManifest, std::string> {
    std::ifstream in{path};

    if (!in) {
        return std::unexpected{std::format("could not open '{}'", path.string())};
    }

    GameManifest manifest;
    std::string line;
    std::size_t number = 0;

    while (std::getline(in, line)) {
        ++number;

        auto const text = trim(line);

        if (text.empty() || text.front() == '#') {
            continue;
        }

        auto const equals = text.find('=');

        if (equals == std::string_view::npos) {
            return std::unexpected{std::format("{}:{}: expected key = \"value\"", path.string(), number)};
        }

        auto const key = trim(text.substr(0, equals));
        auto value = trim(text.substr(equals + 1));

        if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
            return std::unexpected{std::format("{}:{}: the value of '{}' must be a quoted string", path.string(),
                                               number, key)};
        }

        value = value.substr(1, value.size() - 2);

        if (key == "name") {
            manifest.name = value;
        } else if (key == "title") {
            manifest.title = value;
        } else if (key == "game") {
            manifest.game = value;
        } else if (key == "version") {
            manifest.version = value;
        }
    }

    if (manifest.game.empty()) {
        return std::unexpected{std::format("{}: missing 'game'", path.string())};
    }

    return manifest;
}

auto GameManifest::to_text() const -> std::string {
    return std::format("name = \"{}\"\ntitle = \"{}\"\ngame = \"{}\"\nversion = \"{}\"\n", name, title, game, version);
}
