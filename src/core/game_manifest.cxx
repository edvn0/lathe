#include "core/game_manifest.hxx"

#include <charconv>
#include <format>
#include <fstream>
#include <optional>
#include <type_traits>
#include <string_view>

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

    template<typename T>
    auto parse_number(std::string_view text) -> std::optional<T> {
        T value{};
        auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size()) {
            return std::nullopt;
        }
        return value;
    }

    // One `key = value` of the [gi] table; false (with `error` set) when the key or the value is not acceptable.
    auto set_gi(GiSettings &gi, std::string_view key, std::string_view value, std::string &error) -> bool {
        auto const quoted = value.size() >= 2 && value.front() == '"' && value.back() == '"';
        auto const bare = quoted ? value.substr(1, value.size() - 2) : value;
        auto const flag = [&](bool &target) {
            if (bare == "true" || bare == "false") {
                target = bare == "true";
                return true;
            }
            error = std::format("'{}' must be true or false", key);
            return false;
        };
        auto const number = [&](auto &target, auto low, auto high) {
            using T = std::remove_reference_t<decltype(target)>;
            auto const parsed = parse_number<T>(bare);
            if (!parsed || *parsed < static_cast<T>(low) || *parsed > static_cast<T>(high)) {
                error = std::format("'{}' must be a number from {} to {}", key, low, high);
                return false;
            }
            target = *parsed;
            return true;
        };
        if (key == "enabled") {
            return flag(gi.enabled);
        }
        if (key == "far_cascade") {
            return flag(gi.far_cascade);
        }
        if (key == "rays_per_probe") {
            return number(gi.rays_per_probe, 64, 256);
        }
        if (key == "probe_spacing") {
            return number(gi.probe_spacing, 0.1F, 16.0F);
        }
        if (key == "intensity") {
            return number(gi.intensity, 0.0F, 16.0F);
        }
        if (key == "emitter_intensity") {
            return number(gi.emitter_intensity, 0.0F, 100000.0F);
        }
        if (key == "synthesize_lights") {
            if (bare != "auto" && bare != "on" && bare != "off") {
                error = "'synthesize_lights' must be \"auto\", \"on\" or \"off\"";
                return false;
            }
            gi.synthesize_lights = std::string{bare};
            return true;
        }
        error = std::format("unknown key '{}' in [gi]", key);
        return false;
    }
}

auto GameManifest::load(std::filesystem::path const &path, bool require_game) -> std::expected<GameManifest, std::string> {
    std::ifstream in{path};

    if (!in) {
        return std::unexpected{std::format("could not open '{}'", path.string())};
    }

    GameManifest manifest;
    std::string line;
    std::size_t number = 0;
    auto in_gi = false;

    while (std::getline(in, line)) {
        ++number;

        auto const text = trim(line);

        if (text.empty() || text.front() == '#') {
            continue;
        }

        if (text.front() == '[') {
            if (text != "[gi]") {
                return std::unexpected{std::format("{}:{}: unknown table '{}' (only [gi])", path.string(), number, text)};
            }
            in_gi = true;
            continue;
        }

        auto const equals = text.find('=');

        if (equals == std::string_view::npos) {
            return std::unexpected{std::format("{}:{}: expected key = \"value\"", path.string(), number)};
        }

        auto const key = trim(text.substr(0, equals));
        auto value = trim(text.substr(equals + 1));

        if (in_gi) {
            auto message = std::string{};
            if (!set_gi(manifest.gi, key, value, message)) {
                return std::unexpected{std::format("{}:{}: {}", path.string(), number, message)};
            }
            continue;
        }

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
        } else if (key == "entry") {
            manifest.entry = value;
        } else if (key == "scene") {
            manifest.scene = value;
        }
    }

    if (require_game && manifest.game.empty()) {
        return std::unexpected{std::format("{}: missing 'game'", path.string())};
    }

    return manifest;
}

auto GameManifest::to_text() const -> std::string {
    auto text = std::format("name = \"{}\"\ntitle = \"{}\"\ngame = \"{}\"\nversion = \"{}\"\n", name, title, game, version);

    if (!entry.empty()) {
        text += std::format("entry = \"{}\"\n", entry);
    }

    if (!scene.empty()) {
        text += std::format("scene = \"{}\"\n", scene);
    }

    if (gi != GiSettings{}) {
        text += std::format("\n[gi]\nenabled = {}\nrays_per_probe = {}\nprobe_spacing = {}\nintensity = {}\n"
                            "emitter_intensity = {}\nfar_cascade = {}\nsynthesize_lights = \"{}\"\n",
                            gi.enabled, gi.rays_per_probe, gi.probe_spacing, gi.intensity, gi.emitter_intensity,
                            gi.far_cascade, gi.synthesize_lights);
    }

    return text;
}
