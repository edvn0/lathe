#include <doctest/doctest.h>

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/command_line.hxx"

namespace {

    enum class Mode : std::uint8_t { fast, slow };

    constexpr std::array<std::pair<std::string_view, Mode>, 2> mode_choices{{{"fast", Mode::fast}, {"slow", Mode::slow}}};

    struct Fixture {
        CommandLine cli{"prog", "Does things."};
        bool verbose = false;
        std::uint32_t count = 1;
        std::optional<bool> toggled;
        std::optional<Mode> mode;
        std::filesystem::path path;
        std::vector<std::string> items;
        std::string input;

        Fixture() {
            auto general = cli.group("General");
            general.flag("--verbose", "Talk more", verbose);
            general.value("--count", "N", "How many", count, CommandLineGroup::Range<std::uint32_t>{.min = 1, .max = 8});
            general.toggle("--toggle", "On or off", toggled);
            general.value("--path", "FILE", "Where", path);

            auto advanced = cli.group("Advanced");
            advanced.choice<Mode>("--mode", "Speed", mode_choices, mode);
            advanced.option(
                    "--item", "X", "Repeatable",
                    [this](std::string_view text) -> std::expected<void, std::string> {
                        items.emplace_back(text);
                        return {};
                    },
                    true);
        }

        [[nodiscard]] auto parse(std::vector<char const *> const &args) -> std::expected<CommandLine::Outcome, std::string> {
            return cli.parse(std::span<char const *const>{args});
        }
    };

} // namespace

TEST_CASE("command line: values in both spellings") {
    Fixture f;
    auto const outcome = f.parse({"--verbose", "--count=3", "--toggle", "off", "--path=a/b", "--mode=slow"});
    REQUIRE(outcome.has_value());
    CHECK(*outcome == CommandLine::Outcome::run);
    CHECK(f.verbose);
    CHECK(f.count == 3);
    CHECK(f.toggled == false);
    CHECK(f.path == "a/b");
    CHECK(f.mode == Mode::slow);
}

TEST_CASE("command line: repeatable options see every occurrence") {
    Fixture f;
    REQUIRE(f.parse({"--item=a", "--item=b"}).has_value());
    CHECK(f.items == std::vector<std::string>{"a", "b"});
}

TEST_CASE("command line: errors") {
    for (auto const *bad: {"--count=0", "--count=9", "--count=x", "--toggle=maybe", "--mode=medium", "--bogus"}) {
        Fixture f;
        CHECK_FALSE(f.parse({bad}).has_value());
    }
}

TEST_CASE("command line: help and positionals") {
    Fixture f;
    auto const outcome = f.parse({"--help"});
    REQUIRE(outcome.has_value());
    CHECK(*outcome == CommandLine::Outcome::help);

    auto const text = f.cli.help_text();
    CHECK(text.find("General:") < text.find("Advanced:"));
    CHECK(text.find("--count <N>") != std::string::npos);
    CHECK(text.find("--mode <fast|slow>") != std::string::npos);

    CommandLine cli{"prog", ""};
    std::string input;
    cli.positional("input", "The input", input);
    std::array<char const *, 1> args{"in.hdr"};
    REQUIRE(cli.parse(args).has_value());
    CHECK(input == "in.hdr");
    CHECK_FALSE(cli.parse(std::span<char const *const>{}).has_value());
}
