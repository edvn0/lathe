#include <doctest/doctest.h>

#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "core/command_line.hxx"
#include "core/cvar.hxx"

namespace {

    enum class Shade : std::uint8_t { light, dark };

    constexpr std::array<std::pair<std::string_view, Shade>, 2> shades{{{"light", Shade::light}, {"dark", Shade::dark}}};

    auto check_even(std::string_view text) -> std::expected<void, std::string> {
        if (text.size() % 2 == 0) {
            return {};
        }

        return std::unexpected(std::string{"needs an even length"});
    }

}

TEST_CASE("a CVar reports its default until something sets it") {
    CVar<float> scale{{.name = "test_scale", .group = "Test", .help = "", .default_value = 1.5F}};

    CHECK(scale.get() == 1.5F);
    CHECK_FALSE(scale.has_value());
    CHECK(scale.value_or(9.0F) == 9.0F);
    CHECK_FALSE(scale.value().has_value());

    REQUIRE(scale.set_text("2", CVarSource::command_line).has_value());

    CHECK(scale.get() == 2.0F);
    CHECK(scale.has_value());
    CHECK(scale.value_or(9.0F) == 2.0F);
    CHECK(scale.source() == CVarSource::command_line);
    CHECK(scale.text() == "2");
    CHECK(scale.default_text() == "1.5");

    scale.reset();

    CHECK(scale.get() == 1.5F);
    CHECK_FALSE(scale.has_value());
}

TEST_CASE("numbers are range checked and must be finite") {
    CVar<std::uint32_t> images{{.name = "test_images", .group = "Test", .help = "", .min = 2, .max = 8}};
    CVar<float> hz{{.name = "test_hz", .group = "Test", .help = "", .min = 1.0F}};

    CHECK(images.set_text("4", CVarSource::runtime).has_value());
    CHECK_FALSE(images.set_text("9", CVarSource::runtime).has_value());
    CHECK_FALSE(images.set_text("1", CVarSource::runtime).has_value());
    CHECK_FALSE(images.set_text("-1", CVarSource::runtime).has_value());
    CHECK_FALSE(images.set_text("four", CVarSource::runtime).has_value());
    CHECK_FALSE(images.set_text("4x", CVarSource::runtime).has_value());
    CHECK(images.get() == 4U);

    CHECK(hz.set_text("144", CVarSource::runtime).has_value());
    CHECK_FALSE(hz.set_text("nan", CVarSource::runtime).has_value());
    CHECK_FALSE(hz.set_text("inf", CVarSource::runtime).has_value());
    CHECK_FALSE(hz.set_text("0.5", CVarSource::runtime).has_value());
}

TEST_CASE("enums, strings, paths and bools parse and print") {
    CVar<Shade> shade{{.name = "test_shade", .group = "Test", .help = "", .choices = shades}};
    CVar<std::string> tag{{.name = "test_tag", .group = "Test", .help = "", .validator = check_even}};
    CVar<std::filesystem::path> file{{.name = "test_file", .group = "Test", .help = ""}};
    CVar<bool> flag{{.name = "test_flag", .group = "Test", .help = ""}};
    CVar<bool> toggle{{.name = "test_toggle", .group = "Test", .help = "", .toggle = true}};

    CHECK(shade.hint() == "light|dark");
    REQUIRE(shade.set_text("dark", CVarSource::runtime).has_value());
    CHECK(shade.get() == Shade::dark);
    CHECK(shade.text() == "dark");
    CHECK_FALSE(shade.set_text("grey", CVarSource::runtime).has_value());

    CHECK(tag.set_text("ab", CVarSource::runtime).has_value());
    CHECK_FALSE(tag.set_text("abc", CVarSource::runtime).has_value());
    CHECK(tag.get() == "ab");

    CHECK(file.set_text("a/b.lbf", CVarSource::runtime).has_value());
    CHECK_FALSE(file.set_text("", CVarSource::runtime).has_value());
    CHECK(file.get() == std::filesystem::path{"a/b.lbf"});

    CHECK(flag.is_flag());
    CHECK_FALSE(toggle.is_flag());
    CHECK(toggle.set_text("off", CVarSource::runtime).has_value());
    CHECK(toggle.has_value());
    CHECK_FALSE(toggle.get());
    CHECK_FALSE(toggle.set_text("maybe", CVarSource::runtime).has_value());
}

TEST_CASE("a list collects what one source gives it and starts over for the next") {
    CVar<std::vector<std::string>> sweeps{{.name = "test_sweeps", .group = "Test", .help = ""}};

    REQUIRE(sweeps.set_text("a", CVarSource::environment).has_value());
    REQUIRE(sweeps.set_text("b", CVarSource::command_line).has_value());
    REQUIRE(sweeps.set_text("c", CVarSource::command_line).has_value());

    CHECK(sweeps.get() == std::vector<std::string>{"b", "c"});
    CHECK(sweeps.text() == "b,c");
    CHECK(sweeps.is_list());
}

TEST_CASE("names map to command line options and environment variables") {
    CVar<bool> flag{{.name = "test_max_quality", .group = "Test", .help = ""}};

    CHECK(flag.cli_name() == "--test-max-quality");
    CHECK(flag.env_name() == "LATHE_TEST_MAX_QUALITY");
}

TEST_CASE("the environment is applied first and the command line overrides it") {
    CVar<std::uint32_t> frames{{.name = "test_env_frames", .group = "Test Env", .help = "", .hint = "N"}};
    CVar<std::string> action{{.name = "test_env_action", .group = "Test Env", .help = "", .cli_only = true}};
    CVar<std::uint32_t> broken{{.name = "test_env_broken", .group = "Test Env", .help = ""}};
    CVar<bool> quick{{.name = "test_env_quick", .group = "Test Env", .help = ""}};

    setenv("LATHE_TEST_ENV_FRAMES", "30", 1);
    setenv("LATHE_TEST_ENV_ACTION", "run", 1);
    setenv("LATHE_TEST_ENV_BROKEN", "lots", 1);

    auto const problems = CVarRegistry::global().apply_environment();

    unsetenv("LATHE_TEST_ENV_FRAMES");
    unsetenv("LATHE_TEST_ENV_ACTION");
    unsetenv("LATHE_TEST_ENV_BROKEN");

    CHECK(frames.get() == 30U);
    CHECK(frames.source() == CVarSource::environment);
    CHECK_FALSE(action.has_value());
    CHECK_FALSE(broken.has_value());
    REQUIRE(problems.size() == 1);
    CHECK(problems.front().find("LATHE_TEST_ENV_BROKEN") != std::string::npos);

    CommandLine cli{"lathe", ""};
    CVarRegistry::global().attach(cli);
    std::array<char const *, 3> args{"--test-env-frames=45", "--test-env-quick", "--test-env-action=go"};

    REQUIRE(cli.parse(args).has_value());

    CHECK(frames.get() == 45U);
    CHECK(frames.source() == CVarSource::command_line);
    CHECK(quick.get());
    CHECK(action.get() == "go");
    CHECK(cli.help_text().find("--test-env-frames <N>") != std::string::npos);

    std::array<char const *, 1> bad{"--test-env-frames=many"};
    CHECK_FALSE(cli.parse(bad).has_value());
}

TEST_CASE("variables can be found and set by name at run time") {
    CVar<std::uint32_t> level{{.name = "test_named_level", .group = "Test"}};

    REQUIRE(CVarRegistry::global().find("test_named_level") == &level);
    CHECK(CVarRegistry::global().set("test_named_level", "7").has_value());
    CHECK(level.get() == 7U);
    CHECK(level.source() == CVarSource::runtime);
    CHECK_FALSE(CVarRegistry::global().set("test_named_level", "x").has_value());
    CHECK_FALSE(CVarRegistry::global().set("test_no_such_level", "1").has_value());
}

TEST_CASE("a variable leaves the registry when it is destroyed") {
    {
        CVar<bool> brief{{.name = "test_brief", .group = "Test"}};
        CHECK(CVarRegistry::global().find("test_brief") == &brief);
    }

    CHECK(CVarRegistry::global().find("test_brief") == nullptr);
}

TEST_CASE("every engine variable has a unique name, command line option and environment variable") {
    std::set<std::string> names;
    std::set<std::string> options;
    std::set<std::string> environment;

    for (auto const *cvar: CVarRegistry::global().all()) {
        if (cvar->name().starts_with("test_")) {
            continue;
        }

        CHECK_MESSAGE(names.insert(std::string{cvar->name()}).second, cvar->name());
        CHECK_MESSAGE(options.insert(cvar->cli_name()).second, cvar->cli_name());
        CHECK_MESSAGE(environment.insert(cvar->env_name()).second, cvar->env_name());
        CHECK_MESSAGE(!cvar->help().empty(), cvar->name());
        CHECK_MESSAGE(!cvar->group().empty(), cvar->name());
    }

    CHECK(names.contains("render_scale"));
    CHECK(names.contains("platform"));
    CHECK(names.contains("window_size"));
}
