#include "core/cvar.hxx"

#include <array>
#include <cctype>
#include <cstdlib>

#include "core/command_line.hxx"

CVarBase::CVarBase(Meta meta)
    : name_(meta.name), group_(meta.group), help_(meta.help), help_fn_(std::move(meta.help_fn)),
      cli_only_(meta.cli_only) {}

CVarBase::~CVarBase() { CVarRegistry::global().remove(*this); }

auto CVarBase::cli_name() const -> std::string {
    std::string cli{"--"};

    for (auto const character: name_) {
        cli += character == '_' ? '-' : character;
    }

    return cli;
}

auto CVarBase::env_name() const -> std::string {
    std::string env{"LATHE_"};

    for (auto const character: name_) {
        env += static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    }

    return env;
}

auto CVarBase::source() const -> CVarSource {
    std::scoped_lock const lock{mutex_};
    return source_;
}

auto to_string(CVarSource source) -> std::string_view {
    switch (source) {
        case CVarSource::fallback:
            return "default";
        case CVarSource::environment:
            return "environment";
        case CVarSource::command_line:
            return "command line";
        case CVarSource::runtime:
            return "set at run time";
    }

    return "";
}

auto CVarRegistry::global() -> CVarRegistry & {
    static CVarRegistry registry;

    return registry;
}

auto CVarRegistry::add(CVarBase &cvar) -> void {
    std::scoped_lock const lock{mutex_};

    cvars_.push_back(&cvar);
}

auto CVarRegistry::remove(CVarBase &cvar) -> void {
    std::scoped_lock const lock{mutex_};

    std::erase(cvars_, &cvar);
}

auto CVarRegistry::find(std::string_view name) const -> CVarBase * {
    std::scoped_lock const lock{mutex_};

    auto const found = std::ranges::find(cvars_, name, &CVarBase::name);

    return found == cvars_.end() ? nullptr : *found;
}

auto CVarRegistry::all() const -> std::vector<CVarBase *> {
    std::scoped_lock const lock{mutex_};

    return cvars_;
}

auto CVarRegistry::apply_environment() -> std::vector<std::string> {
    std::vector<std::string> problems;

    for (auto *const cvar: all()) {
        if (cvar->cli_only()) {
            continue;
        }

        auto const env = cvar->env_name();
        auto const *const value = std::getenv(env.c_str());

        if (value == nullptr || *value == '\0') {
            continue;
        }

        if (auto const applied = cvar->set_text(value, CVarSource::environment); !applied) {
            problems.push_back(std::format("{}: {}", env, applied.error()));
        }
    }

    return problems;
}

auto CVarRegistry::attach(CommandLine &cli) -> void {
    // Registration order depends on how the libraries are linked; --help reads better with the groups in a fixed one.
    constexpr std::array<std::string_view, 9> group_order{
            "Game",         "Scene",     "Paths",    "Display",   "Presentation",
            "Diagnostics",  "Packaging", "Benchmark", "Benchmark comparison",
    };

    auto cvars = all();

    std::ranges::stable_sort(cvars, [&](CVarBase const *left, CVarBase const *right) {
        return std::ranges::find(group_order, left->group()) < std::ranges::find(group_order, right->group());
    });

    for (auto *const cvar: cvars) {
        auto group = cli.group(cvar->group());

        if (cvar->is_flag()) {
            group.flag(cvar->cli_name(), cvar->help(), [cvar] {
                static_cast<void>(cvar->set_text("on", CVarSource::command_line));
            });
        } else {
            group.option(
                    cvar->cli_name(), cvar->hint(), cvar->help(),
                    [cvar](std::string_view text) { return cvar->set_text(text, CVarSource::command_line); },
                    cvar->is_list());
        }
    }

    cli.note("Every option can also be set in the environment: LATHE_ and the option's name in capitals with _ for -, "
             "e.g. LATHE_RENDER_SCALE=2. The command line wins. Options that start an action (--bake-shaders, "
             "--benchmark, ...) are command line only.");
}

auto CVarRegistry::set(std::string_view name, std::string_view text) -> std::expected<void, std::string> {
    auto *const cvar = find(name);

    if (cvar == nullptr) {
        return std::unexpected(std::format("no variable called '{}'", name));
    }

    return cvar->set_text(text, CVarSource::runtime);
}

auto CVarRegistry::reset_all() -> void {
    for (auto *const cvar: all()) {
        cvar->reset();
    }
}
