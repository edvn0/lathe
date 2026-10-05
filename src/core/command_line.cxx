#include "core/command_line.hxx"

#include <algorithm>
#include <format>

#include <lyra/lyra.hpp>

struct CommandLine::Impl {
    struct Entry {
        std::string left;
        std::string help;
    };

    struct Group {
        std::string title;
        std::vector<Entry> entries;
    };

    std::string program;
    std::string description;
    lyra::cli cli;
    bool show_help = false;
    std::vector<Group> groups;
    std::vector<std::string> positionals;
    std::vector<Entry> arguments;
};

CommandLine::CommandLine(std::string program, std::string description) : impl_(std::make_unique<Impl>()) {
    impl_->program = std::move(program);
    impl_->description = std::move(description);
    impl_->cli |= lyra::help(impl_->show_help);
}

CommandLine::~CommandLine() = default;

auto CommandLine::group(std::string_view title) -> CommandLineGroup {
    auto const found = std::ranges::find(impl_->groups, title, &Impl::Group::title);
    if (found != impl_->groups.end()) {
        return CommandLineGroup{*this, static_cast<std::size_t>(found - impl_->groups.begin())};
    }

    impl_->groups.push_back(Impl::Group{.title = std::string{title}});
    return CommandLineGroup{*this, impl_->groups.size() - 1};
}

auto CommandLine::add_flag(std::size_t group, std::string_view name, std::string_view help, bool &target) -> void {
    impl_->cli |= lyra::opt(target)[std::string{name}];
    impl_->groups[group].entries.push_back({.left = std::string{name}, .help = std::string{help}});
}

auto CommandLine::add_option(std::size_t group, std::string_view name, std::string_view hint, std::string_view help,
                             CommandLineGroup::Setter setter, bool repeatable) -> void {
    auto parser = lyra::opt(
            [setter = std::move(setter), name = std::string{name}](std::string const &text) -> lyra::parser_result {
                if (auto applied = setter(text); !applied) {
                    return lyra::parser_result::error(lyra::parser_result_type::no_match,
                                                      std::format("{}: {}", name, applied.error()));
                }
                return lyra::parser_result::ok(lyra::parser_result_type::matched);
            },
            std::string{hint})[std::string{name}];

    if (repeatable) {
        parser.cardinality(0, 0);
    }

    impl_->cli |= parser;
    impl_->groups[group].entries.push_back({.left = std::format("{} <{}>", name, hint), .help = std::string{help}});
}

auto CommandLineGroup::flag(std::string_view name, std::string_view help, bool &target) -> void {
    owner_->add_flag(index_, name, help, target);
}

auto CommandLineGroup::option(std::string_view name, std::string_view hint, std::string_view help, Setter setter,
                              bool repeatable) -> void {
    owner_->add_option(index_, name, hint, help, std::move(setter), repeatable);
}

auto CommandLineGroup::toggle(std::string_view name, std::string_view help, std::optional<bool> &target) -> void {
    option(name, "on|off", help, [&target](std::string_view text) -> std::expected<void, std::string> {
        if (text == "on") {
            target = true;
        } else if (text == "off") {
            target = false;
        } else {
            return std::unexpected(std::format("'{}' (expected on or off)", text));
        }
        return {};
    });
}

auto CommandLine::positional(std::string_view name, std::string_view help, std::string &target) -> void {
    impl_->cli |= lyra::arg(
                          [&target](std::string const &text) {
                              target = text;
                          },
                          std::string{name})
                          .required();
    impl_->positionals.emplace_back(name);
    impl_->arguments.push_back({.left = std::format("<{}>", name), .help = std::string{help}});
}

auto CommandLine::parse(std::span<char const *const> args) -> std::expected<Outcome, std::string> {
    // lyra::args takes the program name from the first element.
    std::vector<std::string> storage;
    storage.reserve(args.size() + 1);
    storage.push_back(impl_->program);
    storage.insert(storage.end(), args.begin(), args.end());

    impl_->show_help = false;
    auto const result = impl_->cli.parse(lyra::args{storage.begin(), storage.end()});

    if (impl_->show_help) {
        return Outcome::help;
    }
    if (!result) {
        return std::unexpected(result.message());
    }
    return Outcome::run;
}

auto CommandLine::help_text() const -> std::string {
    std::string usage = std::format("Usage: {} [options]", impl_->program);
    for (auto const &name: impl_->positionals) {
        usage += std::format(" <{}>", name);
    }

    std::string text = usage + "\n";
    if (!impl_->description.empty()) {
        text += "\n" + impl_->description + "\n";
    }

    // One column for every section, so the descriptions line up.
    constexpr std::size_t max_left_width = 40;
    constexpr std::string_view help_left = "-h, --help, -?";
    auto left_width = help_left.size();
    auto const measure = [&](std::vector<Impl::Entry> const &entries) {
        for (auto const &entry: entries) {
            if (entry.left.size() <= max_left_width) {
                left_width = std::max(left_width, entry.left.size());
            }
        }
    };
    measure(impl_->arguments);
    for (auto const &group: impl_->groups) {
        measure(group.entries);
    }

    auto const append_entry = [&](std::string_view left, std::string_view help) {
        if (left.size() > left_width) {
            text += std::format("  {}\n  {:{}}  {}\n", left, "", left_width, help);
        } else {
            text += std::format("  {:{}}  {}\n", left, left_width, help);
        }
    };

    if (!impl_->arguments.empty()) {
        text += "\nArguments:\n";
        for (auto const &entry: impl_->arguments) {
            append_entry(entry.left, entry.help);
        }
    }

    for (auto const &group: impl_->groups) {
        if (group.entries.empty()) {
            continue;
        }
        text += std::format("\n{}:\n", group.title);
        for (auto const &entry: group.entries) {
            append_entry(entry.left, entry.help);
        }
    }

    text += "\nHelp:\n";
    append_entry(help_left, "Show this message and exit.");
    return text;
}

auto CommandLine::split(std::string_view text, char separator) -> std::vector<std::string_view> {
    std::vector<std::string_view> parts;
    while (!text.empty()) {
        auto const at = text.find(separator);
        parts.push_back(text.substr(0, at));
        if (at == std::string_view::npos) {
            break;
        }
        text = text.substr(at + 1);
    }
    return parts;
}
