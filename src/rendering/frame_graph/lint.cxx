#include "rendering/frame_graph/lint.hxx"

#include <algorithm>
#include <ranges>

namespace frame_graph {

    auto lint(GraphDesc const &graph, CompiledGraph const &compiled) -> std::vector<LintFinding> {
        auto findings = std::vector<LintFinding>{};

        auto const read_by_anyone = [&](std::uint32_t resource, std::uint32_t version, std::size_t except) {
            return std::ranges::any_of(graph.passes | std::views::enumerate, [&](auto const &entry) {
                auto const &[index, pass] = entry;
                return static_cast<std::size_t>(index) != except &&
                       std::ranges::any_of(pass.accesses, [&](AccessDesc const &access) {
                           return access.resource == resource && access.version == version;
                       });
            });
        };

        for (auto index = std::size_t{0}; index < graph.passes.size(); ++index) {
            auto const &pass = graph.passes[index];
            if (pass.owner != Owner::game) {
                continue;
            }

            if (pass.accesses.empty()) {
                findings.push_back({LintKind::empty_game_pass, pass.name, {}});
                continue;
            }
            if (index < compiled.pass_culled.size() && compiled.pass_culled[index]) {
                findings.push_back({LintKind::culled_game_pass, pass.name, {}});
                continue;
            }

            for (auto const &access: pass.accesses) {
                auto const &resource = graph.resources[access.resource];
                if (!access.produces || resource.imported || resource.owner != Owner::game) {
                    continue;
                }
                // A write turns version v into v + 1, and whoever uses v + 1 consumes it.
                if (!read_by_anyone(access.resource, access.version + 1, index)) {
                    findings.push_back({LintKind::unread_game_write, pass.name, resource.name});
                }
            }
        }
        return findings;
    }

}
