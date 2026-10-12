#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    enum class LintKind : std::uint8_t {
        culled_game_pass,
        unread_game_write,
        empty_game_pass,
    };

    struct LintFinding {
        LintKind kind = LintKind::culled_game_pass;
        std::string pass;
        std::string resource;
    };

    // Things a compiled graph accepts silently but that are almost certainly a mistake in a game pass: the pass was
    // culled because nothing reads its output, it writes a transient nobody reads, or it declares nothing at all.
    [[nodiscard]] auto lint(GraphDesc const &graph, CompiledGraph const &compiled) -> std::vector<LintFinding>;

    [[nodiscard]] constexpr auto to_string_view(LintKind kind) noexcept -> std::string_view {
        switch (kind) {
            case LintKind::culled_game_pass:
                return "culled_game_pass";
            case LintKind::unread_game_write:
                return "unread_game_write";
            case LintKind::empty_game_pass:
                return "empty_game_pass";
        }
        return "unknown_lint";
    }

}

template<>
struct std::formatter<frame_graph::LintFinding> : std::formatter<std::string_view> {
    auto format(frame_graph::LintFinding const &finding, std::format_context &context) const {
        auto const text = std::format("{} (pass '{}', resource '{}')", frame_graph::to_string_view(finding.kind),
                                      finding.pass, finding.resource);
        return std::formatter<std::string_view>::format(text, context);
    }
};
