#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gui {

    struct NodePosition {
        float x = 0.0F;
        float y = 0.0F;

        auto operator==(NodePosition const &) const noexcept -> bool = default;
    };

    using NodePositions = std::unordered_map<std::uintptr_t, NodePosition>;

    // Node ids must survive restarts, so they hash the name instead of using an interned pointer. FNV-1a, cut to
    // 56 bits so the editor can derive pin and link ids from it without overflow.
    [[nodiscard]] constexpr auto stable_key(std::string_view scope, std::string_view name) noexcept -> std::uintptr_t {
        auto hash = std::uint64_t{0xCBF29CE484222325ULL};
        auto const mix = [&](std::string_view text) {
            for (auto const character: text) {
                hash ^= static_cast<std::uint8_t>(character);
                hash *= 0x100000001B3ULL;
            }
        };
        mix(scope);
        mix(":");
        mix(name);
        return static_cast<std::uintptr_t>(hash & ((std::uint64_t{1} << 56U) - 1));
    }

    [[nodiscard]] inline auto serialise_positions(NodePositions const &positions) -> std::string {
        auto keys = std::vector<std::uintptr_t>{};
        keys.reserve(positions.size());
        for (auto const &entry: positions) {
            keys.push_back(entry.first);
        }
        std::ranges::sort(keys);

        auto text = std::string{};
        for (auto const key: keys) {
            auto const &position = positions.at(key);
            text += std::format("{:x} {} {}\n", key, position.x, position.y);
        }
        return text;
    }

    // Malformed lines are skipped so a damaged file loses some positions instead of all of them.
    [[nodiscard]] inline auto parse_positions(std::string_view text) -> NodePositions {
        auto positions = NodePositions{};

        while (!text.empty()) {
            auto const end = text.find('\n');
            auto line = text.substr(0, end);
            text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);

            auto key = std::uintptr_t{};
            auto const key_end = std::from_chars(line.data(), line.data() + line.size(), key, 16);
            if (key_end.ec != std::errc{} || key_end.ptr == line.data() + line.size() || *key_end.ptr != ' ') {
                continue;
            }

            auto position = NodePosition{};
            auto const x_begin = key_end.ptr + 1;
            auto const x_end = std::from_chars(x_begin, line.data() + line.size(), position.x);
            if (x_end.ec != std::errc{} || x_end.ptr == line.data() + line.size() || *x_end.ptr != ' ') {
                continue;
            }

            auto const y_end = std::from_chars(x_end.ptr + 1, line.data() + line.size(), position.y);
            if (y_end.ec != std::errc{}) {
                continue;
            }

            positions[key] = position;
        }

        return positions;
    }

}
