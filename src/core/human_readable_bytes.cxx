#include "core/human_readable_bytes.hxx"

#include <array>
#include <format>
#include <string_view>

namespace detail {

    auto human_readable_bytes_impl(std::uint64_t const size) -> std::string {
        constexpr std::array<std::string_view, 7> units{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};

        std::size_t unit = 0;
        auto whole = size;
        auto remainder = std::uint64_t{0};
        while (whole >= 1024 && unit + 1 < units.size()) {
            remainder = whole % 1024;
            whole /= 1024;
            ++unit;
        }

        if (remainder == 0) {
            return std::format("{} {}", whole, units[unit]);
        }
        auto const value = static_cast<double>(whole) + static_cast<double>(remainder) / 1024.0;
        return std::format("{:.2f} {}", value, units[unit]);
    }

}
