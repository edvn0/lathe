#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace logger {

    enum class Level : std::uint8_t {
        trace,
        debug,
        info,
        warn,
        error,
        fatal,
    };

    struct ConsoleMessage {
        Level level{};
        std::string message;
    };

    class Logger {
    public:
        [[nodiscard]]
        static auto the() -> Logger &;

        auto message(Level level, std::string_view message) -> void;

        // Swaps the pending console messages into `output`. Reuse `output` between frames to recycle allocations.
        auto drain_console(std::vector<ConsoleMessage> &output) -> void;

    private:
        Logger() = default;
    };

    inline auto trace(std::string_view message) -> void { Logger::the().message(Level::trace, message); }

    inline auto debug(std::string_view message) -> void { Logger::the().message(Level::debug, message); }

    inline auto info(std::string_view message) -> void { Logger::the().message(Level::info, message); }

    inline auto warn(std::string_view message) -> void { Logger::the().message(Level::warn, message); }

    inline auto error(std::string_view message) -> void { Logger::the().message(Level::error, message); }

    inline auto fatal(std::string_view message) -> void { Logger::the().message(Level::fatal, message); }

    template<typename... Args>
    auto trace(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::trace, std::format(format, std::forward<Args>(args)...));
    }

    template<typename... Args>
    auto debug(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::debug, std::format(format, std::forward<Args>(args)...));
    }

    template<typename... Args>
    auto info(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::info, std::format(format, std::forward<Args>(args)...));
    }

    template<typename... Args>
    auto warn(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::warn, std::format(format, std::forward<Args>(args)...));
    }

    template<typename... Args>
    auto error(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::error, std::format(format, std::forward<Args>(args)...));
    }

    template<typename... Args>
    auto fatal(std::format_string<Args...> format, Args &&...args) -> void {
        Logger::the().message(Level::fatal, std::format(format, std::forward<Args>(args)...));
    }

} // namespace logger

// Unqualified info()/warn()/error() throughout the engine.
using logger::debug;
using logger::error;
using logger::fatal;
using logger::info;
using logger::trace;
using logger::warn;
