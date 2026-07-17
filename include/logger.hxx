#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace logger {

    enum class Level : std::uint8_t { trace, debug, info, warn, error, fatal };

    class Logger {
    public:
        static auto the() -> Logger &;
        auto message(Level level, std::string_view message) -> void;

    private:
        Logger() = default;
    };

} // namespace logger

namespace detail {

    template<typename... Args>
    auto create_log_message(logger::Level level, std::format_string<Args...> format_string, Args &&...args) -> void {
        auto message = std::format(format_string, std::forward<Args>(args)...);
        logger::Logger::the().message(level, message);
    }
} // namespace detail

template<typename... Args>
auto info(std::format_string<Args...> format_string, Args &&...args) -> void {
    return detail::create_log_message(logger::Level::info, format_string, std::forward<Args>(args)...);
}

template<typename... Args>
auto error(std::format_string<Args...> format_string, Args &&...args) -> void {
    return detail::create_log_message(logger::Level::error, format_string, std::forward<Args>(args)...);
}

template<typename... Args>
auto debug(std::format_string<Args...> format_string, Args &&...args) -> void {
    return detail::create_log_message(logger::Level::debug, format_string, std::forward<Args>(args)...);
}
template<typename... Args>
auto warn(std::format_string<Args...> format_string, Args &&...args) -> void {
    return detail::create_log_message(logger::Level::warn, format_string, std::forward<Args>(args)...);
}
