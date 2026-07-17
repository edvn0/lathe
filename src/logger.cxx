#include <logger.hxx>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string_view>

namespace logger {

    namespace {

        auto to_spdlog_level(Level level) noexcept -> spdlog::level::level_enum {
            switch (level) {
                case Level::trace:
                    return spdlog::level::trace;
                case Level::debug:
                    return spdlog::level::debug;
                case Level::info:
                    return spdlog::level::info;
                case Level::warn:
                    return spdlog::level::warn;
                case Level::error:
                    return spdlog::level::err;
                case Level::fatal:
                    return spdlog::level::critical;
            }

            return spdlog::level::off;
        }

        auto get_logger() -> std::shared_ptr<spdlog::logger> {
            static auto instance = [] {
                auto logger = spdlog::stdout_color_mt("app");

                logger->set_pattern("%^[%L] %v%$");
                logger->set_level(spdlog::level::trace);

                return logger;
            }();

            return instance;
        }

    } // namespace

    auto Logger::the() -> Logger & {
        static Logger instance;
        return instance;
    }

    auto Logger::message(Level level, std::string_view message) -> void {
        get_logger()->log(to_spdlog_level(level), "{}", message);
    }

} // namespace logger
