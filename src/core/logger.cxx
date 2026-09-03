#include "core/logger.hxx"

#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

        auto from_spdlog_level(spdlog::level::level_enum level) noexcept -> Level {
            switch (level) {
                case spdlog::level::trace:
                    return Level::trace;

                case spdlog::level::debug:
                    return Level::debug;

                case spdlog::level::info:
                    return Level::info;

                case spdlog::level::warn:
                    return Level::warn;

                case spdlog::level::err:
                    return Level::error;

                case spdlog::level::critical:
                    return Level::fatal;

                default:
                    return Level::info;
            }
        }

        class TerminalSink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            auto drain(std::vector<ConsoleMessage> &output) -> void {
                output.clear();

                std::scoped_lock lock{this->mutex_};
                pending_.swap(output);
            }

        protected:
            auto sink_it_(spdlog::details::log_msg const &message) -> void override {
                pending_.push_back(ConsoleMessage{
                        .level = from_spdlog_level(message.level),
                        .message =
                                std::string{
                                        message.payload.data(),
                                        message.payload.size(),
                                },
                });
            }

            auto flush_() -> void override {}

        private:
            std::vector<ConsoleMessage> pending_;
        };

        struct LoggerState {
            std::shared_ptr<TerminalSink> terminal_sink;
            std::shared_ptr<spdlog::logger> logger;
        };

        auto logger_state() -> LoggerState & {
            static auto state = [] {
                auto terminal_sink = std::make_shared<TerminalSink>();

                auto app_logger = spdlog::stdout_color_mt("app");

                app_logger->sinks().push_back(terminal_sink);

                app_logger->set_pattern("%^[%L] %v%$");

                app_logger->set_level(spdlog::level::trace);

                return LoggerState{
                        .terminal_sink = std::move(terminal_sink),

                        .logger = std::move(app_logger),
                };
            }();

            return state;
        }

    } // namespace

    auto Logger::the() -> Logger & {
        static Logger instance;
        return instance;
    }

    auto Logger::message(Level level, std::string_view message) -> void {
        logger_state().logger->log(to_spdlog_level(level), spdlog::string_view_t{
                                                                   message.data(),
                                                                   message.size(),
                                                           });
    }

    auto Logger::drain_console(std::vector<ConsoleMessage> &output) -> void {
        logger_state().terminal_sink->drain(output);
    }

} // namespace logger
