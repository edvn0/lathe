#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "scripting/script_engine.hxx"

namespace gui {
    struct ScriptWidgetFrame {
        bool ran = false;
        std::uint32_t transforms_written = 0;
    };

    class ScriptWidget {
    public:
        explicit ScriptWidget(ScriptEngineSettings settings = {});

        auto draw(std::optional<ScriptWorld> world) -> ScriptWidgetFrame;

    private:
        auto run(ScriptWorld world) -> ScriptWidgetFrame;
        static auto report_error(ScriptError const &error) -> void;

        ScriptEngineSettings settings_;
        std::optional<ScriptEngine> engine_;
        std::string source_;
        std::optional<ScriptError> last_error_;
        std::optional<std::chrono::microseconds> last_duration_;
        std::uint32_t last_transforms_written_ = 0;
    };
}
