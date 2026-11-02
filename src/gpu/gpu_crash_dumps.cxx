#include "gpu/gpu_crash_dumps.hxx"

namespace gpu_crash_dumps {

    auto mode_from(std::string_view text, bool after_device_loss) -> std::optional<Mode> {
        if (text == "off") {
            return Mode::off;
        }
        if (text == "standard") {
            return Mode::standard;
        }
        if (text == "full") {
            return Mode::full;
        }
        if (text == "auto") {
            return after_device_loss ? Mode::full : Mode::standard;
        }
        return std::nullopt;
    }

}
