#include "runtime/presentation_arguments.hxx"

#include <array>
#include <string_view>
#include <utility>

#include "core/command_line.hxx"

namespace {

    constexpr std::array<std::pair<std::string_view, PresentModeChoice>, 4> present_mode_choices{{
            {"immediate", PresentModeChoice::immediate},
            {"mailbox", PresentModeChoice::mailbox},
            {"fifo", PresentModeChoice::fifo},
            {"fifo_relaxed", PresentModeChoice::fifo_relaxed},
    }};

}

PresentationArguments::PresentationArguments(CommandLine &cli) {
    auto group = cli.group("Presentation");
    group.toggle("--vsync", "Cap the frame rate at the refresh rate (off by default while benchmarking)", vsync);
    group.choice<PresentModeChoice>("--present-mode",
                                    "Present mode outright (benchmarks prefer immediate: MAILBOX can still pace "
                                    "acquisition to the refresh rate)",
                                    present_mode_choices, present_mode);
    group.value("--swapchain-images", "N", "Swapchain images to ask for, 2 to 8", swapchain_images,
                CommandLineGroup::Range<std::uint32_t>{.min = 2, .max = 8});
}

