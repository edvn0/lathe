#pragma once

#include <cstdint>
#include <optional>

class CommandLine;

enum class PresentModeChoice : std::uint8_t {
    immediate,
    mailbox,
    fifo,
    fifo_relaxed,
};

// The swapchain presentation flags shared by every executable that opens a window.
struct PresentationArguments {
    explicit PresentationArguments(CommandLine &cli);

    PresentationArguments(PresentationArguments const &) = delete;
    auto operator=(PresentationArguments const &) -> PresentationArguments & = delete;

    std::optional<bool> vsync;
    std::optional<PresentModeChoice> present_mode;
    std::optional<std::uint32_t> swapchain_images;
};
