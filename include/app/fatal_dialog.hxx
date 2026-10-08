#pragma once

#include <string>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <vector>

extern char **environ; // NOLINT(readability-redundant-declaration)
#endif

inline auto show_fatal_dialog(std::string_view title, std::string_view message) noexcept -> bool {
#if defined(_WIN32)
    auto const widen = [](std::string_view text) {
        auto const size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
        std::wstring wide(static_cast<std::size_t>(size), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
        return wide;
    };

    return MessageBoxW(nullptr, widen(message).c_str(), widen(title).c_str(),
                       MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST) != 0;
#else
    if (std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr) {
        return false;
    }

    auto const title_text = std::string{title};
    auto const message_text = std::string{message};

    std::array<std::vector<std::string>, 3> const candidates{{
            {"zenity", "--error", "--no-markup", "--width=440", "--title", title_text, "--text", message_text},
            {"kdialog", "--error", message_text, "--title", title_text},
            {"xmessage", "-center", "-buttons", "Quit:0", title_text + "\n\n" + message_text},
    }};

    for (auto const &candidate: candidates) {
        std::vector<char *> argv;

        for (auto const &argument: candidate) {
            argv.push_back(const_cast<char *>(argument.c_str()));
        }

        argv.push_back(nullptr);

        pid_t child = 0;

        if (posix_spawnp(&child, argv.front(), nullptr, nullptr, argv.data(), environ) != 0) {
            continue;
        }

        int status = 0;

        while (waitpid(child, &status, 0) < 0) {
            if (errno != EINTR) {
                return true;
            }
        }

        return true;
    }

    return false;
#endif
}
