#pragma once

#include <slang.h>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <string_view>

enum class SlangLibraryErrorType : std::uint8_t {
    invalid_argument,
    executable_path_failed,
    library_not_found,
    library_load_failed,
    symbol_not_found,
    global_session_failed,
};

struct SlangLibraryError {
    SlangLibraryErrorType type = SlangLibraryErrorType::invalid_argument;

    SlangResult result = SLANG_OK;

    std::string diagnostics;
};

template<>
struct std::formatter<SlangLibraryErrorType> : std::formatter<std::string_view> {
    constexpr auto format(SlangLibraryErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case SlangLibraryErrorType::invalid_argument:
                    return "invalid_argument";
                case SlangLibraryErrorType::executable_path_failed:
                    return "executable_path_failed";
                case SlangLibraryErrorType::library_not_found:
                    return "library_not_found";
                case SlangLibraryErrorType::library_load_failed:
                    return "library_load_failed";
                case SlangLibraryErrorType::symbol_not_found:
                    return "symbol_not_found";
                case SlangLibraryErrorType::global_session_failed:
                    return "global_session_failed";
            }

            return "unknown_slang_library_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

class SlangLibrary {
public:
    SlangLibrary() noexcept;

    ~SlangLibrary();

    SlangLibrary(SlangLibrary const &) = delete;

    auto operator=(SlangLibrary const &) -> SlangLibrary & = delete;

    SlangLibrary(SlangLibrary &&other) noexcept;

    auto operator=(SlangLibrary &&other) noexcept -> SlangLibrary &;

    [[nodiscard]]
    static auto create(std::filesystem::path const &library_path) -> std::expected<SlangLibrary, SlangLibraryError>;

    [[nodiscard]]
    static auto create_from_executable_directory(std::filesystem::path const &library_name =
#if defined(_WIN32)
                                                         L"slang-compiler.dll"
#else
                                                         "libslang.so"
#endif
                                                 ) -> std::expected<SlangLibrary, SlangLibraryError>;

    [[nodiscard]]
    auto create_global_session(slang::IGlobalSession **session) const noexcept -> SlangResult;

    [[nodiscard]]
    auto create_global_session(SlangGlobalSessionDesc const &description,
                               slang::IGlobalSession **session) const noexcept -> SlangResult;

    [[nodiscard]]
    auto valid() const noexcept -> bool;

    [[nodiscard]]
    auto path() const noexcept -> std::filesystem::path const &;

    auto destroy() noexcept -> void;

private:
    struct Impl;

    explicit SlangLibrary(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};
