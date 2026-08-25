#include "slang_library.hxx"

#include <windows.h>

#include <algorithm>
#include <format>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <bit>

#include "logger.hxx"

namespace {

    using SlangCreateGlobalSessionFunction = SlangResult (*)(SlangGlobalSessionDesc const *description,
                                                             slang::IGlobalSession **session);

    [[nodiscard]]
    auto make_error(SlangLibraryErrorType type, SlangResult result, std::string diagnostics) -> SlangLibraryError {
        return SlangLibraryError{
                .type = type,
                .result = result,
                .diagnostics = std::move(diagnostics),
        };
    }

    [[nodiscard]]
    auto trim_windows_message(std::string message) -> std::string {
        while (!message.empty()) {
            auto const character = message.back();

            if (character != '\r' && character != '\n' && character != ' ' && character != '\t') {
                break;
            }

            message.pop_back();
        }

        return message;
    }

    [[nodiscard]]
    auto wide_to_utf8(std::wstring_view text) -> std::string {
        if (text.empty()) {
            return {};
        }

        auto const required_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                                                       static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);

        if (required_size <= 0) {
            return {};
        }

        auto result = std::string(static_cast<std::size_t>(required_size), '\0');

        auto const written_size =
                WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                                    result.data(), required_size, nullptr, nullptr);

        if (written_size <= 0) {
            return {};
        }

        result.resize(static_cast<std::size_t>(written_size));

        return result;
    }

    [[nodiscard]]
    auto windows_error_message(DWORD error) -> std::string {
        if (error == ERROR_SUCCESS) {
            return "No Windows error was reported.";
        }

        wchar_t *message_buffer = nullptr;

        auto const flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;

        auto const character_count = FormatMessageW(flags, nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                                    reinterpret_cast<wchar_t *>(&message_buffer), 0, nullptr);

        if (character_count == 0 || message_buffer == nullptr) {
            return std::format("Windows error {}.", error);
        }

        auto const message = wide_to_utf8(std::wstring_view{
                message_buffer,
                static_cast<std::size_t>(character_count),
        });

        LocalFree(message_buffer);

        if (message.empty()) {
            return std::format("Windows error {}.", error);
        }

        return trim_windows_message(message);
    }

    [[nodiscard]]
    auto executable_path() -> std::expected<std::filesystem::path, SlangLibraryError> {
        auto buffer = std::vector<wchar_t>(512, L'\0');

        while (true) {
            SetLastError(ERROR_SUCCESS);

            auto const length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));

            if (length == 0) {
                auto const error = GetLastError();

                return std::unexpected{make_error(SlangLibraryErrorType::executable_path_failed, SLANG_FAIL,
                                                  std::format("GetModuleFileNameW() "
                                                              "failed: {}",
                                                              windows_error_message(error)))};
            }

            if (length < static_cast<DWORD>(buffer.size() - 1)) {
                buffer.resize(static_cast<std::size_t>(length));

                return std::filesystem::path{
                        buffer.begin(),
                        buffer.end(),
                };
            }

            if (buffer.size() >= 32768) {
                return std::unexpected{make_error(SlangLibraryErrorType::executable_path_failed, SLANG_FAIL,
                                                  "The executable path exceeds "
                                                  "the supported Windows path "
                                                  "length.")};
            }

            buffer.resize(std::min<std::size_t>(buffer.size() * 2, 32768), L'\0');
        }
    }

    [[nodiscard]]
    auto absolute_path(std::filesystem::path const &path) -> std::expected<std::filesystem::path, SlangLibraryError> {
        std::error_code error_code;

        auto result = std::filesystem::absolute(path, error_code);

        if (error_code) {
            return std::unexpected{make_error(SlangLibraryErrorType::invalid_argument, SLANG_E_INVALID_ARG,
                                              std::format("Failed to resolve the absolute "
                                                          "Slang library path '{}': {}",
                                                          path.string(), error_code.message()))};
        }

        result = result.lexically_normal();

        return result;
    }

} // namespace

struct SlangLibrary::Impl {
    HMODULE module = nullptr;

    SlangCreateGlobalSessionFunction create_global_session = nullptr;

    std::filesystem::path library_path;
};

SlangLibrary::SlangLibrary() noexcept = default;

SlangLibrary::SlangLibrary(std::unique_ptr<Impl> impl) noexcept : impl_{std::move(impl)} {}

SlangLibrary::~SlangLibrary() { destroy(); }

SlangLibrary::SlangLibrary(SlangLibrary &&other) noexcept : impl_{std::move(other.impl_)} {}

auto SlangLibrary::operator=(SlangLibrary &&other) noexcept -> SlangLibrary & {
    if (this == &other) {
        return *this;
    }

    destroy();

    impl_ = std::move(other.impl_);

    return *this;
}

auto SlangLibrary::create(std::filesystem::path const &library_path) -> std::expected<SlangLibrary, SlangLibraryError> {
    if (library_path.empty()) {
        return std::unexpected{make_error(SlangLibraryErrorType::invalid_argument, SLANG_E_INVALID_ARG,
                                          "The Slang library path is empty.")};
    }

    /*
     * Preserve the path exactly. MinGW's std::filesystem normalization can
     * turn a UNC prefix:
     *
     *   \\server\share
     *
     * into:
     *
     *   \server\share
     */
    auto resolved_path = library_path;
    auto const &native_path = resolved_path.native();

    debug("Path passed to GetFileAttributesW: {}", wide_to_utf8(native_path));

    debug("Path starts with UNC prefix: {}",
          native_path.size() >= 2 && native_path[0] == L'\\' && native_path[1] == L'\\');

    SetLastError(ERROR_SUCCESS);

    auto const attributes = GetFileAttributesW(native_path.c_str());

    if (attributes == INVALID_FILE_ATTRIBUTES) {
        auto const error = GetLastError();

        return std::unexpected{make_error(SlangLibraryErrorType::library_not_found, SLANG_E_NOT_FOUND,
                                          std::format("The Slang runtime library was not found: {}\n"
                                                      "GetFileAttributesW() failed: {} "
                                                      "(Windows error {})",
                                                      wide_to_utf8(native_path), windows_error_message(error), error))};
    }

    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return std::unexpected{
                make_error(SlangLibraryErrorType::library_not_found, SLANG_E_NOT_FOUND,
                           std::format("The Slang runtime path refers to a directory: {}", wide_to_utf8(native_path)))};
    }

    auto impl = std::make_unique<Impl>();

    SetLastError(ERROR_SUCCESS);

    impl->module = LoadLibraryExW(native_path.c_str(), nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

    if (impl->module == nullptr) {
        auto const error = GetLastError();

        return std::unexpected{make_error(SlangLibraryErrorType::library_load_failed, SLANG_E_NOT_AVAILABLE,
                                          std::format("Failed to load the Slang runtime library '{}': "
                                                      "{} (Windows error {})",
                                                      wide_to_utf8(native_path), windows_error_message(error), error))};
    }

    SetLastError(ERROR_SUCCESS);

    auto const symbol = GetProcAddress(impl->module, "slang_createGlobalSession2");

    if (symbol == nullptr) {
        auto const error = GetLastError();

        FreeLibrary(impl->module);
        impl->module = nullptr;

        return std::unexpected{make_error(SlangLibraryErrorType::symbol_not_found, SLANG_E_NOT_FOUND,
                                          std::format("The Slang runtime library '{}' does not export "
                                                      "'slang_createGlobalSession2': {} "
                                                      "(Windows error {})",
                                                      wide_to_utf8(native_path), windows_error_message(error), error))};
    }

    static_assert(sizeof(symbol) == sizeof(SlangCreateGlobalSessionFunction));

    impl->create_global_session = std::bit_cast<SlangCreateGlobalSessionFunction>(symbol);

    impl->library_path = std::move(resolved_path);

    return SlangLibrary{
            std::move(impl),
    };
}

auto SlangLibrary::create_from_executable_directory(std::filesystem::path const &library_name)
        -> std::expected<SlangLibrary, SlangLibraryError> {
    if (library_name.empty()) {
        return std::unexpected{make_error(SlangLibraryErrorType::invalid_argument, SLANG_E_INVALID_ARG,
                                          "The Slang library name is empty.")};
    }

    if (library_name.has_parent_path()) {
        return std::unexpected{make_error(SlangLibraryErrorType::invalid_argument, SLANG_E_INVALID_ARG,
                                          std::format("Expected only a Slang library filename, "
                                                      "but received: {}",
                                                      library_name.string()))};
    }

    auto executable_path_result = executable_path();

    if (!executable_path_result) {
        return std::unexpected{std::move(executable_path_result.error())};
    }

    auto executable_native = executable_path_result->native();

    auto const separator = executable_native.find_last_of(L"\\/");

    if (separator == std::wstring::npos) {
        return std::unexpected{make_error(
                SlangLibraryErrorType::executable_path_failed, SLANG_FAIL,
                std::format("The executable path has no directory separator: {}", wide_to_utf8(executable_native)))};
    }

    executable_native.resize(separator + 1);

    auto const library_native = library_name.native();

    executable_native.append(library_native.data(), library_native.size());

    debug("[Slang Library] Executable path: {}", wide_to_utf8(executable_path_result->native()));

    debug("[Slang Library] Slang runtime path: {}", wide_to_utf8(executable_native));

    debug("[Slang Library] Slang path starts with UNC prefix: {}",
          executable_native.size() >= 2 && executable_native[0] == L'\\' && executable_native[1] == L'\\');

    return create(std::filesystem::path{std::move(executable_native)});
}

auto SlangLibrary::create_global_session(slang::IGlobalSession **session) const noexcept -> SlangResult {
    auto const description = SlangGlobalSessionDesc{};

    return create_global_session(description, session);
}

auto SlangLibrary::create_global_session(SlangGlobalSessionDesc const &description,
                                         slang::IGlobalSession **session) const noexcept -> SlangResult {
    if (session == nullptr) {
        return SLANG_E_INVALID_ARG;
    }

    *session = nullptr;

    if (!valid()) {
        return SLANG_E_NOT_AVAILABLE;
    }

    return impl_->create_global_session(&description, session);
}

auto SlangLibrary::valid() const noexcept -> bool {
    return impl_ != nullptr && impl_->module != nullptr && impl_->create_global_session != nullptr;
}

auto SlangLibrary::path() const noexcept -> std::filesystem::path const & {
    static auto const empty_path = std::filesystem::path{};

    if (impl_ == nullptr) {
        return empty_path;
    }

    return impl_->library_path;
}

auto SlangLibrary::destroy() noexcept -> void {
    if (impl_ == nullptr) {
        return;
    }

    /*
     * All Slang objects created through this DLL must already have been
     * released before FreeLibrary() is called.
     *
     * SlangCompiler should therefore declare SlangLibrary before its
     * global-session ComPtr, or explicitly reset the global session before
     * calling SlangLibrary::destroy().
     */
    impl_->create_global_session = nullptr;

    if (impl_->module != nullptr) {
        FreeLibrary(impl_->module);
        impl_->module = nullptr;
    }

    impl_->library_path.clear();

    impl_.reset();
}
