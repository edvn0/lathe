#include "slang_library.hxx"

#include <climits>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <format>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

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
    auto dl_error_message() -> std::string {
        auto const *message = dlerror();

        if (message == nullptr) {
            return "No dynamic linker error was reported.";
        }

        return std::string{message};
    }

    [[nodiscard]]
    auto executable_path() -> std::expected<std::filesystem::path, SlangLibraryError> {
        auto buffer = std::vector<char>(512, '\0');

        while (true) {
            errno = 0;
            auto const length = readlink("/proc/self/exe", buffer.data(), buffer.size());

            if (length < 0) {
                auto const error = errno;

                return std::unexpected{
                        make_error(SlangLibraryErrorType::executable_path_failed, SLANG_FAIL,
                                   std::format("readlink(\"/proc/self/exe\") failed: {}", std::strerror(error)))};
            }

            if (static_cast<std::size_t>(length) < buffer.size() - 1) {
                buffer.resize(static_cast<std::size_t>(length));

                return std::filesystem::path{
                        buffer.begin(),
                        buffer.end(),
                };
            }

            if (buffer.size() >= 32768) {
                return std::unexpected{make_error(SlangLibraryErrorType::executable_path_failed, SLANG_FAIL,
                                                  "The executable path exceeds the supported "
                                                  "path length.")};
            }

            buffer.resize(std::min<std::size_t>(buffer.size() * 2, 32768), '\0');
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
    void *handle = nullptr;

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

    auto resolved_path = library_path;

    debug("[Slang Library] Path passed to dlopen: {}", resolved_path.string());

    std::error_code status_error{};

    auto const status = std::filesystem::status(resolved_path, status_error);

    if (status_error) {
        return std::unexpected{make_error(SlangLibraryErrorType::library_not_found, SLANG_E_NOT_FOUND,
                                          std::format("The Slang runtime library was not found: {}\n"
                                                      "stat() failed: {}",
                                                      resolved_path.string(), status_error.message()))};
    }

    if (!std::filesystem::exists(status)) {
        return std::unexpected{
                make_error(SlangLibraryErrorType::library_not_found, SLANG_E_NOT_FOUND,
                           std::format("The Slang runtime library was not found: {}", resolved_path.string()))};
    }

    if (std::filesystem::is_directory(status)) {
        return std::unexpected{
                make_error(SlangLibraryErrorType::library_not_found, SLANG_E_NOT_FOUND,
                           std::format("The Slang runtime path refers to a directory: {}", resolved_path.string()))};
    }

    auto impl = std::make_unique<Impl>();

    auto absolute_result = absolute_path(resolved_path);
    if (!absolute_result) {
        return std::unexpected{std::move(absolute_result.error())};
    }

    dlerror(); // Clear any pending error.

    impl->handle = dlopen(absolute_result->c_str(), RTLD_NOW | RTLD_LOCAL);

    if (impl->handle == nullptr) {
        return std::unexpected{make_error(SlangLibraryErrorType::library_load_failed, SLANG_E_NOT_AVAILABLE,
                                          std::format("Failed to load the Slang runtime library '{}': {}",
                                                      absolute_result->string(), dl_error_message()))};
    }

    dlerror(); // Clear any pending error.

    auto *symbol = dlsym(impl->handle, "slang_createGlobalSession2");

    if (symbol == nullptr) {
        auto const error_message = dl_error_message();

        dlclose(impl->handle);
        impl->handle = nullptr;

        return std::unexpected{make_error(SlangLibraryErrorType::symbol_not_found, SLANG_E_NOT_FOUND,
                                          std::format("The Slang runtime library '{}' does not export "
                                                      "'slang_createGlobalSession2': {}",
                                                      absolute_result->string(), error_message))};
    }

    impl->create_global_session = std::bit_cast<SlangCreateGlobalSessionFunction>(symbol);

    impl->library_path = std::move(*absolute_result);

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

    auto directory = executable_path_result->parent_path();

    auto const library_path = directory / library_name;

    debug("[Slang Library] Executable path: {}", executable_path_result->string());

    debug("[Slang Library] Slang runtime path: {}", library_path.string());

    return create(library_path);
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
    return impl_ != nullptr && impl_->handle != nullptr && impl_->create_global_session != nullptr;
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

    impl_->create_global_session = nullptr;
    if (impl_->handle != nullptr) {
        dlclose(impl_->handle);
        impl_->handle = nullptr;
    }

    impl_->library_path.clear();

    impl_.reset();
}
