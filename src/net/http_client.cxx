#include "net/http.hxx"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/sha256.hxx"

namespace {

    [[nodiscard]]
    auto equals_ignore_case(std::string_view a, std::string_view b) noexcept -> bool {
        return std::ranges::equal(a, b, [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right));
        });
    }

    [[nodiscard]]
    auto http_error(HttpErrorType type, std::string_view message, int status_code = 0) -> HttpError {
        return HttpError{
                .type = type,
                .status_code = status_code,
                .cause = ErrorContext{.message = FlyString{message}},
        };
    }

    // The scheme of an absolute http(s) URI, lowercased; empty for anything else, including a missing host.
    [[nodiscard]]
    auto http_scheme(std::string_view uri) -> std::string {
        auto const separator = uri.find("://");

        if (separator == std::string_view::npos || separator + 3 >= uri.size()) {
            return {};
        }

        std::string scheme{uri.substr(0, separator)};
        std::ranges::transform(scheme, scheme.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });

        return scheme == "http" || scheme == "https" ? scheme : std::string{};
    }

    [[nodiscard]]
    auto validate_uri(std::string_view uri, bool allow_insecure_http) -> std::expected<void, HttpError> {
        auto const scheme = http_scheme(uri);

        if (scheme.empty()) {
            return std::unexpected{http_error(HttpErrorType::invalid_uri, "uri is not an absolute http(s) URI")};
        }

        if (scheme == "http" && !allow_insecure_http) {
            return std::unexpected{http_error(HttpErrorType::insecure_scheme, "plain http is not allowed")};
        }

        return {};
    }

    [[nodiscard]]
    auto read_file(std::filesystem::path const &path) -> std::optional<std::vector<std::byte>> {
        std::ifstream file{path, std::ios::binary | std::ios::ate};

        if (!file) {
            return std::nullopt;
        }

        auto const size = file.tellg();

        if (size < 0) {
            return std::nullopt;
        }

        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(bytes.data()), size);

        if (!file) {
            return std::nullopt;
        }

        return bytes;
    }

    // Writes next to `destination` and renames over it, so readers see the whole file or none of it.
    [[nodiscard]]
    auto write_file_atomically(std::filesystem::path const &destination, std::span<std::byte const> bytes) -> bool {
        std::error_code ec;

        if (destination.has_parent_path()) {
            std::filesystem::create_directories(destination.parent_path(), ec);

            if (ec) {
                return false;
            }
        }

        auto temporary = destination;
        temporary += ".part";

        {
            std::ofstream file{temporary, std::ios::binary | std::ios::trunc};
            file.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            file.flush();

            if (!file) {
                std::filesystem::remove(temporary, ec);
                return false;
            }
        }

        std::filesystem::rename(temporary, destination, ec);

        if (ec) {
            std::filesystem::remove(temporary, ec);
            return false;
        }

        return true;
    }

    [[nodiscard]]
    auto lowercase(std::string value) -> std::string {
        std::ranges::transform(value, value.begin(), [](char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

} // namespace

auto HttpHeaders::add(std::string name, std::string value) -> void {
    entries_.push_back(HttpHeader{.name = std::move(name), .value = std::move(value)});
}

auto HttpHeaders::find(std::string_view name) const noexcept -> std::optional<std::string_view> {
    auto const it =
            std::ranges::find_if(entries_, [&](HttpHeader const &h) { return equals_ignore_case(h.name, name); });

    if (it == entries_.end()) {
        return std::nullopt;
    }

    return std::string_view{it->value};
}

class HttpClient::Impl {
public:
    Impl(HttpClientOptions options_in, std::unique_ptr<IHttpMessageHandler> handler_in) :
        options(std::move(options_in)), handler(handler_in ? std::move(handler_in) : make_curl_http_handler()) {
        auto const count = std::max(1U, options.worker_count);

        workers.reserve(count);

        for (unsigned i = 0; i < count; ++i) {
            workers.emplace_back([this](std::stop_token stop) { run_worker(stop); });
        }
    }

    ~Impl() {
        // In-flight requests see this through the per-request stop source; queued ones are failed below.
        shutdown.request_stop();

        for (auto &worker: workers) {
            worker.request_stop();
        }

        wake.notify_all();

        // jthread joins on destruction.
        workers.clear();

        for (auto &task: queue) {
            task(true);
        }
    }

    Impl(Impl const &) = delete;
    Impl(Impl &&) = delete;
    auto operator=(Impl const &) -> Impl & = delete;
    auto operator=(Impl &&) -> Impl & = delete;

    // Runs `work(stop)` on a worker and delivers its result, or `canceled` if the client shuts down or `stop` fires
    // first. `Result` must be constructible from an HttpError wrapped in std::unexpected.
    template<class Result, class Work>
    [[nodiscard]]
    auto enqueue(std::stop_token user_stop, Work work) -> std::future<Result> {
        auto promise = std::make_shared<std::promise<Result>>();
        auto future = promise->get_future();

        auto task = [this, promise, user_stop = std::move(user_stop), work = std::move(work)](bool cancelled) mutable {
            if (cancelled || user_stop.stop_requested()) {
                promise->set_value(std::unexpected{http_error(HttpErrorType::canceled, "request canceled")});
                return;
            }

            // One token for the handler that fires on either the caller's or the client's stop.
            std::stop_source request_stop;
            std::stop_callback const on_user_stop{user_stop, [&request_stop] { request_stop.request_stop(); }};
            std::stop_callback const on_shutdown{shutdown.get_token(),
                                                 [&request_stop] { request_stop.request_stop(); }};

            promise->set_value(work(request_stop.get_token()));
        };

        {
            std::scoped_lock const lock{mutex};
            queue.emplace_back(std::move(task));
        }

        wake.notify_one();

        return future;
    }

    [[nodiscard]]
    auto send(HttpRequestMessage request, std::stop_token stop) -> HttpResult {
        if (auto valid = validate_uri(request.uri, options.allow_insecure_http); !valid) {
            return std::unexpected{valid.error()};
        }

        for (auto const &header: options.default_request_headers.entries()) {
            if (!request.headers.find(header.name)) {
                request.headers.add(header.name, header.value);
            }
        }

        return handler->send(request, options, std::move(stop));
    }

    HttpClientOptions options;
    std::unique_ptr<IHttpMessageHandler> handler;

private:
    auto run_worker(std::stop_token worker_stop) -> void {
        while (true) {
            std::move_only_function<void(bool)> task;

            {
                std::unique_lock lock{mutex};

                if (!wake.wait(lock, worker_stop, [this] { return !queue.empty(); })) {
                    return;
                }

                task = std::move(queue.front());
                queue.pop_front();
            }

            task(false);
        }
    }

    std::mutex mutex;
    std::condition_variable_any wake;
    std::deque<std::move_only_function<void(bool)>> queue;
    std::stop_source shutdown;
    // Declared last: workers use everything above, so they must be joined first.
    std::vector<std::jthread> workers;
};

HttpClient::HttpClient(HttpClientOptions options, std::unique_ptr<IHttpMessageHandler> handler) :
    impl_(std::make_unique<Impl>(std::move(options), std::move(handler))) {}

HttpClient::~HttpClient() = default;

auto HttpClient::options() const noexcept -> HttpClientOptions const & { return impl_->options; }

auto HttpClient::send_async(HttpRequestMessage request, std::stop_token stop) -> std::future<HttpResult> {
    return impl_->enqueue<HttpResult>(std::move(stop), [this, request = std::move(request)](std::stop_token token) {
        return impl_->send(request, std::move(token));
    });
}

auto HttpClient::get_async(std::string uri, std::stop_token stop) -> std::future<HttpResult> {
    return send_async(HttpRequestMessage{.method = HttpMethod::get, .uri = std::move(uri)}, std::move(stop));
}

auto HttpClient::post_async(std::string uri, std::vector<std::byte> content, std::string content_type,
                            std::stop_token stop) -> std::future<HttpResult> {
    HttpRequestMessage request{.method = HttpMethod::post, .uri = std::move(uri), .content = std::move(content)};
    request.headers.add("Content-Type", std::move(content_type));

    return send_async(std::move(request), std::move(stop));
}

auto HttpClient::get_byte_array_async(std::string uri, std::stop_token stop)
        -> std::future<std::expected<std::vector<std::byte>, HttpError>> {
    using Result = std::expected<std::vector<std::byte>, HttpError>;

    return impl_->enqueue<Result>(std::move(stop), [this, uri = std::move(uri)](std::stop_token token) -> Result {
        auto response = impl_->send(HttpRequestMessage{.method = HttpMethod::get, .uri = uri}, std::move(token));

        if (!response) {
            return std::unexpected{response.error()};
        }

        if (!response->is_success_status_code()) {
            return std::unexpected{http_error(HttpErrorType::unsuccessful_status_code,
                                              "server answered with a non-2xx status", response->status_code)};
        }

        return std::move(response->content);
    });
}

auto HttpClient::get_file_async(HttpFileDownload download, std::stop_token stop)
        -> std::future<std::expected<void, HttpError>> {
    using Result = std::expected<void, HttpError>;

    return impl_->enqueue<Result>(
            std::move(stop), [this, download = std::move(download)](std::stop_token token) -> Result {
                auto const expected_hash = lowercase(download.sha256);

                if (expected_hash.size() != 64U) {
                    return std::unexpected{
                            http_error(HttpErrorType::hash_mismatch, "sha256 must be 64 hex characters")};
                }

                if (auto const existing = read_file(download.destination);
                    existing && sha256_hex(*existing) == expected_hash) {
                    return {};
                }

                auto response = impl_->send(HttpRequestMessage{.method = HttpMethod::get, .uri = download.uri},
                                            std::move(token));

                if (!response) {
                    return std::unexpected{response.error()};
                }

                if (!response->is_success_status_code()) {
                    return std::unexpected{http_error(HttpErrorType::unsuccessful_status_code,
                                                      "server answered with a non-2xx status", response->status_code)};
                }

                if (sha256_hex(response->content) != expected_hash) {
                    return std::unexpected{http_error(HttpErrorType::hash_mismatch,
                                                      "downloaded file does not match the expected sha256")};
                }

                if (!write_file_atomically(download.destination, response->content)) {
                    return std::unexpected{http_error(HttpErrorType::io_error, "could not write the downloaded file")};
                }

                return {};
            });
}
