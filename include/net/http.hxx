#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/error_context.hxx"

enum class HttpMethod : std::uint8_t {
    get,
    head,
    post,
    put,
    patch,
    del,
};

struct HttpHeader {
    std::string name;
    std::string value;
};

class HttpHeaders {
public:
    auto add(std::string name, std::string value) -> void;

    [[nodiscard]]
    auto find(std::string_view name) const noexcept -> std::optional<std::string_view>;

    [[nodiscard]]
    auto entries() const noexcept -> std::vector<HttpHeader> const & {
        return entries_;
    }

    auto clear() noexcept -> void { entries_.clear(); }

private:
    std::vector<HttpHeader> entries_;
};

struct HttpRequestMessage {
    HttpMethod method = HttpMethod::get;
    std::string uri;
    HttpHeaders headers;
    std::vector<std::byte> content;
};

struct HttpResponseMessage {
    int status_code = 0;
    HttpHeaders headers;
    std::vector<std::byte> content;

    [[nodiscard]]
    auto is_success_status_code() const noexcept -> bool {
        return status_code >= 200 && status_code < 300;
    }

    [[nodiscard]]
    auto content_as_string() const -> std::string {
        return {reinterpret_cast<char const *>(content.data()), content.size()};
    }
};

enum class HttpErrorType : std::uint8_t {
    invalid_uri,
    insecure_scheme,
    connection_failed,
    tls_failure,
    timeout,
    canceled,
    too_many_redirects,
    response_too_large,
    unsuccessful_status_code,
    io_error,
    hash_mismatch,
    backend_error,
};

struct HttpError {
    HttpErrorType type = HttpErrorType::backend_error;
    int status_code = 0;
    std::optional<ErrorCause> cause{std::nullopt};
};

template<>
struct std::formatter<HttpErrorType> : std::formatter<std::string_view> {
    constexpr auto format(HttpErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case HttpErrorType::invalid_uri:
                    return "invalid_uri";
                case HttpErrorType::insecure_scheme:
                    return "insecure_scheme";
                case HttpErrorType::connection_failed:
                    return "connection_failed";
                case HttpErrorType::tls_failure:
                    return "tls_failure";
                case HttpErrorType::timeout:
                    return "timeout";
                case HttpErrorType::canceled:
                    return "canceled";
                case HttpErrorType::too_many_redirects:
                    return "too_many_redirects";
                case HttpErrorType::response_too_large:
                    return "response_too_large";
                case HttpErrorType::unsuccessful_status_code:
                    return "unsuccessful_status_code";
                case HttpErrorType::io_error:
                    return "io_error";
                case HttpErrorType::hash_mismatch:
                    return "hash_mismatch";
                case HttpErrorType::backend_error:
                    return "backend_error";
            }

            return "unknown_http_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct HttpClientOptions {
    std::chrono::milliseconds timeout{std::chrono::seconds{30}};
    std::chrono::milliseconds connect_timeout{std::chrono::seconds{10}};
    std::size_t max_response_bytes = std::size_t{64} * 1024U * 1024U;
    long max_redirects = 5;
    bool allow_insecure_http = false;
    std::string user_agent = "lathe";
    HttpHeaders default_request_headers;
    std::filesystem::path ca_bundle;
    unsigned worker_count = 2;
};

using HttpResult = std::expected<HttpResponseMessage, HttpError>;

class IHttpMessageHandler {
public:
    IHttpMessageHandler() = default;
    IHttpMessageHandler(IHttpMessageHandler const &) = delete;
    IHttpMessageHandler(IHttpMessageHandler &&) = delete;
    auto operator=(IHttpMessageHandler const &) -> IHttpMessageHandler & = delete;
    auto operator=(IHttpMessageHandler &&) -> IHttpMessageHandler & = delete;
    virtual ~IHttpMessageHandler() = default;

    [[nodiscard]]
    virtual auto send(HttpRequestMessage const &request, HttpClientOptions const &options, std::stop_token stop)
            -> HttpResult = 0;
};

[[nodiscard]]
auto make_curl_http_handler() -> std::unique_ptr<IHttpMessageHandler>;

struct HttpFileDownload {
    std::string uri;
    std::filesystem::path destination;
    std::string sha256;
};

class HttpClient {
public:
    explicit HttpClient(HttpClientOptions options = {}, std::unique_ptr<IHttpMessageHandler> handler = nullptr);

    HttpClient(HttpClient const &) = delete;
    HttpClient(HttpClient &&) = delete;
    auto operator=(HttpClient const &) -> HttpClient & = delete;
    auto operator=(HttpClient &&) -> HttpClient & = delete;

    ~HttpClient();

    [[nodiscard]]
    auto options() const noexcept -> HttpClientOptions const &;

    [[nodiscard]]
    auto send_async(HttpRequestMessage request, std::stop_token stop = {}) -> std::future<HttpResult>;

    [[nodiscard]]
    auto get_async(std::string uri, std::stop_token stop = {}) -> std::future<HttpResult>;

    [[nodiscard]]
    auto post_async(std::string uri, std::vector<std::byte> content, std::string content_type,
                    std::stop_token stop = {}) -> std::future<HttpResult>;

    [[nodiscard]]
    auto get_byte_array_async(std::string uri, std::stop_token stop = {})
            -> std::future<std::expected<std::vector<std::byte>, HttpError>>;

    [[nodiscard]]
    auto get_file_async(HttpFileDownload download, std::stop_token stop = {})
            -> std::future<std::expected<void, HttpError>>;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
