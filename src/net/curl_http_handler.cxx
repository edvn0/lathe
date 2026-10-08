#include <curl/curl.h>

#include <array>
#include <filesystem>
#include <mutex>
#include <string>

#include "net/http.hxx"

namespace {

    auto ensure_curl_initialised() -> void {
        static std::once_flag once;
        std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    struct EasyDeleter {
        auto operator()(CURL *easy) const noexcept -> void { curl_easy_cleanup(easy); }
    };

    struct HeaderListDeleter {
        auto operator()(curl_slist *list) const noexcept -> void { curl_slist_free_all(list); }
    };

    struct Transfer {
        HttpResponseMessage response;
        std::size_t max_bytes = 0;
        bool too_large = false;
        std::stop_token stop;
    };

    auto write_body(char *data, std::size_t size, std::size_t count, void *user) -> std::size_t {
        auto &transfer = *static_cast<Transfer *>(user);
        auto const bytes = size * count;

        if (transfer.response.content.size() + bytes > transfer.max_bytes) {
            transfer.too_large = true;
            return 0;
        }

        auto const *begin = reinterpret_cast<std::byte const *>(data);
        transfer.response.content.insert(transfer.response.content.end(), begin, begin + bytes);

        return bytes;
    }

    auto read_header(char *data, std::size_t size, std::size_t count, void *user) -> std::size_t {
        auto &transfer = *static_cast<Transfer *>(user);
        std::string_view line{data, size * count};

        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.remove_suffix(1);
        }

        if (line.starts_with("HTTP/")) {
            transfer.response.headers.clear();
        } else if (auto const colon = line.find(':'); colon != std::string_view::npos) {
            auto value = line.substr(colon + 1);

            while (!value.empty() && value.front() == ' ') {
                value.remove_prefix(1);
            }

            transfer.response.headers.add(std::string{line.substr(0, colon)}, std::string{value});
        }

        return size * count;
    }

    auto on_progress(void *user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
        return static_cast<Transfer *>(user)->stop.stop_requested() ? 1 : 0;
    }

    [[nodiscard]]
    auto find_ca_bundle(HttpClientOptions const &options) -> std::filesystem::path {
        if (!options.ca_bundle.empty()) {
            return options.ca_bundle;
        }

        constexpr std::array<char const *, 4> candidates{
                "/etc/ssl/certs/ca-certificates.crt",
                "/etc/pki/tls/certs/ca-bundle.crt",
                "/etc/ssl/ca-bundle.pem",
                "/etc/ssl/cert.pem",
        };

        std::error_code ec;

        for (auto const *candidate: candidates) {
            if (std::filesystem::exists(candidate, ec)) {
                return candidate;
            }
        }

        return {};
    }

    [[nodiscard]]
    auto make_error(HttpErrorType type, CURLcode code, char const *detail) -> HttpError {
        return HttpError{
                .type = type,
                .cause = ErrorContext{.message = FlyString{curl_easy_strerror(code)}, .diagnostics = detail},
        };
    }

    [[nodiscard]]
    auto classify(CURLcode code, Transfer const &transfer) -> HttpErrorType {
        if (transfer.too_large) {
            return HttpErrorType::response_too_large;
        }

        switch (code) {
            case CURLE_ABORTED_BY_CALLBACK:
                return HttpErrorType::canceled;
            case CURLE_OPERATION_TIMEDOUT:
                return HttpErrorType::timeout;
            case CURLE_TOO_MANY_REDIRECTS:
                return HttpErrorType::too_many_redirects;
            case CURLE_UNSUPPORTED_PROTOCOL:
                return HttpErrorType::insecure_scheme;
            case CURLE_URL_MALFORMAT:
                return HttpErrorType::invalid_uri;
            case CURLE_FILESIZE_EXCEEDED:
                return HttpErrorType::response_too_large;
            case CURLE_COULDNT_RESOLVE_HOST:
            case CURLE_COULDNT_RESOLVE_PROXY:
            case CURLE_COULDNT_CONNECT:
            case CURLE_SEND_ERROR:
            case CURLE_RECV_ERROR:
            case CURLE_GOT_NOTHING:
                return HttpErrorType::connection_failed;
            case CURLE_SSL_CONNECT_ERROR:
            case CURLE_PEER_FAILED_VERIFICATION:
            case CURLE_SSL_CACERT_BADFILE:
            case CURLE_SSL_CERTPROBLEM:
            case CURLE_SSL_CIPHER:
            case CURLE_SSL_ISSUER_ERROR:
                return HttpErrorType::tls_failure;
            default:
                return HttpErrorType::backend_error;
        }
    }

    class CurlHttpHandler final : public IHttpMessageHandler {
    public:
        CurlHttpHandler() { ensure_curl_initialised(); }

        [[nodiscard]]
        auto send(HttpRequestMessage const &request, HttpClientOptions const &options, std::stop_token stop)
                -> HttpResult override {
            std::unique_ptr<CURL, EasyDeleter> const easy{curl_easy_init()};

            if (!easy) {
                return std::unexpected{make_error(HttpErrorType::backend_error, CURLE_FAILED_INIT, "curl_easy_init")};
            }

            Transfer transfer;
            transfer.max_bytes = options.max_response_bytes;
            transfer.stop = std::move(stop);

            auto const *protocols = options.allow_insecure_http ? "http,https" : "https";
            std::array<char, CURL_ERROR_SIZE> error_buffer{};

            auto *handle = easy.get();
            curl_easy_setopt(handle, CURLOPT_URL, request.uri.c_str());
            curl_easy_setopt(handle, CURLOPT_PROTOCOLS_STR, protocols);
            curl_easy_setopt(handle, CURLOPT_REDIR_PROTOCOLS_STR, protocols);
            curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(handle, CURLOPT_MAXREDIRS, options.max_redirects);
            curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, static_cast<long>(options.timeout.count()));
            curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(options.connect_timeout.count()));
            curl_easy_setopt(handle, CURLOPT_USERAGENT, options.user_agent.c_str());
            curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 2L);
            curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, error_buffer.data());
            curl_easy_setopt(handle, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(options.max_response_bytes));

            if (auto const bundle = find_ca_bundle(options); !bundle.empty()) {
                curl_easy_setopt(handle, CURLOPT_CAINFO, bundle.c_str());
            }

            curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_body);
            curl_easy_setopt(handle, CURLOPT_WRITEDATA, &transfer);
            curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, read_header);
            curl_easy_setopt(handle, CURLOPT_HEADERDATA, &transfer);
            curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, on_progress);
            curl_easy_setopt(handle, CURLOPT_XFERINFODATA, &transfer);
            curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);

            switch (request.method) {
                case HttpMethod::get:
                    curl_easy_setopt(handle, CURLOPT_HTTPGET, 1L);
                    break;
                case HttpMethod::head:
                    curl_easy_setopt(handle, CURLOPT_NOBODY, 1L);
                    break;
                case HttpMethod::post:
                    curl_easy_setopt(handle, CURLOPT_POST, 1L);
                    break;
                case HttpMethod::put:
                    curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, "PUT");
                    break;
                case HttpMethod::patch:
                    curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, "PATCH");
                    break;
                case HttpMethod::del:
                    curl_easy_setopt(handle, CURLOPT_CUSTOMREQUEST, "DELETE");
                    break;
            }

            if (!request.content.empty()) {
                curl_easy_setopt(handle, CURLOPT_POSTFIELDS, reinterpret_cast<char const *>(request.content.data()));
                curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.content.size()));
            }

            std::unique_ptr<curl_slist, HeaderListDeleter> header_list;

            for (auto const &header: request.headers.entries()) {
                auto const line = header.name + ": " + header.value;

                if (auto *appended = curl_slist_append(header_list.get(), line.c_str()); appended != nullptr) {
                    static_cast<void>(header_list.release());
                    header_list.reset(appended);
                }
            }

            if (header_list) {
                curl_easy_setopt(handle, CURLOPT_HTTPHEADER, header_list.get());
            }

            auto const code = curl_easy_perform(handle);

            if (code != CURLE_OK) {
                return std::unexpected{make_error(classify(code, transfer), code, error_buffer.data())};
            }

            long status = 0;
            curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
            transfer.response.status_code = static_cast<int>(status);

            return std::move(transfer.response);
        }
    };

}

auto make_curl_http_handler() -> std::unique_ptr<IHttpMessageHandler> {
    return std::make_unique<CurlHttpHandler>();
}
