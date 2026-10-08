#include <doctest/doctest.h>

#include "core/sha256.hxx"
#include "net/http.hxx"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace {

    [[nodiscard]]
    auto bytes_of(std::string_view text) -> std::vector<std::byte> {
        auto const *begin = reinterpret_cast<std::byte const *>(text.data());
        return {begin, begin + text.size()};
    }

    class FakeHandler final : public IHttpMessageHandler {
    public:
        using Responder = std::function<HttpResult(HttpRequestMessage const &)>;

        FakeHandler(Responder responder, std::atomic<int> *calls) : responder_(std::move(responder)), calls_(calls) {}

        [[nodiscard]]
        auto send(HttpRequestMessage const &request, HttpClientOptions const &, std::stop_token) -> HttpResult override {
            ++*calls_;
            return responder_(request);
        }

    private:
        Responder responder_;
        std::atomic<int> *calls_;
    };

    auto ok_response(std::string_view body, int status = 200) -> HttpResult {
        return HttpResponseMessage{.status_code = status, .content = bytes_of(body)};
    }

    [[nodiscard]]
    auto temp_path(std::string_view name) -> std::filesystem::path {
        return std::filesystem::temp_directory_path() / "lathe_http_client_test" / name;
    }

}

TEST_SUITE("unit") {
    TEST_CASE("sha256_hex: known vectors") {
        CHECK(sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

        auto const abc = bytes_of("abc");
        CHECK(sha256_hex(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    TEST_CASE("HttpHeaders: lookup ignores case and returns the first match") {
        HttpHeaders headers;
        headers.add("Content-Type", "text/plain");
        headers.add("content-type", "ignored");

        CHECK(headers.find("CONTENT-TYPE") == "text/plain");
        CHECK_FALSE(headers.find("Accept").has_value());
    }

    TEST_CASE("HttpClient: get_async returns the handler's response") {
        std::atomic<int> calls = 0;
        HttpClient client{{}, std::make_unique<FakeHandler>([](HttpRequestMessage const &) { return ok_response("hi"); },
                                                            &calls)};

        auto const result = client.get_async("https://example.test/a").get();

        REQUIRE(result.has_value());
        CHECK(result->status_code == 200);
        CHECK(result->content_as_string() == "hi");
    }

    TEST_CASE("HttpClient: rejects plain http unless allowed, and non-http uris") {
        std::atomic<int> calls = 0;
        auto const responder = [](HttpRequestMessage const &) { return ok_response("hi"); };

        {
            HttpClient client{{}, std::make_unique<FakeHandler>(responder, &calls)};

            auto const insecure = client.get_async("http://example.test/").get();
            REQUIRE_FALSE(insecure.has_value());
            CHECK(insecure.error().type == HttpErrorType::insecure_scheme);

            auto const file = client.get_async("file:///etc/passwd").get();
            REQUIRE_FALSE(file.has_value());
            CHECK(file.error().type == HttpErrorType::invalid_uri);

            CHECK(calls == 0);
        }

        HttpClient permissive{{.allow_insecure_http = true}, std::make_unique<FakeHandler>(responder, &calls)};
        CHECK(permissive.get_async("http://example.test/").get().has_value());
    }

    TEST_CASE("HttpClient: default headers fill in but never override request headers") {
        std::atomic<int> calls = 0;
        HttpRequestMessage seen;
        HttpClientOptions options;
        options.default_request_headers.add("Accept", "default");
        options.default_request_headers.add("X-Extra", "extra");

        HttpClient client{options, std::make_unique<FakeHandler>(
                                           [&seen](HttpRequestMessage const &request) {
                                               seen = request;
                                               return ok_response("");
                                           },
                                           &calls)};

        HttpRequestMessage request{.uri = "https://example.test/"};
        request.headers.add("Accept", "mine");
        REQUIRE(client.send_async(std::move(request)).get().has_value());

        CHECK(seen.headers.find("Accept") == "mine");
        CHECK(seen.headers.find("X-Extra") == "extra");
    }

    TEST_CASE("HttpClient: get_byte_array_async fails on a non-2xx status") {
        std::atomic<int> calls = 0;
        HttpClient client{{}, std::make_unique<FakeHandler>(
                                      [](HttpRequestMessage const &) { return ok_response("nope", 404); }, &calls)};

        auto const result = client.get_byte_array_async("https://example.test/missing").get();

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == HttpErrorType::unsuccessful_status_code);
        CHECK(result.error().status_code == 404);
    }

    TEST_CASE("HttpClient: a cancelled request never reaches the handler") {
        std::atomic<int> calls = 0;
        HttpClient client{{}, std::make_unique<FakeHandler>([](HttpRequestMessage const &) { return ok_response(""); },
                                                            &calls)};

        std::stop_source source;
        source.request_stop();

        auto const result = client.get_async("https://example.test/", source.get_token()).get();

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().type == HttpErrorType::canceled);
        CHECK(calls == 0);
    }

    TEST_CASE("HttpClient: get_file_async keeps only a download whose hash matches") {
        auto const destination = temp_path("download.bin");
        std::filesystem::remove(destination);

        std::atomic<int> calls = 0;
        HttpClient client{{}, std::make_unique<FakeHandler>([](HttpRequestMessage const &) { return ok_response("abc"); },
                                                            &calls)};

        auto const good_hash = std::string{"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"};

        auto const wrong = client
                                   .get_file_async({.uri = "https://example.test/f",
                                                    .destination = destination,
                                                    .sha256 = std::string(64, '0')})
                                   .get();
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().type == HttpErrorType::hash_mismatch);
        CHECK_FALSE(std::filesystem::exists(destination));

        auto const right =
                client.get_file_async({.uri = "https://example.test/f", .destination = destination, .sha256 = good_hash})
                        .get();
        REQUIRE(right.has_value());
        CHECK(std::filesystem::exists(destination));

        auto const calls_before = calls.load();
        REQUIRE(client.get_file_async({.uri = "https://example.test/f", .destination = destination, .sha256 = good_hash})
                        .get()
                        .has_value());
        CHECK(calls == calls_before);

        std::filesystem::remove_all(destination.parent_path());
    }
}
