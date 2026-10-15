#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/sha256.hxx"
#include "luagame/lua_http.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    auto bytes_of(std::string const &text) -> std::vector<std::byte> {
        std::vector<std::byte> bytes;

        for (auto const c: text) {
            bytes.push_back(static_cast<std::byte>(c));
        }

        return bytes;
    }

    auto hash_of(std::string const &text) -> std::string { return sha256_hex(bytes_of(text)); }

    // Answers requests from a table instead of a network, and counts how often it was asked.
    struct FakeServer {
        std::mutex mutex;
        std::map<std::string, std::pair<int, std::string>> responses;
        std::atomic<int> requests{0};
        std::atomic<bool> hold{false}; // while set, requests wait (until they are cancelled)

        struct Handler final : IHttpMessageHandler {
            explicit Handler(FakeServer &s) : server(s) {}

            auto send(HttpRequestMessage const &request, HttpClientOptions const &, std::stop_token stop)
                    -> HttpResult override {
                ++server.requests;

                while (server.hold.load() && !stop.stop_requested()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds{2});
                }

                if (stop.stop_requested()) {
                    return std::unexpected{HttpError{.type = HttpErrorType::canceled}};
                }

                std::scoped_lock const lock{server.mutex};
                auto const found = server.responses.find(request.uri);
                auto response = HttpResponseMessage{};

                if (found == server.responses.end()) {
                    response.status_code = 404;
                } else {
                    response.status_code = found->second.first;
                    response.content = bytes_of(found->second.second);
                }

                return response;
            }

            FakeServer &server;
        };
    };

    struct Harness {
        FakeServer server;
        std::filesystem::path cache = std::filesystem::temp_directory_path() / "lathe_lua_http_test";
        LuaRuntime runtime = [] {
            auto created = LuaRuntime::create();

            REQUIRE(created.has_value());

            return std::move(*created);
        }();
        LuaHttp http{[this] {
            std::filesystem::remove_all(cache);
            LuaHttpOptions options;

            options.cache_directory = cache;
            options.max_in_flight = 3;
            options.make_handler = [this]() -> std::unique_ptr<IHttpMessageHandler> {
                return std::make_unique<FakeServer::Handler>(server);
            };

            return options;
        }()};

        Harness() { http.open(runtime); }

        ~Harness() { std::filesystem::remove_all(cache); }

        auto run(std::string const &code) -> std::string {
            if (luaL_dostring(runtime.state(), code.c_str()) != LUA_OK) {
                std::string message = lua_tostring(runtime.state(), -1);

                lua_pop(runtime.state(), 1);

                return message;
            }

            return {};
        }

        auto global(std::string const &name) -> std::string {
            lua_getglobal(runtime.state(), name.c_str());

            std::string value = lua_isnil(runtime.state(), -1) ? "<nil>" : luaL_tolstring(runtime.state(), -1, nullptr);

            lua_pop(runtime.state(), lua_isnil(runtime.state(), -1) ? 1 : 2);

            return value;
        }

        // Updates until `condition` (a Lua expression) holds, as the engine does once a frame.
        auto wait_for(std::string const &condition) -> bool {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};

            while (std::chrono::steady_clock::now() < deadline) {
                http.update(runtime);

                if (run("RESULT = (" + condition + ") and true or false").empty() && global("RESULT") == "true") {
                    return true;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds{2});
            }

            return false;
        }
    };
}

TEST_CASE("lua http: get hands the body to on_done on the main thread") {
    Harness h;

    h.server.responses["https://raw.githubusercontent.com/a/b.txt"] = {200, "hello"};

    CHECK(h.run(R"(
        BODY = nil
        REQ = http.get("https://raw.githubusercontent.com/a/b.txt", {
            sha256 = ")" + hash_of("hello") + R"(",
            on_done = function(r) BODY = r:body(); STATUS = r:status() end,
        })
        EARLY = REQ:done()
    )").empty());

    CHECK(h.wait_for("BODY ~= nil"));
    CHECK(h.global("BODY") == "hello");
    CHECK(h.global("STATUS") == "200");
    CHECK(h.run("assert(REQ:ok() and REQ:done() and REQ:error() == nil and http.pending() == 0)").empty());
}

TEST_CASE("lua http: a body that does not match its sha256 is an error and has no body") {
    Harness h;

    h.server.responses["https://raw.githubusercontent.com/a/b.txt"] = {200, "tampered"};

    CHECK(h.run(R"(
        REQ = http.get("https://raw.githubusercontent.com/a/b.txt", { sha256 = ")" + hash_of("hello") + R"(" })
    )").empty());

    CHECK(h.wait_for("REQ:done()"));
    CHECK(h.run("assert(not REQ:ok() and REQ:body() == nil and REQ:error() == 'hash_mismatch', tostring(REQ:error()))")
                  .empty());
}

TEST_CASE("lua http: only https to allowed hosts, and only so many at once") {
    Harness h;

    CHECK(h.run("http.get('http://raw.githubusercontent.com/x')").find("https") != std::string::npos);
    CHECK(h.run("http.get('https://evil.example.com/x')").find("may not reach") != std::string::npos);
    CHECK(h.run("http.get('https://raw.githubusercontent.com@evil.example.com/x')").find("https") !=
          std::string::npos);
    CHECK(h.run("http.get('file:///etc/passwd')").find("https") != std::string::npos);
    CHECK(h.run("http.get('https://raw.githubusercontent.com/x', { sha256 = 'abc' })").find("64") !=
          std::string::npos);

    h.server.hold = true;

    CHECK(h.run("A = http.get('https://github.com/1'); B = http.get('https://github.com/2'); "
                "C = http.get('https://github.com/3')")
                  .empty());
    CHECK(h.run("http.get('https://github.com/4')").find("too many") != std::string::npos);

    h.server.hold = false;
    CHECK(h.wait_for("A:done() and B:done() and C:done()"));
}

TEST_CASE("lua http: a download is kept in the cache, checked, and reused") {
    Harness h;
    std::string const content = "model bytes";
    std::string const url = "https://raw.githubusercontent.com/a/model.glb";

    h.server.responses[url] = {200, content};

    CHECK(h.run("assert(not pcall(http.download, '" + url + "', {}))").empty()); // a download needs a sha256
    CHECK(h.run("assert(not pcall(http.download, '" + url + "', { sha256 = '" + hash_of(content) +
                "', name = '../escape' }))")
                  .empty());

    CHECK(h.run("D = http.download('" + url + "', { sha256 = '" + hash_of(content) + "', name = 'snake/model.glb' })")
                  .empty());
    CHECK(h.wait_for("D:done()"));
    CHECK(h.run("assert(D:ok(), tostring(D:error()))").empty());

    auto const expected = h.cache / "snake" / "model.glb";
    CHECK(h.run("PATH = D:path()").empty());
    CHECK(h.global("PATH").find("snake/model.glb") != std::string::npos);
    REQUIRE(std::filesystem::exists(expected));

    std::ifstream file{expected, std::ios::binary};
    std::string saved{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    CHECK(saved == content);
    CHECK(h.server.requests.load() == 1);

    // Asking again for a file that is already there and matches does not touch the network.
    CHECK(h.run("E = http.download('" + url + "', { sha256 = '" + hash_of(content) + "', name = 'snake/model.glb' })")
                  .empty());
    CHECK(h.wait_for("E:done()"));
    CHECK(h.run("assert(E:ok())").empty());
    CHECK(h.server.requests.load() == 1);
}

TEST_CASE("lua http: a download with the wrong hash fails and leaves no file") {
    Harness h;
    std::string const url = "https://raw.githubusercontent.com/a/model.glb";

    h.server.responses[url] = {200, "something else"};

    CHECK(h.run("D = http.download('" + url + "', { sha256 = '" + hash_of("model bytes") + "', name = 'bad.glb' })")
                  .empty());
    CHECK(h.wait_for("D:done()"));
    CHECK(h.run("assert(not D:ok() and D:path() == nil and D:error():find('hash_mismatch'), tostring(D:error()))")
                  .empty());
    CHECK_FALSE(std::filesystem::exists(h.cache / "bad.glb"));
}

TEST_CASE("lua http: a failing status and a cancelled request report why") {
    Harness h;

    CHECK(h.run("MISSING = http.get('https://raw.githubusercontent.com/nothing/here')").empty());
    CHECK(h.wait_for("MISSING:done()"));
    CHECK(h.run("assert(not MISSING:ok() and MISSING:status() == 404, tostring(MISSING:error()))").empty());

    h.server.hold = true;

    CHECK(h.run("CALLED = false; SLOW = http.get('https://github.com/slow', { on_done = function() CALLED = true end })")
                  .empty());
    CHECK(h.run("SLOW:cancel()").empty());
    CHECK(h.wait_for("SLOW:done()"));
    CHECK(h.run("assert(SLOW:error() == 'canceled' and not CALLED)").empty());
}

TEST_CASE("lua http: an error in on_done does not stop other requests finishing") {
    Harness h;

    h.server.responses["https://github.com/a"] = {200, "a"};
    h.server.responses["https://github.com/b"] = {200, "b"};

    CHECK(h.run(R"(
        GOT = 0
        http.get("https://github.com/a", { on_done = function() error("boom") end })
        http.get("https://github.com/b", { on_done = function(r) GOT = GOT + 1 end })
    )").empty());

    CHECK(h.wait_for("GOT == 1"));
}
