#include "luagame/lua_http.hxx"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <deque>
#include <format>
#include <future>
#include <new>
#include <stop_token>
#include <string_view>
#include <utility>

#include "core/logger.hxx"
#include "core/paths.hxx"
#include "core/sha256.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    constexpr char const *request_meta = "lathe.HttpRequest";

    struct Request {
        enum class Kind : std::uint8_t { get, download };

        Kind kind = Kind::get;
        std::string uri;
        std::string sha256;
        std::filesystem::path path;
        std::stop_source stop;

        std::future<HttpResult> body_future;
        std::future<std::expected<void, HttpError>> file_future;

        bool finished = false;
        bool cancelled = false;
        bool succeeded = false;
        int status = 0;
        std::string body;
        std::string error;

        // Registry references that keep the request object and its callback alive while it is in flight.
        int self_ref = LUA_NOREF;
        int callback_ref = LUA_NOREF;
    };

    using RequestPointer = std::shared_ptr<Request>;

    auto describe(HttpError const &error) -> std::string {
        if (error.status_code != 0) {
            return std::format("{} (status {})", error.type, error.status_code);
        }

        return std::format("{}", error.type);
    }

    auto lowercase(std::string text) -> std::string {
        std::ranges::transform(text, text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        return text;
    }

    auto is_sha256(std::string_view text) -> bool {
        return text.size() == 64U && std::ranges::all_of(text, [](unsigned char c) { return std::isxdigit(c) != 0; });
    }

    // "https://host[:port]/..." -> host, or nothing for any other shape (no userinfo, no other scheme).
    auto url_host(std::string_view url) -> std::optional<std::string> {
        constexpr std::string_view scheme = "https://";

        if (!url.starts_with(scheme)) {
            return std::nullopt;
        }

        auto const rest = url.substr(scheme.size());
        auto const end = rest.find_first_of("/:?#");
        auto const host = rest.substr(0, end);

        if (host.empty() || host.find('@') != std::string_view::npos) {
            return std::nullopt;
        }

        return lowercase(std::string{host});
    }

    // A path under the cache directory: letters, digits, '.', '_', '-' and '/' between parts; no '..'.
    auto valid_cache_name(std::string_view name) -> bool {
        if (name.empty() || name.size() > 120U || name.front() == '/' || name.front() == '.' || name.back() == '/') {
            return false;
        }

        if (name.find("..") != std::string_view::npos || name.find("//") != std::string_view::npos) {
            return false;
        }

        return std::ranges::all_of(name, [](unsigned char c) {
            return std::isalnum(c) != 0 || c == '.' || c == '_' || c == '-' || c == '/';
        });
    }

    auto extension_of(std::string_view url) -> std::string {
        auto const query = url.find_first_of("?#");
        auto const clean = url.substr(0, query);
        auto const slash = clean.rfind('/');
        auto const dot = clean.rfind('.');

        if (dot == std::string_view::npos || (slash != std::string_view::npos && dot < slash) ||
            clean.size() - dot > 9U) {
            return {};
        }

        auto const extension = clean.substr(dot);

        return std::ranges::all_of(extension.substr(1), [](unsigned char c) { return std::isalnum(c) != 0; })
                       ? lowercase(std::string{extension})
                       : std::string{};
    }
}

struct LuaHttp::Impl {
    explicit Impl(LuaHttpOptions o) : options(std::move(o)) {}

    LuaHttpOptions options;

    // Declared before the requests so it outlives their futures.
    std::unique_ptr<HttpClient> client;
    std::deque<RequestPointer> in_flight;

    auto http_client() -> HttpClient & {
        if (!client) {
            HttpClientOptions client_options;

            client_options.worker_count = 2;
            client_options.timeout = std::chrono::seconds{60};
            client_options.max_response_bytes = options.max_download_bytes;
            client = std::make_unique<HttpClient>(std::move(client_options),
                                                  options.make_handler ? options.make_handler() : nullptr);
        }

        return *client;
    }

    auto host_allowed(std::string const &host) const -> bool {
        return std::ranges::find(options.allowed_hosts, host) != options.allowed_hosts.end();
    }

    // Where earlier versions kept script downloads.
    auto legacy_root() const -> std::filesystem::path { return cache_path("downloads/lua").absolute(); }

    // Where the download called 'name' with this hash goes. In the store it is keyed by the hash, so the same bytes are
    // kept once whatever scripts call them; the name only shapes the file's name.
    auto download_path(std::string const &sha256, std::string const &name) const -> std::filesystem::path {
        if (!options.cache_directory.empty()) {
            return options.cache_directory / name;
        }

        auto const downloads = options.downloads.enabled() ? options.downloads : DownloadCache::global();
        auto const stored = downloads.path(sha256, std::filesystem::path{name}.stem().string(),
                                           std::filesystem::path{name}.extension().string(), legacy_root() / name);

        return stored.empty() ? legacy_root() / name : stored;
    }

    // Reads `url` and the options table at `table_index` into a new request; an empty result means `problem` says why.
    auto create(Request::Kind kind, std::string url, lua_State *state, int table_index, std::string &problem)
            -> RequestPointer {
        auto const host = url_host(url);

        if (!host) {
            problem = "the url must be https:// with a host";

            return nullptr;
        }

        if (!host_allowed(*host)) {
            problem = std::format("scripts may not reach '{}'", *host);

            return nullptr;
        }

        if (in_flight.size() >= options.max_in_flight) {
            problem = "too many requests are in flight";

            return nullptr;
        }

        auto request = std::make_shared<Request>();

        request->kind = kind;
        request->uri = std::move(url);

        std::string name;

        if (lua_istable(state, table_index)) {
            lua_getfield(state, table_index, "sha256");

            if (lua_isstring(state, -1) != 0) {
                request->sha256 = lowercase(lua_tostring(state, -1));
            }

            lua_pop(state, 1);

            lua_getfield(state, table_index, "name");

            if (lua_isstring(state, -1) != 0) {
                name = lua_tostring(state, -1);
            }

            lua_pop(state, 1);

            lua_getfield(state, table_index, "on_done");

            if (lua_isfunction(state, -1)) {
                request->callback_ref = luaL_ref(state, LUA_REGISTRYINDEX);
            } else {
                lua_pop(state, 1);
            }
        }

        auto const fail = [&](std::string message) -> RequestPointer {
            problem = std::move(message);

            if (request->callback_ref != LUA_NOREF) {
                luaL_unref(state, LUA_REGISTRYINDEX, request->callback_ref);
                request->callback_ref = LUA_NOREF;
            }

            return nullptr;
        };

        if (!request->sha256.empty() && !is_sha256(request->sha256)) {
            return fail("sha256 must be 64 hexadecimal characters");
        }

        if (kind == Request::Kind::download) {
            if (request->sha256.empty()) {
                return fail("a download needs a sha256");
            }

            if (name.empty()) {
                name = request->sha256 + extension_of(request->uri);
            }

            if (!valid_cache_name(name)) {
                return fail("name must be a plain relative path of letters, digits, '.', '_' and '-'");
            }

            request->path = download_path(request->sha256, name);
        }

        return request;
    }

    auto start(Request &request) -> void {
        if (request.kind == Request::Kind::get) {
            request.body_future = http_client().get_async(request.uri, request.stop.get_token());
        } else {
            request.file_future = http_client().get_file_async(
                    HttpFileDownload{.uri = request.uri, .destination = request.path, .sha256 = request.sha256},
                    request.stop.get_token());
        }
    }

    auto ready(Request &request) -> bool {
        auto const poll = [](auto &future) {
            return future.valid() && future.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
        };

        return request.kind == Request::Kind::get ? poll(request.body_future) : poll(request.file_future);
    }

    auto collect(Request &request) -> void {
        request.finished = true;

        if (request.kind == Request::Kind::download) {
            auto result = request.file_future.get();

            if (result) {
                request.succeeded = true;
                request.status = 200;
            } else {
                request.error = describe(result.error());
                request.status = result.error().status_code;
            }

            return;
        }

        auto result = request.body_future.get();

        if (!result) {
            request.error = describe(result.error());
            request.status = result.error().status_code;

            return;
        }

        request.status = result->status_code;

        if (!result->is_success_status_code()) {
            request.error = std::format("unsuccessful_status_code (status {})", result->status_code);
        } else if (result->content.size() > options.max_body_bytes) {
            request.error = "response_too_large";
        } else if (!request.sha256.empty() && sha256_hex(result->content) != request.sha256) {
            request.error = "hash_mismatch";
        } else {
            request.body = result->content_as_string();
            request.succeeded = true;
        }
    }
};

namespace {
    auto request_arg(lua_State *state, int index) -> RequestPointer & {
        return *static_cast<RequestPointer *>(luaL_checkudata(state, index, request_meta));
    }

    auto impl_of(lua_State *state) -> LuaHttp::Impl & {
        return *static_cast<LuaHttp::Impl *>(lua_touserdata(state, lua_upvalueindex(1)));
    }

    // Shared by http.get and http.download. The message is copied into `message` and raised only after every C++
    // object is gone, because luaL_error does not unwind.
    auto make_request(lua_State *state, Request::Kind kind) -> int {
        auto &impl = impl_of(state);
        char message[256]{};

        {
            std::string url = luaL_checkstring(state, 1);
            std::string problem;
            auto request = impl.create(kind, std::move(url), state, 2, problem);

            if (!request) {
                std::strncpy(message, problem.c_str(), sizeof(message) - 1);
            } else {
                void *const block = lua_newuserdatauv(state, sizeof(RequestPointer), 0);

                new (block) RequestPointer{request};
                luaL_setmetatable(state, request_meta);

                lua_pushvalue(state, -1);
                request->self_ref = luaL_ref(state, LUA_REGISTRYINDEX);

                impl.start(*request);
                impl.in_flight.push_back(std::move(request));
            }
        }

        if (message[0] != '\0') {
            return luaL_error(state, "%s", message);
        }

        return 1;
    }

    auto http_get(lua_State *state) -> int { return make_request(state, Request::Kind::get); }

    auto http_download(lua_State *state) -> int { return make_request(state, Request::Kind::download); }

    auto http_pending(lua_State *state) -> int {
        lua_pushinteger(state, static_cast<lua_Integer>(impl_of(state).in_flight.size()));

        return 1;
    }

    auto request_gc(lua_State *state) -> int {
        static_cast<RequestPointer *>(lua_touserdata(state, 1))->~RequestPointer();

        return 0;
    }

    auto request_done(lua_State *state) -> int {
        lua_pushboolean(state, request_arg(state, 1)->finished ? 1 : 0);

        return 1;
    }

    auto request_ok(lua_State *state) -> int {
        auto const &request = *request_arg(state, 1);

        lua_pushboolean(state, request.finished && request.succeeded ? 1 : 0);

        return 1;
    }

    auto request_error(lua_State *state) -> int {
        auto const &request = *request_arg(state, 1);

        if (request.finished && !request.succeeded) {
            lua_pushstring(state, request.cancelled ? "canceled" : request.error.c_str());
        } else {
            lua_pushnil(state);
        }

        return 1;
    }

    auto request_status(lua_State *state) -> int {
        auto const &request = *request_arg(state, 1);

        lua_pushinteger(state, request.status);

        return 1;
    }

    auto request_body(lua_State *state) -> int {
        auto const &request = *request_arg(state, 1);

        if (request.finished && request.succeeded && request.kind == Request::Kind::get) {
            lua_pushlstring(state, request.body.data(), request.body.size());
        } else {
            lua_pushnil(state);
        }

        return 1;
    }

    auto request_path(lua_State *state) -> int {
        auto const &request = *request_arg(state, 1);

        if (request.finished && request.succeeded && request.kind == Request::Kind::download) {
            auto const text = request.path.generic_string();

            lua_pushlstring(state, text.data(), text.size());
        } else {
            lua_pushnil(state);
        }

        return 1;
    }

    auto request_cancel(lua_State *state) -> int {
        auto &request = *request_arg(state, 1);

        if (!request.finished) {
            request.cancelled = true;
            request.stop.request_stop();
        }

        return 0;
    }

    constexpr std::array request_methods{
            luaL_Reg{"done", &request_done},     luaL_Reg{"ok", &request_ok},
            luaL_Reg{"error", &request_error},   luaL_Reg{"status", &request_status},
            luaL_Reg{"body", &request_body},     luaL_Reg{"path", &request_path},
            luaL_Reg{"cancel", &request_cancel}, luaL_Reg{nullptr, nullptr},
    };
}

LuaHttp::LuaHttp(LuaHttpOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

LuaHttp::~LuaHttp() { reset(); }

auto LuaHttp::reset() -> void {
    for (auto &request: impl_->in_flight) {
        request->stop.request_stop();
    }

    impl_->in_flight.clear();
}

auto LuaHttp::pending() const noexcept -> std::size_t { return impl_->in_flight.size(); }

auto LuaHttp::open(LuaRuntime &runtime) -> void {
    auto *const state = runtime.state();

    if (luaL_newmetatable(state, request_meta) != 0) {
        lua_pushcfunction(state, &request_gc);
        lua_setfield(state, -2, "__gc");

        lua_createtable(state, 0, static_cast<int>(request_methods.size()));
        luaL_setfuncs(state, request_methods.data(), 0);
        lua_setfield(state, -2, "__index");
    }

    lua_pop(state, 1);

    lua_createtable(state, 0, 3);

    auto const add = [&](char const *name, lua_CFunction function) {
        lua_pushlightuserdata(state, impl_.get());
        lua_pushcclosure(state, function, 1);
        lua_setfield(state, -2, name);
    };

    add("get", &http_get);
    add("download", &http_download);
    add("pending", &http_pending);

    lua_setglobal(state, "http");
}

auto LuaHttp::update(LuaRuntime &runtime) -> void {
    auto &impl = *impl_;
    auto *const state = runtime.state();

    // A callback may start more requests, so work from a snapshot of what finished.
    std::vector<RequestPointer> finished;

    for (auto it = impl.in_flight.begin(); it != impl.in_flight.end();) {
        auto &request = **it;

        if (impl.ready(request)) {
            impl.collect(request);
            finished.push_back(std::move(*it));
            it = impl.in_flight.erase(it);
        } else {
            ++it;
        }
    }

    for (auto const &request: finished) {
        auto const callback = std::exchange(request->callback_ref, LUA_NOREF);
        auto const self = std::exchange(request->self_ref, LUA_NOREF);

        if (callback != LUA_NOREF && !request->cancelled) {
            lua_rawgeti(state, LUA_REGISTRYINDEX, callback);
            lua_rawgeti(state, LUA_REGISTRYINDEX, self);

            auto const result = runtime.protected_call(1, 0);

            if (!result.ok) {
                error("[lua] http on_done for {}: {}", request->uri, result.error);
            }
        }

        luaL_unref(state, LUA_REGISTRYINDEX, callback);
        luaL_unref(state, LUA_REGISTRYINDEX, self);
    }
}

auto lua_http_downloaded_path(lua_State *state, int index) -> std::optional<std::filesystem::path> {
    auto *const block = static_cast<RequestPointer *>(luaL_testudata(state, index, request_meta));

    if (block == nullptr) {
        return std::nullopt;
    }

    auto const &request = **block;

    if (request.kind != Request::Kind::download || !request.finished || !request.succeeded) {
        return std::nullopt;
    }

    return request.path;
}
