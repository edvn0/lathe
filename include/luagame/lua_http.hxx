#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "net/http.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;

struct LuaHttpOptions {
    // Scripts may only reach these hosts (https only; the HttpClient refuses plain http). A redirect may leave the
    // list, so a request that matters should carry a sha256, which downloads always do.
    std::vector<std::string> allowed_hosts{"raw.githubusercontent.com", "github.com", "objects.githubusercontent.com"};

    // Where downloads are kept; empty means the engine's cache directory ("downloads/lua").
    std::filesystem::path cache_directory;

    std::size_t max_in_flight = 8;
    std::size_t max_body_bytes = std::size_t{4} * 1024U * 1024U;

    // Replaces the libcurl handler; tests use it to answer without a network.
    std::function<std::unique_ptr<IHttpMessageHandler>()> make_handler;
};

// The `http` library for scripts: asynchronous requests whose results are handed back on the main thread.
//
//   local request = http.get(url [, { sha256 = "...", on_done = function(request) end }])
//   local request = http.download(url, { sha256 = "...", [name = "dir/file.glb",] [on_done = ...] })
//
// A request has done(), ok(), error(), status(), body() (get), path() (download) and cancel(). `on_done` is called
// from update() once the request has finished, so scripts never see another thread. A download needs a sha256: the
// file is kept in the cache, reused when it already matches, and removed from the picture if it does not. assets.load_model
// accepts a finished download in place of an asset path.
class LuaHttp {
public:
    explicit LuaHttp(LuaHttpOptions options = {});
    ~LuaHttp();

    LuaHttp(LuaHttp const &) = delete;
    auto operator=(LuaHttp const &) -> LuaHttp & = delete;
    LuaHttp(LuaHttp &&) = delete;
    auto operator=(LuaHttp &&) -> LuaHttp & = delete;

    // Registers the `http` global in the runtime's state. This object must outlive the state.
    auto open(LuaRuntime &runtime) -> void;

    // Collects finished requests and calls their on_done functions (under the runtime's call budget).
    auto update(LuaRuntime &runtime) -> void;

    // Cancels everything in flight and forgets it; call before the Lua state is replaced.
    auto reset() -> void;

    [[nodiscard]] auto pending() const noexcept -> std::size_t;

    struct Impl; // public so the library's functions in lua_http.cxx can reach it; nothing else defines it

private:
    std::unique_ptr<Impl> impl_;
};

// The file a finished, successful http.download request at `index` saved, for assets.load_model.
[[nodiscard]] auto lua_http_downloaded_path(lua_State *state, int index) -> std::optional<std::filesystem::path>;
