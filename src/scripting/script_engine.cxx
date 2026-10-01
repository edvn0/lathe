#include "scripting/script_engine.hxx"

#include "scripting/lua_api.hxx"

#include <sol/sol.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <format>
#include <functional>
#include <initializer_list>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"
#include "core/random.hxx"
#include "scripting/lua_allocator.hxx"
#include "scripting/script_runtime.hxx"

namespace {
    using scripting::detail::RunContext;

    struct LuaCloser {
        auto operator()(lua_State *state) const noexcept -> void { lua_close(state); }
    };

    // LRU of compiled chunks, keyed by the hash of their source.
    class ChunkCache {
    public:
        explicit ChunkCache(std::size_t capacity) : capacity_(std::max<std::size_t>(capacity, 1)) {}

        // Moves a hit to the front. A hash collision (same hash, different source) is a miss.
        [[nodiscard]] auto find(std::uint64_t hash, std::string_view source) -> sol::protected_function const * {
            auto const found = by_hash_.find(hash);
            if (found == by_hash_.end() || found->second->source != source) {
                return nullptr;
            }
            entries_.splice(entries_.begin(), entries_, found->second);
            return &entries_.front().chunk;
        }

        // Replaces an entry with the same hash; evicts the least recently used one past capacity.
        auto insert(std::uint64_t hash, std::string source, sol::protected_function chunk) -> void {
            if (auto const found = by_hash_.find(hash); found != by_hash_.end()) {
                entries_.erase(found->second);
                by_hash_.erase(found);
            }

            entries_.push_front(Entry{.hash = hash, .source = std::move(source), .chunk = std::move(chunk)});
            by_hash_[hash] = entries_.begin();

            while (entries_.size() > capacity_) {
                by_hash_.erase(entries_.back().hash);
                entries_.pop_back();
            }
        }

        auto clear() -> void {
            by_hash_.clear();
            entries_.clear();
        }

    private:
        struct Entry {
            std::uint64_t hash;
            std::string source;
            sol::protected_function chunk;
        };

        std::size_t capacity_;
        // Front is the most recently used.
        std::list<Entry> entries_;
        std::unordered_map<std::uint64_t, std::list<Entry>::iterator> by_hash_;
    };

    struct RunningGuard {
        explicit RunningGuard(bool &flag) noexcept : flag_(flag) { flag_ = true; }
        RunningGuard(RunningGuard const &) = delete;
        RunningGuard(RunningGuard &&) = delete;
        auto operator=(RunningGuard const &) -> RunningGuard & = delete;
        auto operator=(RunningGuard &&) -> RunningGuard & = delete;
        ~RunningGuard() { flag_ = false; }

    private:
        bool &flag_;
    };

    // Only reachable on a real out-of-memory (or a bug): everything outside run()'s protected call is far below
    // the memory cap.
    auto lua_panic(lua_State *state) -> int {
        char const *const message = lua_tostring(state, -1);
        logger::fatal("Lua panic: {}", message != nullptr ? std::string_view{message} : std::string_view{"?"});
        std::abort();
    }

    // Only used with LATHE_ENABLE_EXCEPTIONS=ON: turns a C++ exception escaping a binding into a runtime error.
    auto script_exception_handler(lua_State *state, sol::optional<std::exception const &> maybe_exception,
                                  sol::string_view description) -> int {
        std::array<char, 512> text{};
        auto const what = maybe_exception.has_value() ? std::string_view{maybe_exception->what()}
                                                      : std::string_view{description.data(), description.size()};
        auto const result =
                std::format_to_n(text.data(), static_cast<std::ptrdiff_t>(text.size()), "[cpp_exception] {}", what);
        lua_pushlstring(state, text.data(), static_cast<std::size_t>(result.out - text.data()));
        return 1;
    }

    [[nodiscard]] auto copy_error_string(lua_State *state, int index) -> std::string {
        if (lua_type(state, index) != LUA_TSTRING) {
            return "(error object is not a string)";
        }
        std::size_t length = 0;
        char const *const text = lua_tolstring(state, index, &length);
        return std::string{text, length};
    }

    // Priority: timeout > memory > syntax > tagged invalid_entity > runtime.
    [[nodiscard]] auto classify(int status, std::string_view raw_message, RunContext const &context,
                                LuaMemoryBudget const &budget, ScriptEngineSettings const &settings,
                                std::span<std::string const> api_names) -> ScriptError {
        if (context.timed_out) {
            return ScriptError{
                    .kind = ScriptErrorKind::timeout,
                    .message = std::format("exceeded the {} ms time budget", settings.timeout.count()),
                    .line = context.timeout_line > 0 ? std::optional<std::int32_t>{context.timeout_line} : std::nullopt,
            };
        }

        auto const location = split_lua_error_location(raw_message, scripting::detail::chunk_name);

        // luaL_Buffer's resizebox raises an allocation failure as a plain runtime error with this text.
        bool const buffer_out_of_memory = budget.limit_hit && location.text == "not enough memory";
        if (status == LUA_ERRMEM || (status == LUA_ERRERR && budget.limit_hit) || buffer_out_of_memory) {
            return ScriptError{
                    .kind = ScriptErrorKind::memory,
                    .message = std::format("exceeded the {} memory limit", human_readable_bytes(budget.limit_bytes)),
                    .line = location.line,
            };
        }

        if (status == LUA_ERRSYNTAX) {
            return ScriptError{
                    .kind = ScriptErrorKind::syntax, .message = std::string{location.text}, .line = location.line};
        }

        if (location.text.starts_with(scripting::detail::invalid_entity_tag)) {
            return ScriptError{
                    .kind = ScriptErrorKind::invalid_entity,
                    .message = std::string{location.text.substr(scripting::detail::invalid_entity_tag.size())},
                    .line = location.line,
            };
        }

        if (status == LUA_ERRERR) {
            return ScriptError{.kind = ScriptErrorKind::runtime, .message = "error in error handler"};
        }

        return ScriptError{
                .kind = ScriptErrorKind::runtime,
                .message = with_name_suggestion(location.text, api_names),
                .line = location.line,
        };
    }

    // Appends the string keys of the table at `index`.
    auto append_string_keys(lua_State *state, int index, std::vector<std::string> &names) -> void {
        index = lua_absindex(state, index);
        lua_pushnil(state);
        while (lua_next(state, index) != 0) {
            lua_pop(state, 1);
            if (lua_type(state, -1) == LUA_TSTRING) {
                std::size_t length = 0;
                char const *const key = lua_tolstring(state, -1, &length);
                names.emplace_back(key, length);
            }
        }
    }
} // namespace

struct ScriptEngine::Impl {
    explicit Impl(ScriptEngineSettings const &engine_settings) :
        settings(engine_settings), budget{.limit_bytes = engine_settings.memory_limit_bytes,
                                          .error_handler_headroom_bytes = engine_settings.error_handler_headroom_bytes},
        random(make_random_engine(engine_settings.random_stream)), chunks(engine_settings.chunk_cache_capacity) {}

    Impl(Impl const &) = delete;
    Impl(Impl &&) = delete;
    auto operator=(Impl const &) -> Impl & = delete;
    auto operator=(Impl &&) -> Impl & = delete;

    ~Impl() {
        if (!state) {
            return;
        }
        // __gc finalizers run during lua_close: make every VM instruction hit an expired deadline.
        context.deadline = std::chrono::steady_clock::time_point::min();
        lua_sethook(state.get(), &scripting::detail::deadline_hook, LUA_MASKCOUNT, 1);
        // References into the state are released while it is still open.
        chunks.clear();
        run_chunk = sol::protected_function{};
    }

    // Member order matters: destruction runs in reverse, so the budget and context outlive the state.
    ScriptEngineSettings settings;
    LuaMemoryBudget budget;
    std::mt19937 random;
    scripting::detail::EntityIndex entity_index;
    // *lua_getextraspace(state) == &context
    RunContext context;
    std::vector<std::string> api_names;
    // Closed after everything below is released.
    std::unique_ptr<lua_State, LuaCloser> state;
    // Wraps run_chunk_entry; calls through it use message_handler.
    sol::protected_function run_chunk;
    ChunkCache chunks;
    ScriptEngineStats stats;
    bool running = false;
};

auto ScriptEngine::create(ScriptEngineSettings const &settings) -> std::expected<ScriptEngine, ScriptError> {
    auto impl = std::make_unique<Impl>(settings);

    lua_State *const state = lua_newstate(&lua_budget_alloc, &impl->budget);
    if (state == nullptr) {
        return std::unexpected(
                ScriptError{.kind = ScriptErrorKind::memory, .message = "could not create the Lua state"});
    }
    impl->state.reset(state);

    auto &context = impl->context;
    context.timeout = settings.timeout;
    context.max_printed_lines = settings.max_print_lines_per_run;
    context.budget = &impl->budget;
    context.entity_index = &impl->entity_index;
    context.entity_index_rebuilds = &impl->stats.entity_index_rebuilds;
    context.random = &impl->random;
    *static_cast<RunContext **>(lua_getextraspace(state)) = &context;

    // What sol::state's constructor does: atpanic, sol2's per-state default handler global, exception handler.
    sol::set_default_state(state, &lua_panic);
    sol::set_default_exception_handler(state, &script_exception_handler);
    // Stored as a global inside this state, not a C++ static, so several engines can coexist.
    sol::protected_function::set_default_handler(sol::make_object(state, &scripting::detail::message_handler));

    scripting::detail::build_sandbox(state, context);
    // Also inspected below for api_names before it is moved into the registry.
    lua_pushvalue(state, -1);
    context.sandbox_ref = luaL_ref(state, LUA_REGISTRYINDEX);

    lua_pushcfunction(state, &scripting::detail::run_chunk_entry);
    impl->run_chunk = sol::protected_function(state, -1);
    lua_pop(state, 1);

    auto &names = impl->api_names;
    append_string_keys(state, -1, names);
    lua_pop(state, 1);
    for (auto const *const library: {LUA_STRLIBNAME, LUA_TABLIBNAME, LUA_MATHLIBNAME}) {
        lua_getglobal(state, library);
        append_string_keys(state, -1, names);
        lua_pop(state, 1);
    }
    for (auto const name: scripting::detail::bound_member_names()) {
        names.emplace_back(name);
    }
    std::ranges::sort(names);
    auto const duplicates = std::ranges::unique(names);
    names.erase(duplicates.begin(), duplicates.end());

    lua_settop(state, 0);
    lua_gc(state, LUA_GCCOLLECT);
    impl->budget.reset_run_flags();

    return ScriptEngine{std::move(impl)};
}

ScriptEngine::ScriptEngine(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

ScriptEngine::ScriptEngine(ScriptEngine &&) noexcept = default;

auto ScriptEngine::operator=(ScriptEngine &&) noexcept -> ScriptEngine & = default;

ScriptEngine::~ScriptEngine() = default;

auto ScriptEngine::run(std::string_view source, ScriptWorld world) -> ScriptRunReport {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::steady_clock;

    auto &self = *impl_;
    if (self.running) {
        return {.error = ScriptError{.kind = ScriptErrorKind::busy, .message = "a script is already running"}};
    }
    if (world.registry == nullptr) {
        return {.error = ScriptError{.kind = ScriptErrorKind::runtime, .message = "no scene"}};
    }

    // run() is the outermost C++ frame around the protected call, so RAII is safe here.
    RunningGuard const guard{self.running};
    auto const start = steady_clock::now();
    lua_State *const state = self.state.get();

    auto &context = self.context;
    // Entity refs from another registry go stale.
    if (context.world.registry != world.registry) {
        ++context.world_generation;
    }
    context.world = world;
    context.timeout = self.settings.timeout;
    context.deadline = start + self.settings.timeout;
    context.timed_out = false;
    context.timeout_line = 0;
    context.transforms_written = 0;
    context.printed_lines = 0;
    context.max_printed_lines = self.settings.max_print_lines_per_run;
    self.budget.limit_bytes = self.settings.memory_limit_bytes;
    self.budget.error_handler_headroom_bytes = self.settings.error_handler_headroom_bytes;
    self.budget.reset_run_flags();

    // Armed before compiling: GC steps during parsing can run __gc finalizers, which are Lua code.
    lua_sethook(state, &scripting::detail::deadline_hook, LUA_MASKCOUNT,
                static_cast<int>(self.settings.hook_instruction_interval));

    ScriptRunReport report;
    int status = LUA_OK;
    std::string raw_message;
    auto const hash = static_cast<std::uint64_t>(std::hash<std::string_view>{}(source));
    auto const *chunk = self.chunks.find(hash, source);
    report.chunk_cache_hit = chunk != nullptr;
    ++(chunk != nullptr ? self.stats.chunk_cache_hits : self.stats.chunk_cache_misses);

    if (chunk == nullptr) {
        // "t": text chunks only; binary chunks can crash the VM.
        status = luaL_loadbufferx(state, source.data(), source.size(), "=script", "t");
        if (status == LUA_OK) {
            // The cache's registry reference is taken outside any protected call, so allow it the error-handler
            // headroom: a raise here would be a panic.
            self.budget.in_error_handler = true;
            self.chunks.insert(hash, std::string{source}, sol::protected_function(state, -1));
            self.budget.in_error_handler = false;
            lua_pop(state, 1);
            chunk = self.chunks.find(hash, source);
        } else {
            raw_message = copy_error_string(state, -1);
            lua_pop(state, 1);
        }
    }

    if (status == LUA_OK) {
        // lua_pcall with message_handler; the result is popped at the end of this scope.
        auto const result = self.run_chunk(*chunk);
        status = static_cast<int>(result.status());
        if (status != LUA_OK) {
            raw_message = copy_error_string(state, result.stack_index());
        }
    }
    // In case the handler was interrupted.
    self.budget.in_error_handler = false;

    // Still under the deadline hook, which bounds finalizers.
    lua_gc(state, LUA_GCCOLLECT);
    lua_sethook(state, nullptr, 0, 0);
    lua_settop(state, 0);

    report.transforms_written = context.transforms_written;
    report.peak_bytes = self.budget.peak_bytes;
    if (status != LUA_OK || context.timed_out) {
        report.error = classify(status, raw_message, context, self.budget, self.settings, self.api_names);
    }
    report.duration = duration_cast<microseconds>(steady_clock::now() - start);
    ++self.stats.runs;
    return report;
}

auto ScriptEngine::is_running() const noexcept -> bool { return impl_->running; }

auto ScriptEngine::set_timeout(std::chrono::milliseconds timeout) noexcept -> void {
    impl_->settings.timeout = timeout;
}

auto ScriptEngine::set_memory_limit(std::size_t bytes) noexcept -> void { impl_->settings.memory_limit_bytes = bytes; }

auto ScriptEngine::settings() const noexcept -> ScriptEngineSettings const & { return impl_->settings; }

auto ScriptEngine::stats() const noexcept -> ScriptEngineStats {
    auto stats = impl_->stats;
    stats.lua_bytes_in_use = impl_->budget.used_bytes;
    stats.lua_peak_bytes = impl_->budget.peak_bytes;
    return stats;
}

auto ScriptEngine::api_names() const noexcept -> std::span<std::string const> { return impl_->api_names; }
