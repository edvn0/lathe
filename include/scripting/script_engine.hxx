#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <entt/entity/fwd.hpp>

#include "scripting/script_error.hxx"

struct ScriptEngineSettings {
    std::chrono::milliseconds timeout{50};
    std::size_t memory_limit_bytes = 64ULL * 1024 * 1024;
    std::size_t error_handler_headroom_bytes = 256ULL * 1024;
    std::uint32_t hook_instruction_interval = 1000;
    std::size_t chunk_cache_capacity = 32;
    std::uint32_t max_print_lines_per_run = 256;
    std::uint32_t random_stream = 0x5C1F7U;
};

struct ScriptWorld {
    entt::registry *registry = nullptr;
    std::uint64_t hierarchy_revision = 0;
};

struct ScriptRunReport {
    std::chrono::microseconds duration{};
    std::uint32_t transforms_written = 0;
    bool chunk_cache_hit = false;
    std::size_t peak_bytes = 0;
    std::optional<ScriptError> error;

    [[nodiscard]] auto ok() const noexcept -> bool { return !error.has_value(); }
};

struct ScriptEngineStats {
    std::uint64_t runs = 0;
    std::uint64_t chunk_cache_hits = 0;
    std::uint64_t chunk_cache_misses = 0;
    std::uint64_t entity_index_rebuilds = 0;
    std::size_t lua_bytes_in_use = 0;
    std::size_t lua_peak_bytes = 0;
};

class ScriptEngine {
public:
    [[nodiscard]] static auto
    create(ScriptEngineSettings const &settings = {}) -> std::expected<ScriptEngine, ScriptError>;

    ScriptEngine(ScriptEngine const &) = delete;
    auto operator=(ScriptEngine const &) -> ScriptEngine & = delete;
    ScriptEngine(ScriptEngine &&) noexcept;
    auto operator=(ScriptEngine &&) noexcept -> ScriptEngine &;
    ~ScriptEngine();

    [[nodiscard]] auto run(std::string_view source, ScriptWorld world) -> ScriptRunReport;

    [[nodiscard]] auto is_running() const noexcept -> bool;

    auto set_timeout(std::chrono::milliseconds timeout) noexcept -> void;
    auto set_memory_limit(std::size_t bytes) noexcept -> void;

    [[nodiscard]] auto settings() const noexcept -> ScriptEngineSettings const &;
    [[nodiscard]] auto stats() const noexcept -> ScriptEngineStats;

    [[nodiscard]] auto api_names() const noexcept -> std::span<std::string const>;

private:
    struct Impl;
    explicit ScriptEngine(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};
