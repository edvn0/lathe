#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>

#include "scripting/lua_allocator.hxx"
#include "scripting/script_engine.hxx"

struct lua_State;
struct lua_Debug;

namespace scripting::detail {
    inline constexpr std::string_view chunk_name = "script";
    inline constexpr std::string_view invalid_entity_tag = "[invalid_entity] ";

    struct StringHash {
        using is_transparent = void;
        [[nodiscard]] auto operator()(std::string_view text) const noexcept -> std::size_t {
            return std::hash<std::string_view>{}(text);
        }
    };

    class EntityIndex {
    public:
        auto refresh(ScriptWorld world) -> bool;
        [[nodiscard]] auto find(std::string_view name) const -> entt::entity;
        [[nodiscard]] auto children(entt::entity parent) const -> std::span<entt::entity const>;
        [[nodiscard]] auto names() const noexcept -> std::span<std::string const>;
        auto clear() -> void;

    private:
        entt::registry const *registry_ = nullptr;
        std::uint64_t revision_ = 0;
        bool built_ = false;
        std::unordered_map<std::string, entt::entity, StringHash, std::equal_to<>> by_name_;
        std::unordered_map<entt::entity, std::vector<entt::entity>> children_;
        std::vector<std::string> names_;
    };

    struct RunContext {
        ScriptWorld world{};
        std::uint32_t world_generation = 0;
        std::chrono::steady_clock::time_point deadline{};
        std::chrono::milliseconds timeout{};
        bool timed_out = false;
        std::int32_t timeout_line = 0;
        std::uint32_t transforms_written = 0;
        std::uint32_t printed_lines = 0;
        std::uint32_t max_printed_lines = 256;
        int sandbox_ref = -2;
        LuaMemoryBudget *budget = nullptr;
        EntityIndex *entity_index = nullptr;
        std::uint64_t *entity_index_rebuilds = nullptr;
        std::mt19937 *random = nullptr;
    };

    [[nodiscard]] auto run_context(lua_State *state) noexcept -> RunContext &;

    auto build_sandbox(lua_State *state, RunContext &context) -> void;

    auto deadline_hook(lua_State *state, lua_Debug *activation) -> void;
    auto message_handler(lua_State *state) -> int;
    auto run_chunk_entry(lua_State *state) -> int;

    [[nodiscard]] auto bound_member_names() noexcept -> std::span<std::string_view const>;
}
