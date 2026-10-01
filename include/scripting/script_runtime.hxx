#pragma once

// Internal to engine_scripting: shared by entity_index.cxx, script_bindings.cxx and script_engine.cxx.

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

    // Name -> entity and parent -> children for one (registry, hierarchy_revision). Not HierarchyModel, which is
    // the editor panel's cache.
    class EntityIndex {
    public:
        // Rebuilds if `world` differs from the last build. Returns whether it rebuilt.
        auto refresh(ScriptWorld world) -> bool;
        // entt::null when absent. Duplicate names: lowest entity id wins.
        [[nodiscard]] auto find(std::string_view name) const -> entt::entity;
        // Sorted by entt::to_integral. Callers re-check valid() and Parent.
        [[nodiscard]] auto children(entt::entity parent) const -> std::span<entt::entity const>;
        // For "did you mean" on unknown names. Sorted.
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
        // Bumped when the registry changes; entity refs made under another generation are stale.
        std::uint32_t world_generation = 0;
        std::chrono::steady_clock::time_point deadline{};
        std::chrono::milliseconds timeout{};
        bool timed_out = false;
        std::int32_t timeout_line = 0;
        std::uint32_t transforms_written = 0;
        std::uint32_t printed_lines = 0;
        std::uint32_t max_printed_lines = 256;
        // LUA_NOREF until create() stores the sandbox table.
        int sandbox_ref = -2;
        LuaMemoryBudget *budget = nullptr;
        EntityIndex *entity_index = nullptr;
        std::uint64_t *entity_index_rebuilds = nullptr;
        std::mt19937 *random = nullptr;
    };

    // *static_cast<RunContext **>(lua_getextraspace(state))
    [[nodiscard]] auto run_context(lua_State *state) noexcept -> RunContext &;

    // Pushes the sandbox table: whitelisted base functions, print, read-only string/table/math, scene, Vec3; and
    // registers the lathe.Entity / lathe.Transform metatables.
    auto build_sandbox(lua_State *state, RunContext &context) -> void;

    auto deadline_hook(lua_State *state, lua_Debug *activation) -> void;
    auto message_handler(lua_State *state) -> int;
    auto run_chunk_entry(lua_State *state) -> int;

    // "get_entity", "get_children_or_empty", "get_transform", "translation", "x", "y", "z", "random", "new".
    [[nodiscard]] auto bound_member_names() noexcept -> std::span<std::string_view const>;
} // namespace scripting::detail
