#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <entt/entity/fwd.hpp>

#include "assets/file_change_queue.hxx"
#include "assets/file_change_watcher.hxx"
#include "scene/script_property.hxx"
#include "scripting/lua_api.hxx"

class LuaRuntime;
class Scene;
struct LuaGameHost;

// behaviours: Lua files in assets/scripts/behaviours/ that return a class table,
//
//   return {
//       properties = { { name = "speed", type = "number", default = 3, min = 0, max = 10 } },
//       on_start = function(self) end,
//       on_update = function(self, dt) end,
//       on_destroy = function(self) end,
//   }
//
// An entity with Components::Behaviour gets an instance while the game plays: a table `self` whose metatable
// indexes the class, holding `self.entity` and one field per property (the entity's value or the default). A
// callback that raises an error is reported once and that instance stops; others carry on.
class BehaviourSystem {
public:
    using SourceLoader = std::function<std::optional<std::string>(std::string const &name)>;

    BehaviourSystem(LuaRuntime &runtime, LuaGameHost &host, SourceLoader loader);
    ~BehaviourSystem();

    BehaviourSystem(BehaviourSystem const &) = delete;
    auto operator=(BehaviourSystem const &) -> BehaviourSystem & = delete;

    // The scripts found on disk, by name.
    [[nodiscard]] auto script_names() -> std::span<std::string const>;

    // The script's declared properties, loading it the first time. Null when there is no such script; a script that
    // failed to load has `error` set.
    [[nodiscard]] auto schema(std::string const &name) -> ScriptSchema const *;

    // Reads every loaded script again now. This also happens by itself when a file changes (a file watcher reports it while
    // the editor or the game runs): running instances switch to the new class, gain fields for new
    // properties and lose those that were removed, and an instance stopped by an error runs again.
    auto reload() -> void;

    // Picks up edited, added and deleted script files now. Called every frame, so reloads happen in the editor too.
    auto poll() -> void;

    // Starts new instances (on_start), then calls on_update on each, for the entities of the host's scene.
    auto update(float delta_time) -> void;

    // Pushes the entity's instance of `name` (or nil when it has none).
    auto push_instance(entt::entity entity, std::string const &name) -> void;

    // Makes the instance now rather than at the next update; on_start still waits for it. False with a message
    // when the script can't be loaded.
    [[nodiscard]] auto create_instance(entt::entity entity) -> std::optional<std::string>;

    // Calls on_destroy on every instance and forgets them. For the end of a play session.
    auto stop() -> void;

    [[nodiscard]] auto last_error() const noexcept -> std::string const & { return last_error_; }

private:
    struct Class {
        ScriptSchema schema;
        int ref = LUA_NOREF;
        // The reference belongs to an older, working version of the class; the new file has a problem.
        bool keep_old_ref = false;
    };

    struct Instance {
        std::string script;
        int ref = LUA_NOREF;
        bool started = false;
        bool failed = false;
    };

    auto load_class(std::string const &name) -> Class &;
    auto reload_class(std::string const &name) -> void;
    auto poll_changes() -> void;
    [[nodiscard]] auto script_file(std::string const &name) const -> std::filesystem::path;
    auto remove_instance(entt::entity entity, bool call_destroy) -> void;
    auto call(Instance &instance, entt::entity entity, char const *callback, std::optional<float> argument) -> void;
    auto report(std::string message) -> void;

    LuaRuntime *runtime_;
    LuaGameHost *host_;
    SourceLoader loader_;

    std::unordered_map<std::string, Class> classes_;
    std::vector<std::string> names_;
    // Told by the file watcher (efsw) when a file in the behaviours folder changes.
    FileChangeQueue changes_;
    FileChangeWatcher watcher_;
    bool scanned_ = false;
    bool playing_ = false;

    std::unordered_map<entt::entity, Instance> instances_;
    std::vector<entt::entity> scratch_;
    std::string last_error_;
};
