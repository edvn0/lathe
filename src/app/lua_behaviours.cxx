#include "app/lua_behaviours.hxx"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <system_error>

#include <entt/entt.hpp>

#include "app/lua_game.hxx"
#include "app/lua_game_api.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/toast.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"
#include "scripting/lua_runtime.hxx"
#include "scripting/lua_script_schema.hxx"

namespace {
    constexpr std::string_view behaviours_folder = "assets/scripts/behaviours";

    // Toasts are small; the traceback stays in the log and the inspector.
    auto first_line(std::string const &text) -> std::string { return text.substr(0, text.find('\n')); }

    auto valid_script_name(std::string const &name) -> bool {
        return !name.empty() && name.size() <= 64 && std::ranges::all_of(name, [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        });
    }
}

BehaviourSystem::BehaviourSystem(LuaRuntime &runtime, LuaGameHost &host, SourceLoader loader)
    : runtime_(&runtime), host_(&host), loader_(std::move(loader)) {
    if (auto const folder = Paths::current().data(std::string{behaviours_folder})) {
        std::array const directories{folder->absolute()};

        static_cast<void>(watcher_.start(changes_, directories));
    }
}

// The Lua state, and every reference into it, goes away with the runtime.
BehaviourSystem::~BehaviourSystem() = default;

auto BehaviourSystem::script_names() -> std::span<std::string const> {
    if (!scanned_) {
        scanned_ = true;
        names_.clear();

        if (auto const folder = Paths::current().data(std::string{behaviours_folder})) {
            std::error_code ignored;

            for (auto const &entry: std::filesystem::directory_iterator{folder->absolute(), ignored}) {
                auto const stem = entry.path().stem().string();

                if (entry.is_regular_file(ignored) && entry.path().extension() == ".lua" && valid_script_name(stem)) {
                    names_.push_back(stem);
                }
            }

            std::ranges::sort(names_);
        }
    }

    return names_;
}

auto BehaviourSystem::reload() -> void {
    scanned_ = false;
    last_error_.clear();

    std::vector<std::string> loaded_names;

    for (auto const &[name, loaded]: classes_) {
        loaded_names.push_back(name);
    }

    for (auto const &name: loaded_names) {
        reload_class(name);
    }
}

auto BehaviourSystem::poll() -> void { poll_changes(); }

auto BehaviourSystem::poll_changes() -> void {
    auto const paths = changes_.drain();

    if (paths.empty()) {
        return;
    }

    // Scripts added, deleted or renamed show up in the lists too.
    scanned_ = false;

    for (auto const &path: paths) {
        auto const name = path.stem().string();

        if (path.extension() != ".lua" || !classes_.contains(name)) {
            continue;
        }

        // Editors save in more than one step; the file may be gone for a moment.
        std::error_code ignored;

        if (!std::filesystem::exists(path, ignored)) {
            continue;
        }

        info("[lua] behaviour '{}' changed on disk, reloading", name);
        reload_class(name);
    }
}

auto BehaviourSystem::script_file(std::string const &name) const -> std::filesystem::path {
    auto const path = Paths::current().data(std::string{behaviours_folder} + "/" + name + ".lua");

    return path ? path->absolute() : std::filesystem::path{};
}

// Loads `name` again. A script that loads replaces the class and running instances switch to it, gaining fields for
// properties that are new and losing those that are gone. One that doesn't load leaves them on the old class and
// shows its problem in the inspector until the file is fixed.
auto BehaviourSystem::reload_class(std::string const &name) -> void {
    auto const found = classes_.find(name);

    if (found == classes_.end()) {
        return;
    }

    auto previous = std::move(found->second);

    classes_.erase(found);

    auto &fresh = load_class(name);
    auto *const state = runtime_->state();

    if (fresh.ref == LUA_NOREF) {
        gui::toast_error("Behaviour failed to load", std::format("{}: {}", name, first_line(fresh.schema.error)));

        // Keep what instances run on.
        fresh.ref = previous.ref;
        fresh.keep_old_ref = true;

        return;
    }

    last_error_.clear();
    gui::toast_success("Behaviour reloaded", name);

    for (auto &[entity, instance]: instances_) {
        if (instance.script != name || instance.ref == LUA_NOREF) {
            continue;
        }

        instance.failed = false;

        auto const top = lua_gettop(state);

        lua_rawgeti(state, LUA_REGISTRYINDEX, instance.ref);

        if (lua_getmetatable(state, -1) != 0) {
            lua_rawgeti(state, LUA_REGISTRYINDEX, fresh.ref);
            lua_setfield(state, -2, "__index");
            lua_pop(state, 1);
        }

        auto const *const component = host_->scene != nullptr && host_->scene->get_registry().valid(entity)
                                              ? host_->scene->get_registry().try_get<Components::Behaviour>(entity)
                                              : nullptr;

        for (auto const &old_desc: previous.schema.properties) {
            if (fresh.schema.find(old_desc.name) == nullptr) {
                lua_pushnil(state);
                lua_setfield(state, top + 1, old_desc.name.c_str());
            }
        }

        for (auto const &desc: fresh.schema.properties) {
            auto const *const old_desc = previous.schema.find(desc.name);

            lua_getfield(state, top + 1, desc.name.c_str());

            // A property that was there with the same type keeps what the script did to it.
            auto const keep = old_desc != nullptr && old_desc->type == desc.type && !lua_isnil(state, -1);

            lua_pop(state, 1);

            if (keep) {
                continue;
            }

            auto const *const stored = component != nullptr ? component->find(desc.name) : nullptr;

            push_script_property(state, *host_, desc,
                                 stored != nullptr ? script_value_sanitised(desc, stored->value) : desc.default_value);
            lua_setfield(state, top + 1, desc.name.c_str());
        }

        lua_settop(state, top);
    }

    luaL_unref(state, LUA_REGISTRYINDEX, previous.ref);
}

auto BehaviourSystem::load_class(std::string const &name) -> Class & {
    if (auto const found = classes_.find(name); found != classes_.end()) {
        return found->second;
    }

    auto &loaded = classes_[name];
    loaded.schema.name = name;


    auto *const state = runtime_->state();
    auto const top = lua_gettop(state);

    auto const fail = [&](std::string message) -> Class & {
        lua_settop(state, top);
        loaded.schema.properties.clear();
        loaded.schema.error = std::move(message);
        error("[lua] behaviour '{}': {}", name, loaded.schema.error);

        return loaded;
    };

    if (!valid_script_name(name)) {
        return fail("not a valid script name");
    }

    auto const source = loader_(name);

    if (!source) {
        return fail("no such file in assets/scripts/behaviours");
    }

    auto const chunk = std::format("@behaviours/{}.lua", name);

    if (luaL_loadbufferx(state, source->data(), source->size(), chunk.c_str(), "t") != LUA_OK) {
        return fail(lua_tostring(state, -1) != nullptr ? lua_tostring(state, -1) : "(unknown error)");
    }

    if (auto const ran = runtime_->protected_call(0, 1, true); !ran.ok) {
        return fail(ran.error);
    }

    if (!lua_istable(state, -1)) {
        return fail("the script must return a table");
    }

    loaded.schema = read_script_schema(state, -1, name);

    if (!loaded.schema.error.empty()) {
        return fail(loaded.schema.error);
    }

    loaded.ref = luaL_ref(state, LUA_REGISTRYINDEX);
    lua_settop(state, top);

    return loaded;
}

auto BehaviourSystem::schema(std::string const &name) -> ScriptSchema const * {
    poll_changes();

    auto const names = script_names();

    if (std::ranges::find(names, name) == names.end()) {
        return nullptr;
    }

    return &load_class(name).schema;
}

auto BehaviourSystem::report(std::string message) -> void {
    error("[lua] {}", message);
    gui::toast_error("Behaviour error", first_line(message));

    if (last_error_.empty()) {
        last_error_ = std::move(message);
    }
}

auto BehaviourSystem::create_instance(entt::entity entity) -> std::optional<std::string> {
    // Scripts added while the editor scene is showing wait for the game to play.
    if (!playing_ || host_->scene == nullptr) {
        return std::nullopt;
    }

    auto &registry = host_->scene->get_registry();
    auto const *const component = registry.try_get<Components::Behaviour>(entity);

    if (component == nullptr) {
        return "the entity has no script";
    }

    if (auto const found = instances_.find(entity); found != instances_.end()) {
        if (found->second.script == component->script) {
            return std::nullopt;
        }

        remove_instance(entity, true);
    }

    auto &instance = instances_[entity];
    instance.script = component->script;

    auto const &loaded = load_class(component->script);

    if (loaded.ref == LUA_NOREF) {
        instance.failed = true;

        return std::format("script '{}': {}", component->script, loaded.schema.error);
    }

    auto *const state = runtime_->state();
    auto const top = lua_gettop(state);

    lua_createtable(state, 0, static_cast<int>(loaded.schema.properties.size()) + 1);

    push_lua_entity(state, entity);
    lua_setfield(state, -2, "entity");

    for (auto const &desc: loaded.schema.properties) {
        auto const *const stored = component->find(desc.name);
        auto const value = stored != nullptr ? script_value_sanitised(desc, stored->value) : desc.default_value;

        push_script_property(state, *host_, desc, value);
        lua_setfield(state, -2, desc.name.c_str());
    }

    lua_createtable(state, 0, 1);
    lua_rawgeti(state, LUA_REGISTRYINDEX, loaded.ref);
    lua_setfield(state, -2, "__index");
    lua_setmetatable(state, -2);

    instance.ref = luaL_ref(state, LUA_REGISTRYINDEX);
    lua_settop(state, top);

    return std::nullopt;
}

auto BehaviourSystem::call(Instance &instance, entt::entity entity, char const *callback,
                              std::optional<float> argument) -> void {
    if (instance.failed || instance.ref == LUA_NOREF) {
        return;
    }

    auto *const state = runtime_->state();
    auto const top = lua_gettop(state);

    lua_rawgeti(state, LUA_REGISTRYINDEX, instance.ref);
    lua_getfield(state, -1, callback);

    if (!lua_isfunction(state, -1)) {
        lua_settop(state, top);

        return;
    }

    lua_pushvalue(state, -2);

    auto const arguments = argument ? 2 : 1;

    if (argument) {
        lua_pushnumber(state, static_cast<double>(*argument));
    }

    if (auto const ran = runtime_->protected_call(arguments, 0); !ran.ok) {
        instance.failed = true;
        report(std::format("script '{}' on entity {}: {}: {}", instance.script,
                           static_cast<std::uint32_t>(entt::to_integral(entity)), callback, ran.error));
    }

    lua_settop(state, top);
}

auto BehaviourSystem::remove_instance(entt::entity entity, bool call_destroy) -> void {
    auto const found = instances_.find(entity);

    if (found == instances_.end()) {
        return;
    }

    auto instance = std::move(found->second);

    instances_.erase(found);

    if (call_destroy && instance.started) {
        call(instance, entity, "on_destroy", std::nullopt);
    }

    luaL_unref(runtime_->state(), LUA_REGISTRYINDEX, instance.ref);
}

auto BehaviourSystem::update(float delta_time) -> void {
    playing_ = true;

    poll_changes();

    if (host_->scene == nullptr) {
        return;
    }

    auto &registry = host_->scene->get_registry();

    // Instances whose entity or script is gone.
    scratch_.clear();

    for (auto const &[entity, instance]: instances_) {
        auto const *const component = registry.valid(entity) ? registry.try_get<Components::Behaviour>(entity) : nullptr;

        if (component == nullptr || component->script != instance.script) {
            scratch_.push_back(entity);
        }
    }

    for (auto const entity: scratch_) {
        remove_instance(entity, true);
    }

    // Callbacks may add entities or scripts, so walk a copy.
    scratch_.clear();

    for (auto const entity: registry.view<Components::Behaviour>()) {
        scratch_.push_back(entity);
    }

    auto const entities = scratch_;

    for (auto const entity: entities) {
        if (!instances_.contains(entity)) {
            if (auto const problem = create_instance(entity)) {
                report(std::format("entity {}: {}", static_cast<std::uint32_t>(entt::to_integral(entity)), *problem));
            }
        }
    }

    for (auto const entity: entities) {
        auto const found = instances_.find(entity);

        if (found == instances_.end() || found->second.started) {
            continue;
        }

        found->second.started = true;
        call(found->second, entity, "on_start", std::nullopt);
    }

    for (auto const entity: entities) {
        auto const found = instances_.find(entity);

        if (found != instances_.end() && registry.valid(entity)) {
            call(found->second, entity, "on_update", delta_time);
        }
    }
}

auto BehaviourSystem::push_instance(entt::entity entity, std::string const &name) -> void {
    auto *const state = runtime_->state();

    if (auto const found = instances_.find(entity);
        found != instances_.end() && found->second.script == name && found->second.ref != LUA_NOREF) {
        lua_rawgeti(state, LUA_REGISTRYINDEX, found->second.ref);

        return;
    }

    lua_pushnil(state);
}

auto BehaviourSystem::stop() -> void {
    scratch_.clear();

    for (auto const &[entity, instance]: instances_) {
        scratch_.push_back(entity);
    }

    auto const entities = scratch_;

    for (auto const entity: entities) {
        remove_instance(entity, true);
    }

    instances_.clear();
    playing_ = false;
}
