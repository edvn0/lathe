#include "app/lua_game_api.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <expected>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>

#include "app/lua_game.hxx"
#include "assets/material_storage.hxx"
#include "assets/model.hxx"
#include "assets/model_streamer.hxx"
#include "assets/texture_streamer.hxx"
#include "core/fly_string.hxx"
#include "app/lua_animation.hxx"
#include "core/paths.hxx"
#include "physics/character_body.hxx"
#include "physics/mesh_collider.hxx"
#include "physics/physics_world.hxx"
#include "rendering/effect_system.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"
#include "scripting/lua_particles.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    constexpr char const *entity_meta = "lathe.Entity";
    constexpr char const *model_meta = "lathe.Model";
    constexpr char const *material_meta = "lathe.Material";
    constexpr char const *effect_shader_meta = "lathe.EffectShader";
    constexpr char const *effect_meta = "lathe.Effect";
    constexpr char const *effect_buffer_meta = "lathe.EffectBuffer";
    constexpr char const *character_meta = "lathe.Character";
    constexpr char const *rig_meta = "lathe.Rig";
    constexpr char const *actor_meta = "lathe.Actor";

    struct LuaEntity {
        entt::entity id;
    };

    struct LuaModel {
        ModelHandle handle;
    };

    struct LuaMaterial {
        MaterialHandle handle;
    };

    static_assert(std::is_trivially_copyable_v<ModelHandle>);
    static_assert(std::is_trivially_copyable_v<MaterialHandle>);

    auto game_host(lua_State *state) noexcept -> LuaGameHost & {
        return *static_cast<LuaGameHost *>(LuaRuntime::host(state));
    }

    auto registry_of(lua_State *state) -> entt::registry & {
        auto *const scene = game_host(state).scene;

        if (scene == nullptr) {
            luaL_error(state, "no scene is active");
        }

        return scene->get_registry();
    }

    auto push_entity(lua_State *state, entt::entity id) -> void {
        auto *const entity = static_cast<LuaEntity *>(lua_newuserdatauv(state, sizeof(LuaEntity), 0));

        entity->id = id;

        luaL_setmetatable(state, entity_meta);
    }

    auto push_model(lua_State *state, ModelHandle handle) -> void {
        auto *const model = static_cast<LuaModel *>(lua_newuserdatauv(state, sizeof(LuaModel), 0));

        model->handle = handle;

        luaL_setmetatable(state, model_meta);
    }

    auto entity_arg(lua_State *state, int index) -> entt::entity {
        auto const *const entity = static_cast<LuaEntity const *>(luaL_checkudata(state, index, entity_meta));
        auto &registry = registry_of(state);

        if (!registry.valid(entity->id)) {
            luaL_error(state, "the entity no longer exists");
        }

        return entity->id;
    }

    // Fields of an optional options table at `index`; a missing table or field gives the fallback.
    auto option_number(lua_State *state, int index, char const *key, double fallback) -> double {
        return lua_istable(state, index) ? lua_field_number(state, index, key, fallback) : fallback;
    }

    auto option_bool(lua_State *state, int index, char const *key, bool fallback) -> bool {
        if (!lua_istable(state, index)) {
            return fallback;
        }

        lua_getfield(state, index, key);
        auto const value = lua_isnil(state, -1) ? fallback : lua_toboolean(state, -1) != 0;
        lua_pop(state, 1);

        return value;
    }

    auto option_string(lua_State *state, int index, char const *key, char const *fallback) -> std::string {
        if (!lua_istable(state, index)) {
            return fallback;
        }

        lua_getfield(state, index, key);
        std::string value = lua_isstring(state, -1) != 0 ? lua_tostring(state, -1) : fallback;
        lua_pop(state, 1);

        return value;
    }

    auto model_key(ModelHandle handle) -> std::uint64_t {
        return (static_cast<std::uint64_t>(handle.index) << 32U) | handle.generation;
    }

    auto scene_spawn(lua_State *state) -> int {
        auto *const scene = game_host(state).scene;
        auto const *const name = luaL_checkstring(state, 1);

        if (scene == nullptr) {
            return luaL_error(state, "no scene is active");
        }

        auto const entity = Entity{scene, std::string_view{name}};

        entity.emplace<Components::Transform>(Components::Transform{});

        push_entity(state, static_cast<entt::entity>(entity));

        return 1;
    }

    auto scene_find(lua_State *state) -> int {
        auto const wanted = std::string_view{luaL_checkstring(state, 1)};
        auto &registry = registry_of(state);

        for (auto const entity: registry.view<Components::Meta>()) {
            if (registry.get<Components::Meta>(entity).name.view() == wanted) {
                push_entity(state, entity);

                return 1;
            }
        }

        for (auto const entity: registry.view<Components::GeneratedMeta>()) {
            if (registry.get<Components::GeneratedMeta>(entity).name == wanted) {
                push_entity(state, entity);

                return 1;
            }
        }

        lua_pushnil(state);

        return 1;
    }

    auto scene_clear(lua_State *state) -> int {
        registry_of(state).clear();

        return 0;
    }

    auto entity_valid(lua_State *state) -> int {
        auto const *const entity = static_cast<LuaEntity const *>(luaL_checkudata(state, 1, entity_meta));
        auto *const scene = game_host(state).scene;

        lua_pushboolean(state, scene != nullptr && scene->get_registry().valid(entity->id) ? 1 : 0);

        return 1;
    }

    auto entity_destroy(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);

        registry_of(state).destroy(id);

        return 0;
    }

    auto entity_name(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &registry = registry_of(state);

        if (auto const *meta = registry.try_get<Components::Meta>(id)) {
            lua_pushstring(state, std::string{meta->name.view()}.c_str());
        } else if (auto const *generated = registry.try_get<Components::GeneratedMeta>(id)) {
            lua_pushstring(state, generated->name.c_str());
        } else {
            lua_pushstring(state, "");
        }

        return 1;
    }

    auto entity_set_position(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const x = static_cast<float>(luaL_checknumber(state, 2));
        auto const y = static_cast<float>(luaL_checknumber(state, 3));
        auto const z = static_cast<float>(luaL_checknumber(state, 4));

        registry_of(state).patch<Components::Transform>(id, [&](Components::Transform &t) { t.position = {x, y, z}; });

        return 0;
    }

    auto entity_position(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const &position = registry_of(state).get_or_emplace<Components::Transform>(id).position;

        lua_pushnumber(state, position.x);
        lua_pushnumber(state, position.y);
        lua_pushnumber(state, position.z);

        return 3;
    }

    auto entity_set_euler(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const x = static_cast<float>(luaL_checknumber(state, 2));
        auto const y = static_cast<float>(luaL_checknumber(state, 3));
        auto const z = static_cast<float>(luaL_checknumber(state, 4));

        registry_of(state).patch<Components::Transform>(
                id, [&](Components::Transform &t) { t.rotation = glm::quat{glm::vec3{x, y, z}}; });

        return 0;
    }

    auto entity_set_scale(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const x = static_cast<float>(luaL_checknumber(state, 2));
        auto const y = lua_isnone(state, 3) ? x : static_cast<float>(luaL_checknumber(state, 3));
        auto const z = lua_isnone(state, 4) ? x : static_cast<float>(luaL_checknumber(state, 4));

        registry_of(state).patch<Components::Transform>(id, [&](Components::Transform &t) { t.scale = {x, y, z}; });

        return 0;
    }

    auto entity_set_model(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const *const model = static_cast<LuaModel const *>(luaL_checkudata(state, 2, model_meta));

        registry_of(state).emplace_or_replace<Components::Model>(id, Components::Model{.model = model->handle});

        return 0;
    }

    auto entity_set_material(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const *const material = static_cast<LuaMaterial const *>(luaL_checkudata(state, 2, material_meta));

        registry_of(state).emplace_or_replace<Components::MaterialOverride>(
                id, Components::MaterialOverride{.material = material->handle});

        return 0;
    }

    auto entity_set_outlined(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto const outlined = lua_toboolean(state, 2) != 0;
        auto &registry = registry_of(state);

        if (outlined) {
            registry.emplace_or_replace<Components::Outlined>(id);
        } else {
            registry.remove<Components::Outlined>(id);
        }

        return 0;
    }

    // Applies the particle table at stack index 2 to `emitter`. On a problem the message is left on the stack and the
    // result is false, so the caller raises it after every C++ object here is gone.
    auto apply_particles(lua_State *state, Components::ParticleEmitter &emitter) -> bool {
        luaL_checktype(state, 2, LUA_TTABLE);

        if (auto const problem = read_particle_table(state, 2, emitter)) {
            lua_pushlstring(state, problem->data(), problem->size());

            return false;
        }

        lua_getfield(state, 2, "material");

        if (!lua_isnil(state, -1)) {
            auto const *const material = static_cast<LuaMaterial const *>(luaL_testudata(state, -1, material_meta));

            if (material == nullptr) {
                lua_pushstring(state, "'material' must be a material from assets.material");

                return false;
            }

            emitter.material = material->handle;
        }

        lua_pop(state, 1);

        return true;
    }

    // entity:add_particles{count = 2000, lifetime = 1.5, gravity = {0, -4, 0}, material = m, ...}
    auto entity_add_particles(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &registry = registry_of(state);

        if (registry.all_of<Components::ParticleEmitter>(id)) {
            return luaL_error(state, "the entity already has particles; use set_particles to change them");
        }

        auto emitter = Components::ParticleEmitter{};

        if (!apply_particles(state, emitter)) {
            return lua_error(state);
        }

        registry.emplace<Components::ParticleEmitter>(id, emitter);

        return 0;
    }

    // entity:set_particles{rate = 0}: changes the fields that are given and keeps the rest.
    auto entity_set_particles(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &registry = registry_of(state);
        auto *const current = registry.try_get<Components::ParticleEmitter>(id);

        if (current == nullptr) {
            return luaL_error(state, "the entity has no particles; use add_particles first");
        }

        auto emitter = *current;

        if (!apply_particles(state, emitter)) {
            return lua_error(state);
        }

        *current = emitter;

        return 0;
    }

    auto entity_remove_particles(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);

        registry_of(state).remove<Components::ParticleEmitter>(id);

        return 0;
    }


    // ---- physics ----------------------------------------------------------------------------------------------

    struct LuaCharacter {
        CharacterBody body;
    };

    static_assert(std::is_trivially_destructible_v<CharacterBody>);

    auto physics_world_of(lua_State *state) -> PhysicsWorld & {
        auto *const scene = game_host(state).scene;

        if (scene == nullptr || scene->physics_world == nullptr) {
            luaL_error(state, "physics is only available while the game is playing");
        }

        return *scene->physics_world;
    }

    auto character_arg(lua_State *state, int index) -> CharacterBody & {
        return static_cast<LuaCharacter *>(luaL_checkudata(state, index, character_meta))->body;
    }

    auto physics_character(lua_State *state) -> int {
        auto const x = static_cast<float>(luaL_checknumber(state, 1));
        auto const y = static_cast<float>(luaL_checknumber(state, 2));
        auto const z = static_cast<float>(luaL_checknumber(state, 3));
        auto const radius = static_cast<float>(option_number(state, 4, "radius", 0.35));
        auto const height = static_cast<float>(option_number(state, 4, "height", 1.8));

        if (!(radius > 0.0F) || !(height > 2.0F * radius)) {
            return luaL_error(state, "a character needs radius > 0 and height > 2 * radius");
        }

        MovementParams params;
        params.gravity = static_cast<float>(option_number(state, 4, "gravity", params.gravity));
        params.jump_height = static_cast<float>(option_number(state, 4, "jump_height", params.jump_height));
        params.step_height = static_cast<float>(option_number(state, 4, "step_height", params.step_height));
        params.max_slope_degrees = static_cast<float>(option_number(state, 4, "max_slope", params.max_slope_degrees));
        params.ground_accel = static_cast<float>(option_number(state, 4, "ground_accel", params.ground_accel));
        params.ground_decel = static_cast<float>(option_number(state, 4, "ground_decel", params.ground_decel));
        params.air_accel = static_cast<float>(option_number(state, 4, "air_accel", params.air_accel));

        auto *const character = static_cast<LuaCharacter *>(lua_newuserdatauv(state, sizeof(LuaCharacter), 0));
        new (character) LuaCharacter{CharacterBody{{x, y, z}, radius, height, params}};
        luaL_setmetatable(state, character_meta);

        return 1;
    }

    // update(dt, move_x, move_z, jump_pressed, jump_held): moves at the given world-space velocity (metres per
    // second), with gravity, slopes, steps and jumping, in fixed substeps.
    auto character_update(lua_State *state) -> int {
        auto &body = character_arg(state, 1);
        auto const delta = std::clamp(static_cast<float>(luaL_checknumber(state, 2)), 0.0F, 0.1F);
        auto const &world = physics_world_of(state);

        CharacterInput input;
        input.desired_velocity = {static_cast<float>(luaL_checknumber(state, 3)), 0.0F,
                                  static_cast<float>(luaL_checknumber(state, 4))};
        input.jump_pressed = lua_toboolean(state, 5) != 0;
        input.jump_held = lua_toboolean(state, 6) != 0;

        constexpr float substep = 1.0F / 60.0F;
        auto const steps = std::max(1, static_cast<int>(std::ceil(delta / substep)));

        for (int step = 0; step < steps; ++step) {
            body.step(world, input, delta / static_cast<float>(steps));
            input.jump_pressed = false;
        }

        return 0;
    }

    auto character_position(lua_State *state) -> int {
        auto const &position = character_arg(state, 1).position();

        lua_pushnumber(state, position.x);
        lua_pushnumber(state, position.y);
        lua_pushnumber(state, position.z);

        return 3;
    }

    auto character_velocity(lua_State *state) -> int {
        auto const &velocity = character_arg(state, 1).velocity();

        lua_pushnumber(state, velocity.x);
        lua_pushnumber(state, velocity.y);
        lua_pushnumber(state, velocity.z);

        return 3;
    }

    auto character_grounded(lua_State *state) -> int {
        lua_pushboolean(state, character_arg(state, 1).grounded() ? 1 : 0);

        return 1;
    }

    auto character_teleport(lua_State *state) -> int {
        character_arg(state, 1).teleport({static_cast<float>(luaL_checknumber(state, 2)),
                                          static_cast<float>(luaL_checknumber(state, 3)),
                                          static_cast<float>(luaL_checknumber(state, 4))});

        return 0;
    }

    constexpr std::array character_methods{
            luaL_Reg{"update", &character_update},     luaL_Reg{"position", &character_position},
            luaL_Reg{"velocity", &character_velocity}, luaL_Reg{"grounded", &character_grounded},
            luaL_Reg{"teleport", &character_teleport}, luaL_Reg{nullptr, nullptr},
    };

    // raycast(ox, oy, oz, dx, dy, dz, max_distance) -> distance, px, py, pz, nx, ny, nz  (nil when nothing is hit)
    auto physics_raycast(lua_State *state) -> int {
        auto const &world = physics_world_of(state);
        glm::vec3 const origin{static_cast<float>(luaL_checknumber(state, 1)),
                               static_cast<float>(luaL_checknumber(state, 2)),
                               static_cast<float>(luaL_checknumber(state, 3))};
        glm::vec3 const direction{static_cast<float>(luaL_checknumber(state, 4)),
                                  static_cast<float>(luaL_checknumber(state, 5)),
                                  static_cast<float>(luaL_checknumber(state, 6))};
        auto const max_distance = static_cast<float>(luaL_optnumber(state, 7, 1000.0));

        auto const hit = world.raycast(origin, direction, max_distance);

        if (!hit) {
            lua_pushnil(state);
            return 1;
        }

        for (auto const value: {hit->distance, hit->point.x, hit->point.y, hit->point.z, hit->normal.x, hit->normal.y,
                                hit->normal.z}) {
            lua_pushnumber(state, value);
        }

        return 7;
    }

    // add_mesh_collider(entity, model): the entity's position places the model's triangles as static geometry. The
    // model must have been loaded with { collider = true }; collision starts once its triangles are built.
    auto physics_add_mesh_collider(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const id = entity_arg(state, 1);
        auto const *const model = static_cast<LuaModel const *>(luaL_checkudata(state, 2, model_meta));
        auto const slot = host.collider_slots.find(model_key(model->handle));

        if (slot == host.collider_slots.end()) {
            return luaL_error(state, "the model was not loaded with { collider = true }");
        }

        host.mesh_colliders.push_back({.entity = id, .slot = slot->second});

        return 0;
    }

    // ready(model) -> true once the model's mesh collider is built.
    auto physics_ready(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const *const model = static_cast<LuaModel const *>(luaL_checkudata(state, 1, model_meta));
        auto const slot = host.collider_slots.find(model_key(model->handle));

        lua_pushboolean(state, slot != host.collider_slots.end() && slot->second->ready.load(std::memory_order_acquire));

        return 1;
    }

    constexpr std::array physics_functions{
            luaL_Reg{"character", &physics_character},
            luaL_Reg{"raycast", &physics_raycast},
            luaL_Reg{"add_mesh_collider", &physics_add_mesh_collider},
            luaL_Reg{"ready", &physics_ready},
            luaL_Reg{nullptr, nullptr},
    };


    // ---- animation --------------------------------------------------------------------------------------------

    struct LuaRig {
        std::uint32_t rig;
    };

    struct LuaActor {
        std::uint32_t rig;
        std::uint32_t index;
    };

    // animation.rig(model, options) -> Rig. The model is a skinned one loaded with assets.load_model.
    auto animation_rig(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const *const model = static_cast<LuaModel const *>(luaL_checkudata(state, 1, model_meta));

        if (host.scene == nullptr || host.animation == nullptr) {
            return luaL_error(state, "animation can only be used from on_populate and later");
        }

        LuaAnimationSystem::RigOptions options;
        options.height = static_cast<float>(option_number(state, 2, "height", options.height));
        options.model_height = static_cast<float>(option_number(state, 2, "model_height", options.model_height));
        options.model_feet_y = static_cast<float>(option_number(state, 2, "model_feet_y", options.model_feet_y));
        options.yaw_offset = static_cast<float>(option_number(state, 2, "yaw_offset", options.yaw_offset));
        options.walk_speed = static_cast<float>(option_number(state, 2, "walk_speed", options.walk_speed));
        options.run_speed = static_cast<float>(option_number(state, 2, "run_speed", options.run_speed));

        if (!(options.height > 0.0F) || !(options.model_height > 0.0F) || !(options.walk_speed > 0.0F) ||
            !(options.run_speed > 0.0F)) {
            return luaL_error(state, "height, model_height, walk_speed and run_speed must be positive");
        }

        auto *const rig = static_cast<LuaRig *>(lua_newuserdatauv(state, sizeof(LuaRig), 0));
        rig->rig = host.animation->create_rig(*host.scene, model->handle, options);
        luaL_setmetatable(state, rig_meta);

        return 1;
    }

    auto rig_arg(lua_State *state, int index) -> std::uint32_t {
        return static_cast<LuaRig *>(luaL_checkudata(state, index, rig_meta))->rig;
    }

    auto rig_ready(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const rig = rig_arg(state, 1);

        lua_pushboolean(state, host.renderer != nullptr && host.animation->ready(*host.renderer, rig) ? 1 : 0);

        return 1;
    }

    auto rig_spawn(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const rig = rig_arg(state, 1);

        auto *const actor = static_cast<LuaActor *>(lua_newuserdatauv(state, sizeof(LuaActor), 0));
        actor->rig = rig;
        actor->index = host.animation->add_actor(rig);
        luaL_setmetatable(state, actor_meta);

        return 1;
    }

    constexpr std::array rig_methods{
            luaL_Reg{"ready", &rig_ready},
            luaL_Reg{"spawn", &rig_spawn},
            luaL_Reg{nullptr, nullptr},
    };

    auto actor_arg(lua_State *state, int index) -> LuaAnimationSystem::Actor & {
        auto const *const handle = static_cast<LuaActor const *>(luaL_checkudata(state, index, actor_meta));
        auto *const actor = game_host(state).animation->actor(handle->rig, handle->index);

        if (actor == nullptr) {
            luaL_error(state, "the actor no longer exists");
        }

        return *actor;
    }

    // place(x, y, z, yaw): where its feet are and which way it faces (radians about +Y; 0 faces +Z).
    auto actor_place(lua_State *state) -> int {
        auto &actor = actor_arg(state, 1);

        actor.position = {static_cast<float>(luaL_checknumber(state, 2)), static_cast<float>(luaL_checknumber(state, 3)),
                          static_cast<float>(luaL_checknumber(state, 4))};
        actor.yaw = static_cast<float>(luaL_checknumber(state, 5));

        return 0;
    }

    // move(speed, grounded, vertical_velocity): how it is moving, which picks idle, walk, run or a jump pose.
    auto actor_move(lua_State *state) -> int {
        auto &actor = actor_arg(state, 1);

        actor.inputs.horizontal_speed = std::max(static_cast<float>(luaL_checknumber(state, 2)), 0.0F);
        actor.inputs.grounded = lua_isnone(state, 3) || lua_toboolean(state, 3) != 0;
        actor.inputs.vertical_velocity = static_cast<float>(luaL_optnumber(state, 4, 0.0));

        return 0;
    }

    auto actor_set_visible(lua_State *state) -> int {
        actor_arg(state, 1).active = lua_toboolean(state, 2) != 0;

        return 0;
    }

    auto actor_state(lua_State *state) -> int {
        auto const *const handle = static_cast<LuaActor const *>(luaL_checkudata(state, 1, actor_meta));
        auto const name = game_host(state).animation->state_name(handle->rig, handle->index);

        lua_pushlstring(state, name.data(), name.size());

        return 1;
    }

    constexpr std::array actor_methods{
            luaL_Reg{"place", &actor_place},
            luaL_Reg{"move", &actor_move},
            luaL_Reg{"set_visible", &actor_set_visible},
            luaL_Reg{"state", &actor_state},
            luaL_Reg{nullptr, nullptr},
    };

    constexpr std::array animation_functions{
            luaL_Reg{"rig", &animation_rig},
            luaL_Reg{nullptr, nullptr},
    };

    // body options: shape = "box" | "sphere" | "capsule", half_extents = {x, y, z}, radius, height, mass, restitution,
    // static (boolean), lock_rotation (boolean)
    auto entity_add_body(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &registry = registry_of(state);

        Components::RigidBody body;
        auto const shape = option_string(state, 2, "shape", "box");

        if (shape == "box") {
            body.shape = Components::BodyShape::box;
        } else if (shape == "sphere") {
            body.shape = Components::BodyShape::sphere;
        } else if (shape == "capsule") {
            body.shape = Components::BodyShape::capsule;
            body.lock_rotation = true;
        } else {
            return luaL_error(state, "unknown body shape '%s' (box, sphere or capsule)", shape.c_str());
        }

        if (lua_istable(state, 2)) {
            auto const half = lua_field_vec3(state, 2, "half_extents", {0.5F, 0.5F, 0.5F});
            body.half_extents = {half.x, half.y, half.z};
        }

        body.sphere_radius = static_cast<float>(option_number(state, 2, "radius", body.sphere_radius));
        body.capsule_radius = static_cast<float>(option_number(state, 2, "radius", body.capsule_radius));
        body.capsule_height = static_cast<float>(option_number(state, 2, "height", body.capsule_height));
        body.mass = static_cast<float>(option_number(state, 2, "mass", body.mass));
        body.restitution = static_cast<float>(option_number(state, 2, "restitution", body.restitution));
        body.is_static = option_bool(state, 2, "static", false);
        body.lock_rotation = option_bool(state, 2, "lock_rotation", body.lock_rotation);

        if (!(body.mass > 0.0F) || !(body.sphere_radius > 0.0F) || !(body.capsule_radius > 0.0F) ||
            !(body.capsule_height > 0.0F)) {
            return luaL_error(state, "mass, radius and height must be positive");
        }

        registry.emplace_or_replace<Components::RigidBody>(id, body);

        // The scene may already be simulating (a body made during play); otherwise it is built when play starts.
        if (auto *const scene = game_host(state).scene; scene != nullptr && scene->physics_world != nullptr) {
            if (registry.all_of<Components::PhysicsBody>(id)) {
                scene->physics_world->remove_body(registry, id);
            }

            scene->physics_world->add_body(registry, id, registry.get_or_emplace<Components::Transform>(id), body);
        }

        return 0;
    }

    auto entity_set_velocity(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &world = physics_world_of(state);

        world.set_velocity(registry_of(state), id,
                           {static_cast<float>(luaL_checknumber(state, 2)),
                            static_cast<float>(luaL_checknumber(state, 3)),
                            static_cast<float>(luaL_checknumber(state, 4))});

        return 0;
    }

    auto entity_apply_impulse(lua_State *state) -> int {
        auto const id = entity_arg(state, 1);
        auto &world = physics_world_of(state);

        world.apply_impulse(registry_of(state), id,
                            {static_cast<float>(luaL_checknumber(state, 2)),
                             static_cast<float>(luaL_checknumber(state, 3)),
                             static_cast<float>(luaL_checknumber(state, 4))});

        return 0;
    }

    auto key_down(lua_State *state) -> int {
        lua_pushboolean(state, game_host(state).keys_down.contains(static_cast<int>(luaL_checkinteger(state, 1))) ? 1 : 0);

        return 1;
    }

    // delta() -> dx, dy: mouse movement in pixels since the previous update.
    auto mouse_delta(lua_State *state) -> int {
        auto const &host = game_host(state);

        lua_pushnumber(state, host.mouse_delta_x);
        lua_pushnumber(state, host.mouse_delta_y);

        return 2;
    }

    constexpr std::array entity_methods{
            luaL_Reg{"valid", &entity_valid},
            luaL_Reg{"destroy", &entity_destroy},
            luaL_Reg{"name", &entity_name},
            luaL_Reg{"set_position", &entity_set_position},
            luaL_Reg{"position", &entity_position},
            luaL_Reg{"set_euler", &entity_set_euler},
            luaL_Reg{"set_scale", &entity_set_scale},
            luaL_Reg{"set_model", &entity_set_model},
            luaL_Reg{"set_material", &entity_set_material},
            luaL_Reg{"set_outlined", &entity_set_outlined},
            luaL_Reg{"add_particles", &entity_add_particles},
            luaL_Reg{"set_particles", &entity_set_particles},
            luaL_Reg{"remove_particles", &entity_remove_particles},
            luaL_Reg{"add_body", &entity_add_body},
            luaL_Reg{"set_velocity", &entity_set_velocity},
            luaL_Reg{"apply_impulse", &entity_apply_impulse},
            luaL_Reg{nullptr, nullptr},
    };

    auto assets_load_model(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const *const text = luaL_checkstring(state, 1);

        if (host.renderer == nullptr || host.engine_models == nullptr) {
            return luaL_error(state, "assets can only be loaded from on_populate and later");
        }

        auto const path = AssetPath::from_serialised(text);

        if (!path) {
            return luaL_error(state, "'%s' is not a valid asset path", text);
        }

        auto const wants_collider = option_bool(state, 2, "collider", false);
        auto const min_extent = static_cast<float>(option_number(state, 2, "min_extent", 0.0));

        auto observer = wants_collider ? std::optional{make_collider_observer(min_extent)} : std::nullopt;

        auto const handle = host.renderer->model_streamer().request(
                *host.renderer, *path, host.engine_models->cube, FlyString{std::string_view{text}},
                observer ? std::move(observer->observer) : std::function<void(ModelCpuData const &)>{});

        if (observer) {
            host.collider_slots[model_key(handle)] = observer->slot;
        }

        push_model(state, handle);

        return 1;
    }

    auto assets_cube(lua_State *state) -> int {
        auto &host = game_host(state);

        if (host.engine_models == nullptr) {
            return luaL_error(state, "assets can only be used from on_populate and later");
        }

        push_model(state, host.engine_models->cube);

        return 1;
    }

    auto assets_material(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const *const name = luaL_checkstring(state, 1);
        auto const red = static_cast<float>(luaL_checknumber(state, 2));
        auto const green = static_cast<float>(luaL_checknumber(state, 3));
        auto const blue = static_cast<float>(luaL_checknumber(state, 4));
        auto const roughness = static_cast<float>(luaL_optnumber(state, 5, 0.6));
        auto const metallic = static_cast<float>(luaL_optnumber(state, 6, 0.0));

        if (host.renderer == nullptr) {
            return luaL_error(state, "materials can only be created from on_populate and later");
        }

        auto &images = host.renderer->image_storage();
        auto created = host.renderer->create_material(
                MaterialCreateInfo{
                        .base_colour_factor = glm::vec4{red, green, blue, 1.0F},
                        .metallic_factor = metallic,
                        .roughness_factor = roughness,
                        .base_colour_texture = images.white(),
                        .normal_texture = images.flat_normal(),
                        .metallic_roughness_texture = images.metallic_roughness(),
                        .occlusion_texture = images.occlusion(),
                        .emissive_texture = images.emissive(),
                        .sampler = host.renderer->sampler_storage().linear_repeat(),
                },
                name);

        if (!created) {
            return luaL_error(state, "could not create material '%s'", name);
        }

        auto *const material = static_cast<LuaMaterial *>(lua_newuserdatauv(state, sizeof(LuaMaterial), 0));

        material->handle = *created;

        luaL_setmetatable(state, material_meta);

        return 1;
    }

    auto assets_release_material(lua_State *state) -> int {
        auto &host = game_host(state);
        auto const *const material = static_cast<LuaMaterial const *>(luaL_checkudata(state, 1, material_meta));

        if (host.renderer != nullptr) {
            host.renderer->release_material(material->handle);
        }

        return 0;
    }

    auto assets_pending(lua_State *state) -> int {
        auto &host = game_host(state);

        if (host.renderer == nullptr) {
            lua_pushinteger(state, 0);

            return 1;
        }

        lua_pushinteger(state, static_cast<lua_Integer>(host.renderer->model_streamer().pending_count() +
                                                        host.renderer->texture_streamer().pending_count()));

        return 1;
    }

    constexpr std::array assets_functions{
            luaL_Reg{"load_model", &assets_load_model},
            luaL_Reg{"cube", &assets_cube},
            luaL_Reg{"material", &assets_material},
            luaL_Reg{"release_material", &assets_release_material},
            luaL_Reg{"pending", &assets_pending},
            luaL_Reg{nullptr, nullptr},
    };

    auto camera_pick_plane(lua_State *state) -> int {
        auto const ndc_x = static_cast<float>(luaL_checknumber(state, 1));
        auto const ndc_y = static_cast<float>(luaL_checknumber(state, 2));
        auto const plane_y = static_cast<float>(luaL_checknumber(state, 3));
        auto const &inverse = game_host(state).inverse_view_projection;

        auto const unproject = [&](float depth) {
            auto const point = inverse * glm::vec4{ndc_x, ndc_y, depth, 1.0F};

            return glm::vec3{point} / point.w;
        };

        auto const near_point = unproject(0.0F);
        auto const direction = unproject(1.0F) - near_point;

        if (std::abs(direction.y) < 1e-6F) {
            lua_pushnil(state);

            return 1;
        }

        auto const distance = (plane_y - near_point.y) / direction.y;

        if (distance < 0.0F) {
            lua_pushnil(state);

            return 1;
        }

        auto const hit = near_point + direction * distance;

        lua_pushnumber(state, hit.x);
        lua_pushnumber(state, hit.z);

        return 2;
    }

    constexpr std::array camera_functions{
            luaL_Reg{"pick_plane", &camera_pick_plane},
            luaL_Reg{nullptr, nullptr},
    };

    auto game_quit(lua_State *state) -> int {
        if (auto const &request_exit = game_host(state).host.request_exit) {
            request_exit();
        }

        return 0;
    }

    auto game_time(lua_State *state) -> int {
        lua_pushnumber(state, ImGui::GetTime());

        return 1;
    }

    struct LuaEffectShader {
        EffectShaderId id;
    };

    struct LuaEffect {
        EffectId id;
    };

    struct LuaEffectBuffer {
        EffectBufferId id;
    };

    // Runs `body`, which returns the number of results, or nothing after leaving an error message on the stack. The
    // message is raised here, once everything `body` owned is gone, so no C++ object is skipped by the longjmp.
    template<typename Body>
    auto guarded(lua_State *state, Body &&body) -> int {
        auto const results = body();

        if (!results) {
            return lua_error(state);
        }

        return *results;
    }

    auto push_message(lua_State *state, std::string const &message) -> std::optional<int> {
        lua_pushlstring(state, message.data(), message.size());

        return std::nullopt;
    }

    auto effects_of(lua_State *state) -> EffectSystem * { return game_host(state).host.effects; }

    template<typename T>
    auto push_userdata(lua_State *state, char const *meta, T value) -> void {
        auto *const data = static_cast<T *>(lua_newuserdatauv(state, sizeof(T), 0));

        *data = value;

        luaL_setmetatable(state, meta);
    }

    // The value of a script's field for an effect: a number, a list of numbers, a name, or a buffer.
    auto effect_value_at(lua_State *state, int index) -> std::expected<EffectValue, std::string> {
        auto value = EffectValue{};

        index = lua_absindex(state, index);

        switch (lua_type(state, index)) {
            case LUA_TNUMBER:
                value.numbers.push_back(lua_tonumber(state, index));
                break;
            case LUA_TSTRING:
                value.text = lua_tostring(state, index);
                break;
            case LUA_TUSERDATA: {
                auto const *const buffer = static_cast<LuaEffectBuffer const *>(luaL_testudata(state, index, effect_buffer_meta));

                if (buffer == nullptr) {
                    return std::unexpected("this kind of value cannot be given to an effect");
                }

                value.buffer = buffer->id;
                break;
            }
            case LUA_TTABLE: {
                auto const length = lua_rawlen(state, index);

                if (length == 0 || length > 4) {
                    return std::unexpected("a list of numbers has one to four numbers");
                }

                for (auto position = std::size_t{1}; position <= length; ++position) {
                    lua_rawgeti(state, index, static_cast<lua_Integer>(position));

                    if (lua_type(state, -1) != LUA_TNUMBER) {
                        lua_pop(state, 1);

                        return std::unexpected("a list of numbers can only hold numbers");
                    }

                    value.numbers.push_back(lua_tonumber(state, -1));
                    lua_pop(state, 1);
                }
                break;
            }
            default:
                return std::unexpected("only numbers, lists of numbers, names and buffers can be given to an effect");
        }

        return value;
    }

    // compute.load(path) -> shader: reads the manifest at `path` (under assets/) and registers its shader.
    auto compute_load(lua_State *state) -> int {
        auto const *const path = luaL_checkstring(state, 1);

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            auto const loaded = effects->load(path);

            if (!loaded) {
                return push_message(state, loaded.error());
            }

            push_userdata(state, effect_shader_meta, LuaEffectShader{.id = *loaded});

            return 1;
        });
    }

    // compute.reload(shader): reads the manifest again, keeping the old one if the new one is wrong.
    auto compute_reload(lua_State *state) -> int {
        auto const *const shader = static_cast<LuaEffectShader const *>(luaL_checkudata(state, 1, effect_shader_meta));

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            if (auto const reloaded = effects->reload(shader->id); !reloaded) {
                return push_message(state, reloaded.error());
            }

            return 0;
        });
    }

    // Sets `name` on the effect from the Lua value at `index`.
    auto set_effect_field(lua_State *state, EffectSystem &effects, EffectId effect, std::string_view name, int index)
            -> std::optional<int> {
        auto const value = effect_value_at(state, index);

        if (!value) {
            return push_message(state, std::string{name} + ": " + value.error());
        }

        if (auto const set = effects.set(effect, name, *value); !set) {
            return push_message(state, set.error());
        }

        return 0;
    }

    // compute.instance(shader [, {name = value, ...}]) -> effect
    auto compute_instance(lua_State *state) -> int {
        auto const *const shader = static_cast<LuaEffectShader const *>(luaL_checkudata(state, 1, effect_shader_meta));
        auto const has_values = !lua_isnoneornil(state, 2);

        if (has_values) {
            luaL_checktype(state, 2, LUA_TTABLE);
        }

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            auto const created = effects->instance(shader->id);

            if (!created) {
                return push_message(state, created.error());
            }

            if (has_values) {
                lua_pushnil(state);

                while (lua_next(state, 2) != 0) {
                    if (lua_type(state, -2) != LUA_TSTRING) {
                        effects->destroy(*created);

                        return push_message(state, "effect fields are named by strings");
                    }

                    if (auto const failed = set_effect_field(state, *effects, *created, lua_tostring(state, -2), -1);
                        failed != 0) {
                        // The message is on top; the key and value below it go with the stack.
                        effects->destroy(*created);

                        return failed;
                    }

                    lua_pop(state, 1);
                }
            }

            push_userdata(state, effect_meta, LuaEffect{.id = *created});

            return 1;
        });
    }

    auto effect_arg(lua_State *state, int index) -> EffectId {
        return static_cast<LuaEffect const *>(luaL_checkudata(state, index, effect_meta))->id;
    }

    // effect:set(name, value)
    auto effect_set(lua_State *state) -> int {
        auto const id = effect_arg(state, 1);
        auto const *const name = luaL_checkstring(state, 2);

        luaL_checkany(state, 3);

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            return set_effect_field(state, *effects, id, name, 3);
        });
    }

    // effect:problem() -> the reason the engine last dropped this effect from a frame, or nil.
    auto effect_problem(lua_State *state) -> int {
        auto const id = effect_arg(state, 1);
        auto const *const effects = effects_of(state);
        auto const problem = effects != nullptr ? effects->problem(id) : std::string{};

        if (problem.empty()) {
            lua_pushnil(state);
        } else {
            lua_pushlstring(state, problem.data(), problem.size());
        }

        return 1;
    }

    auto effect_gc(lua_State *state) -> int {
        if (auto *const effects = effects_of(state)) {
            effects->destroy(static_cast<LuaEffect const *>(lua_touserdata(state, 1))->id);
        }

        return 0;
    }

    // compute.buffer(count) -> an opaque run of `count` floats that only effects can use.
    auto compute_buffer(lua_State *state) -> int {
        auto const count = luaL_checkinteger(state, 1);

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            if (count < 1 || count > max_effect_buffer_elements) {
                return push_message(state, "a buffer holds 1 to " + std::to_string(max_effect_buffer_elements) + " floats");
            }

            auto const created = effects->create_buffer(static_cast<std::uint32_t>(count));

            if (!created) {
                return push_message(state, created.error());
            }

            push_userdata(state, effect_buffer_meta, LuaEffectBuffer{.id = *created});

            return 1;
        });
    }

    auto effect_buffer_gc(lua_State *state) -> int {
        if (auto *const effects = effects_of(state)) {
            effects->release_buffer(static_cast<LuaEffectBuffer const *>(lua_touserdata(state, 1))->id);
        }

        return 0;
    }

    // scene.add_effect(effect, slot): slot is "frame_start", "after_depth", "after_lighting" or "before_composite".
    auto scene_add_effect(lua_State *state) -> int {
        auto const id = effect_arg(state, 1);
        auto const *const slot_name = luaL_checkstring(state, 2);

        return guarded(state, [&]() -> std::optional<int> {
            auto *const effects = effects_of(state);

            if (effects == nullptr) {
                return push_message(state, "effects are not available");
            }

            auto const slot = parse_game_slot(slot_name);

            if (!slot) {
                return push_message(state, std::string{"unknown slot '"} + slot_name +
                                                   "'; use frame_start, after_depth, after_lighting or before_composite");
            }

            if (auto const added = effects->add(id, *slot); !added) {
                return push_message(state, added.error());
            }

            return 0;
        });
    }

    auto scene_remove_effect(lua_State *state) -> int {
        auto const id = effect_arg(state, 1);

        if (auto *const effects = effects_of(state)) {
            effects->remove(id);
        }

        return 0;
    }

    // entity.add_particles(e, {...}), entity.set_particles(e, {...}), entity.remove_particles(e): the same functions as
    // the entity methods, for scripts that prefer them as library calls.
    constexpr std::array entity_functions{
            luaL_Reg{"add_particles", &entity_add_particles},
            luaL_Reg{"set_particles", &entity_set_particles},
            luaL_Reg{"remove_particles", &entity_remove_particles},
            luaL_Reg{nullptr, nullptr},
    };

    constexpr std::array compute_functions{
            luaL_Reg{"load", &compute_load},
            luaL_Reg{"reload", &compute_reload},
            luaL_Reg{"instance", &compute_instance},
            luaL_Reg{"buffer", &compute_buffer},
            luaL_Reg{nullptr, nullptr},
    };

    constexpr std::array effect_methods{
            luaL_Reg{"set", &effect_set},
            luaL_Reg{"problem", &effect_problem},
            luaL_Reg{nullptr, nullptr},
    };

    constexpr std::array game_functions{
            luaL_Reg{"quit", &game_quit},
            luaL_Reg{"time", &game_time},
            luaL_Reg{nullptr, nullptr},
    };

    constexpr ImGuiWindowFlags base_window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing;

    auto field_number(lua_State *state, int table, char const *key, float fallback) -> float {
        return static_cast<float>(lua_field_number(state, table, key, fallback));
    }

    auto field_bool(lua_State *state, int table, char const *key) -> bool {
        lua_getfield(state, table, key);

        auto const value = lua_toboolean(state, -1) != 0;

        lua_pop(state, 1);

        return value;
    }

    auto ui_window(lua_State *state) -> int {
        auto const *const id = luaL_checkstring(state, 1);

        luaL_checktype(state, 2, LUA_TTABLE);
        luaL_checktype(state, 3, LUA_TFUNCTION);

        auto const *const viewport = ImGui::GetMainViewport();
        auto flags = base_window_flags;

        if (field_bool(state, 2, "fullscreen")) {
            ImGui::SetNextWindowPos(viewport->Pos);
            ImGui::SetNextWindowSize(viewport->Size);
            flags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav;
        } else {
            flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;

            if (field_bool(state, 2, "centred")) {
                ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2{0.5F, 0.5F});
            } else {
                ImGui::SetNextWindowPos(
                        ImVec2{viewport->WorkPos.x + field_number(state, 2, "x", 16.0F),
                               viewport->WorkPos.y + field_number(state, 2, "y", 16.0F)},
                        ImGuiCond_Always, ImVec2{field_number(state, 2, "pivot_x", 0.0F), field_number(state, 2, "pivot_y", 0.0F)});
            }
        }

        if (field_bool(state, 2, "no_inputs")) {
            flags |= ImGuiWindowFlags_NoInputs;
        }

        ImGui::SetNextWindowBgAlpha(field_number(state, 2, "bg_alpha", 0.85F));

        auto status = LUA_OK;

        if (ImGui::Begin(id, nullptr, flags)) {
            lua_pushvalue(state, 3);
            status = lua_pcall(state, 0, 0, 0);
        }

        ImGui::End();

        if (status != LUA_OK) {
            return lua_error(state);
        }

        return 0;
    }

    auto ui_text(lua_State *state) -> int {
        auto const *const text = luaL_checkstring(state, 1);
        auto const scale = static_cast<float>(luaL_optnumber(state, 2, 1.0));

        ImGui::SetWindowFontScale(scale);
        ImGui::TextUnformatted(text);
        ImGui::SetWindowFontScale(1.0F);

        return 0;
    }

    auto ui_text_disabled(lua_State *state) -> int {
        ImGui::TextDisabled("%s", luaL_checkstring(state, 1));

        return 0;
    }

    auto ui_text_centred(lua_State *state) -> int {
        auto const *const text = luaL_checkstring(state, 1);
        auto const scale = static_cast<float>(luaL_optnumber(state, 2, 1.0));

        ImGui::SetWindowFontScale(scale);

        auto const width = ImGui::CalcTextSize(text).x;

        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - width) * 0.5F));
        ImGui::TextUnformatted(text);
        ImGui::SetWindowFontScale(1.0F);

        return 0;
    }

    auto ui_button(lua_State *state) -> int {
        auto const *const label = luaL_checkstring(state, 1);
        auto const width = static_cast<float>(luaL_optnumber(state, 2, 300.0));
        auto const height = static_cast<float>(luaL_optnumber(state, 3, 52.0));
        auto const scale = static_cast<float>(luaL_optnumber(state, 4, 1.6));

        ImGui::SetWindowFontScale(scale);
        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - width) * 0.5F));

        auto const pressed = ImGui::Button(label, ImVec2{width, height});

        ImGui::SetWindowFontScale(1.0F);

        lua_pushboolean(state, pressed ? 1 : 0);

        return 1;
    }

    auto ui_small_button(lua_State *state) -> int {
        lua_pushboolean(state, ImGui::Button(luaL_checkstring(state, 1)) ? 1 : 0);

        return 1;
    }

    // ui.input_text(label, text [, width]) -> text: an edited single-line field, returned each frame.
    auto ui_input_text(lua_State *state) -> int {
        constexpr std::size_t capacity = 256;

        auto const *const label = luaL_checkstring(state, 1);
        auto const *const current = luaL_checkstring(state, 2);
        auto const width = static_cast<float>(luaL_optnumber(state, 3, 300.0));

        std::array<char, capacity> buffer{};

        std::strncpy(buffer.data(), current, capacity - 1);

        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - width) * 0.5F));
        ImGui::SetNextItemWidth(width);
        ImGui::InputText(label, buffer.data(), capacity);

        lua_pushstring(state, buffer.data());

        return 1;
    }

    auto ui_progress(lua_State *state) -> int {
        auto const fraction = static_cast<float>(luaL_checknumber(state, 1));
        auto const width = static_cast<float>(luaL_optnumber(state, 2, 300.0));
        auto const height = static_cast<float>(luaL_optnumber(state, 3, 6.0));

        ImGui::SetCursorPosX(std::max(0.0F, (ImGui::GetWindowSize().x - width) * 0.5F));
        ImGui::ProgressBar(fraction, ImVec2{width, height}, "");

        return 0;
    }

    auto ui_spinner(lua_State *state) -> int {
        constexpr int segments = 32;
        constexpr float sweep = 4.8F;

        auto const radius = static_cast<float>(luaL_optnumber(state, 1, 28.0));
        auto const thickness = static_cast<float>(luaL_optnumber(state, 2, 5.0));

        auto const origin = ImGui::GetCursorScreenPos();
        auto const centre = ImVec2{ImGui::GetWindowPos().x + ImGui::GetWindowSize().x * 0.5F, origin.y + radius + thickness};

        auto *const draw_list = ImGui::GetWindowDrawList();
        auto const start = std::fmod(static_cast<float>(ImGui::GetTime()) * 4.0F, 6.2831853F);

        draw_list->PathClear();

        for (int index = 0; index <= segments; ++index) {
            auto const angle = start + (static_cast<float>(index) / segments) * sweep;

            draw_list->PathLineTo(ImVec2{centre.x + std::cos(angle) * radius, centre.y + std::sin(angle) * radius});
        }

        draw_list->PathStroke(ImGui::GetColorU32(ImGuiCol_PlotHistogram), ImDrawFlags_None, thickness);

        ImGui::Dummy(ImVec2{0.0F, (radius + thickness) * 2.0F});

        return 0;
    }

    auto ui_dummy(lua_State *state) -> int {
        ImGui::Dummy(ImVec2{static_cast<float>(luaL_checknumber(state, 1)), static_cast<float>(luaL_checknumber(state, 2))});

        return 0;
    }

    auto ui_same_line(lua_State * ) -> int {
        ImGui::SameLine();

        return 0;
    }

    auto ui_separator(lua_State * ) -> int {
        ImGui::Separator();

        return 0;
    }

    auto ui_display_size(lua_State *state) -> int {
        auto const size = ImGui::GetMainViewport()->Size;

        lua_pushnumber(state, size.x);
        lua_pushnumber(state, size.y);

        return 2;
    }

    constexpr std::array ui_functions{
            luaL_Reg{"window", &ui_window},
            luaL_Reg{"text", &ui_text},
            luaL_Reg{"text_disabled", &ui_text_disabled},
            luaL_Reg{"text_centred", &ui_text_centred},
            luaL_Reg{"button", &ui_button},
            luaL_Reg{"small_button", &ui_small_button},
            luaL_Reg{"input_text", &ui_input_text},
            luaL_Reg{"progress", &ui_progress},
            luaL_Reg{"spinner", &ui_spinner},
            luaL_Reg{"dummy", &ui_dummy},
            luaL_Reg{"same_line", &ui_same_line},
            luaL_Reg{"separator", &ui_separator},
            luaL_Reg{"display_size", &ui_display_size},
            luaL_Reg{nullptr, nullptr},
    };

    struct IntegerConstant {
        char const *name;
        int value;
    };

    constexpr std::array key_constants{
            IntegerConstant{"ENTER", GLFW_KEY_ENTER},   IntegerConstant{"SPACE", GLFW_KEY_SPACE},
            IntegerConstant{"ESCAPE", GLFW_KEY_ESCAPE}, IntegerConstant{"BACKSPACE", GLFW_KEY_BACKSPACE},
            IntegerConstant{"TAB", GLFW_KEY_TAB},       IntegerConstant{"UP", GLFW_KEY_UP},
            IntegerConstant{"DOWN", GLFW_KEY_DOWN},     IntegerConstant{"LEFT", GLFW_KEY_LEFT},
            IntegerConstant{"RIGHT", GLFW_KEY_RIGHT},   IntegerConstant{"A", GLFW_KEY_A},
            IntegerConstant{"B", GLFW_KEY_B},           IntegerConstant{"C", GLFW_KEY_C},
            IntegerConstant{"D", GLFW_KEY_D},           IntegerConstant{"E", GLFW_KEY_E},
            IntegerConstant{"F", GLFW_KEY_F},           IntegerConstant{"G", GLFW_KEY_G},
            IntegerConstant{"H", GLFW_KEY_H},           IntegerConstant{"I", GLFW_KEY_I},
            IntegerConstant{"J", GLFW_KEY_J},           IntegerConstant{"K", GLFW_KEY_K},
            IntegerConstant{"L", GLFW_KEY_L},           IntegerConstant{"M", GLFW_KEY_M},
            IntegerConstant{"N", GLFW_KEY_N},           IntegerConstant{"O", GLFW_KEY_O},
            IntegerConstant{"P", GLFW_KEY_P},           IntegerConstant{"Q", GLFW_KEY_Q},
            IntegerConstant{"R", GLFW_KEY_R},           IntegerConstant{"S", GLFW_KEY_S},
            IntegerConstant{"T", GLFW_KEY_T},           IntegerConstant{"U", GLFW_KEY_U},
            IntegerConstant{"V", GLFW_KEY_V},           IntegerConstant{"W", GLFW_KEY_W},
            IntegerConstant{"X", GLFW_KEY_X},           IntegerConstant{"Y", GLFW_KEY_Y},
            IntegerConstant{"Z", GLFW_KEY_Z},           IntegerConstant{"F1", GLFW_KEY_F1},
            IntegerConstant{"F2", GLFW_KEY_F2},         IntegerConstant{"F3", GLFW_KEY_F3},
            IntegerConstant{"LEFT_SHIFT", GLFW_KEY_LEFT_SHIFT},
            IntegerConstant{"LEFT_CONTROL", GLFW_KEY_LEFT_CONTROL},
    };

    constexpr std::array mouse_constants{
            IntegerConstant{"LEFT", GLFW_MOUSE_BUTTON_LEFT},
            IntegerConstant{"RIGHT", GLFW_MOUSE_BUTTON_RIGHT},
            IntegerConstant{"MIDDLE", GLFW_MOUSE_BUTTON_MIDDLE},
    };

    template<std::size_t N>
    auto set_constants(lua_State *state, char const *table_name, std::array<IntegerConstant, N> const &constants) -> void {
        lua_createtable(state, 0, static_cast<int>(N));

        for (auto const &constant: constants) {
            lua_pushinteger(state, constant.value);
            lua_setfield(state, -2, constant.name);
        }

        lua_setglobal(state, table_name);
    }

    template<std::size_t N>
    auto set_library(lua_State *state, char const *table_name, std::array<luaL_Reg, N> const &functions) -> void {
        lua_createtable(state, 0, static_cast<int>(N));
        luaL_setfuncs(state, functions.data(), 0);
        lua_setglobal(state, table_name);
    }
}

auto open_lua_game_api(LuaRuntime &runtime) -> void {
    auto *const state = runtime.state();

    set_library(state, "scene",
                std::array{luaL_Reg{"spawn", &scene_spawn}, luaL_Reg{"find", &scene_find},
                           luaL_Reg{"clear", &scene_clear}, luaL_Reg{"add_effect", &scene_add_effect},
                           luaL_Reg{"remove_effect", &scene_remove_effect}, luaL_Reg{nullptr, nullptr}});
    set_library(state, "physics", physics_functions);
    set_library(state, "animation", animation_functions);
    set_library(state, "compute", compute_functions);
    set_library(state, "entity", entity_functions);
    set_library(state, "assets", assets_functions);
    set_library(state, "camera", camera_functions);
    set_library(state, "ui", ui_functions);

    lua_createtable(state, 0, 3);
    luaL_setfuncs(state, game_functions.data(), 0);
    lua_pushboolean(state, game_host(state).host.player_mode ? 1 : 0);
    lua_setfield(state, -2, "player_mode");
    lua_setglobal(state, "game");

    set_constants(state, "key", key_constants);
    set_constants(state, "mouse", mouse_constants);

    lua_getglobal(state, "key");
    lua_pushcfunction(state, &key_down);
    lua_setfield(state, -2, "down");
    lua_pop(state, 1);

    lua_getglobal(state, "mouse");
    lua_pushcfunction(state, &mouse_delta);
    lua_setfield(state, -2, "delta");
    lua_pop(state, 1);

    for (auto const &[meta, methods]: {std::pair<char const *, luaL_Reg const *>{rig_meta, rig_methods.data()},
                                       std::pair<char const *, luaL_Reg const *>{actor_meta, actor_methods.data()}}) {
        luaL_newmetatable(state, meta);
        lua_createtable(state, 0, 4);
        luaL_setfuncs(state, methods, 0);
        lua_setfield(state, -2, "__index");
        lua_pushstring(state, meta);
        lua_setfield(state, -2, "__name");
        lua_pop(state, 1);
    }

    luaL_newmetatable(state, character_meta);
    lua_createtable(state, 0, static_cast<int>(character_methods.size()));
    luaL_setfuncs(state, character_methods.data(), 0);
    lua_setfield(state, -2, "__index");
    lua_pushstring(state, character_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, entity_meta);
    lua_createtable(state, 0, static_cast<int>(entity_methods.size()));
    luaL_setfuncs(state, entity_methods.data(), 0);
    lua_setfield(state, -2, "__index");
    lua_pushstring(state, entity_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, effect_meta);
    lua_createtable(state, 0, static_cast<int>(effect_methods.size()));
    luaL_setfuncs(state, effect_methods.data(), 0);
    lua_setfield(state, -2, "__index");
    lua_pushcfunction(state, &effect_gc);
    lua_setfield(state, -2, "__gc");
    lua_pushstring(state, effect_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, effect_buffer_meta);
    lua_pushcfunction(state, &effect_buffer_gc);
    lua_setfield(state, -2, "__gc");
    lua_pushstring(state, effect_buffer_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, effect_shader_meta);
    lua_pushstring(state, effect_shader_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, model_meta);
    lua_pushstring(state, model_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    luaL_newmetatable(state, material_meta);
    lua_pushstring(state, material_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    lua_settop(state, 0);
}

auto update_lua_game_physics(LuaGameHost &host) -> void {
    if (host.scene == nullptr || host.scene->physics_world == nullptr) {
        return;
    }

    auto &world = *host.scene->physics_world;
    auto &registry = host.scene->get_registry();

    for (auto &pending: host.mesh_colliders) {
        if (pending.added_to_world == world.id() || !pending.slot->ready.load(std::memory_order_acquire) ||
            !registry.valid(pending.entity)) {
            continue;
        }

        world.add_static_mesh(pending.entity, registry.get_or_emplace<Components::Transform>(pending.entity),
                              pending.slot->mesh);
        pending.added_to_world = world.id();
    }
}
