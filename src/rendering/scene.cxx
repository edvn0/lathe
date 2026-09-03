#include "rendering/scene.hxx"

#include "scene/components.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/entity.hxx"
#include "physics/physics_world.hxx"
#include "rendering/renderer.hxx"
#include "rendering/script_storage.hxx"
#include "core/thread_pool.hxx"

#include <future>
#include <unordered_map>

namespace {
    // ScriptHandle has no std::hash specialization and generation can't
    // change mid-frame, so group by index alone -- ScriptStorage::get()
    // below still revalidates the full handle.
    auto group_entities_by_script(entt::registry &registry)
            -> std::unordered_map<std::uint32_t, std::pair<ScriptHandle, std::vector<entt::entity>>> {
        std::unordered_map<std::uint32_t, std::pair<ScriptHandle, std::vector<entt::entity>>> groups;

        for (auto entity: registry.view<Components::Script>()) {
            auto const handle = registry.get<Components::Script>(entity).script;

            if (!handle.valid()) {
                continue;
            }

            auto &group = groups[handle.index];
            group.first = handle;
            group.second.push_back(entity);
        }

        return groups;
    }
} // namespace

Scene::Scene(Renderer &renderer) : renderer_(renderer) {
    connect_light_signals();
    entt::sink{lights_changed_signal_}.connect<&Renderer::mark_lights_dirty>(renderer);
}

Scene::~Scene() = default;

auto Scene::on_scene_start() -> void {
    physics_world = std::make_unique<PhysicsWorld>(physics_settings, thread_pool(), registry);
    physics_world->populate_from(registry);
}

auto Scene::on_scene_stop() -> void { physics_world.reset(); }

auto Scene::step(float delta_time) -> void {
    ZoneScopedNC("SceneStep", tracy::Color::Firebrick);

    systems::script_update(get_registry(), *this, std::min(delta_time, 0.25F));
    physics_world->step(get_registry(), std::min(delta_time, 0.25F));
}

auto Scene::get_scripts() noexcept -> ScriptStorage & { return renderer_.script_storage(); }

auto Scene::get_scripts() const noexcept -> ScriptStorage const & { return renderer_.script_storage(); }

auto Scene::mark_lights_dirty(entt::registry &, entt::entity) -> void { lights_changed_signal_.publish(); }

auto Scene::on_transform_changed(entt::registry &reg, entt::entity entity) -> void {
    if (reg.any_of<Components::PointLight, Components::SpotLight>(entity)) {
        lights_changed_signal_.publish();
    }
}

// registry.destroy()/remove<RigidBody>() on a physics entity would otherwise
// leak its btRigidBody/btCollisionShape in both the arena and Bullet's world
// forever -- systems::lifetime() already calls physics_world->remove_body()
// itself before destroying an expired entity, but remove_body() is a no-op
// for an entity it doesn't know about, so this hook is the safety net for
// every other path (editor deletion, future gameplay code) rather than the
// only place cleanup happens.
auto Scene::on_rigid_body_destroyed(entt::registry &reg, entt::entity entity) -> void {
    if (physics_world) {
        physics_world->remove_body(reg, entity);
    }
}

auto Scene::on_script_attached(entt::registry &reg, entt::entity entity) -> void {
    auto const handle = reg.get<Components::Script>(entity).script;
    if (auto *script = renderer_.script_storage().get(handle)) {
        script->on_attach(Entity{this, entity});
    }
}

auto Scene::on_script_detached(entt::registry &reg, entt::entity entity) -> void {
    auto const handle = reg.get<Components::Script>(entity).script;
    if (auto *script = renderer_.script_storage().get(handle)) {
        script->on_detach(Entity{this, entity});
    }
}

auto Scene::attach_debug_renderer(debug_draw::DebugRenderer &renderer) -> void {
    if (physics_world) {
        physics_world->attach_debug_drawer(renderer);
    }
}

auto Scene::detach_debug_renderer() -> void {
    if (physics_world) {
        physics_world->detach_debug_drawer();
    }
}

auto Scene::connect_light_signals() -> void {
    registry.on_construct<Components::PointLight>().connect<&Scene::mark_lights_dirty>(*this);
    registry.on_update<Components::PointLight>().connect<&Scene::mark_lights_dirty>(*this);
    registry.on_destroy<Components::PointLight>().connect<&Scene::mark_lights_dirty>(*this);

    registry.on_construct<Components::SpotLight>().connect<&Scene::mark_lights_dirty>(*this);
    registry.on_update<Components::SpotLight>().connect<&Scene::mark_lights_dirty>(*this);
    registry.on_destroy<Components::SpotLight>().connect<&Scene::mark_lights_dirty>(*this);

    registry.on_update<Components::Transform>().connect<&Scene::on_transform_changed>(*this);

    registry.on_destroy<Components::RigidBody>().connect<&Scene::on_rigid_body_destroyed>(*this);

    registry.on_construct<Components::Script>().connect<&Scene::on_script_attached>(*this);
    registry.on_destroy<Components::Script>().connect<&Scene::on_script_detached>(*this);
}

void systems::lifetime(entt::registry &registry, PhysicsWorld &physics, float dt) {
    auto view = registry.view<Components::Lifetime>();

    std::vector<entt::entity> expired;

    for (auto entity: view) {
        auto &lifetime = view.get<Components::Lifetime>(entity);
        lifetime.remaining_seconds -= dt;
        if (lifetime.remaining_seconds <= 0.0f) {
            expired.push_back(entity);
        }
    }

    for (auto entity: expired) {
        physics.remove_body(registry, entity);
        registry.destroy(entity);
    }
}

auto systems::script_update(entt::registry &registry, Scene &scene, float delta_time) -> void {
    auto groups = group_entities_by_script(registry);

    std::vector<std::future<void>> futures;
    futures.reserve(groups.size());

    for (auto &[index, group]: groups) {
        auto &[handle, entities] = group;

        auto *script = scene.get_scripts().get(handle);
        if (script == nullptr) {
            continue;
        }

        if (script->parallelizable() && entities.size() > 1) {
            futures.push_back(thread_pool().submit_task([script, &scene, entities = std::move(entities), delta_time] {
                for (auto entity: entities) {
                    script->on_update(ScriptEntity{&scene, entity}, delta_time);
                }
            }));
        } else {
            for (auto entity: entities) {
                script->on_update(ScriptEntity{&scene, entity}, delta_time);
            }
        }
    }

    for (auto &future: futures) {
        future.get(); // barrier -- next system stage must see every update applied
    }
}

auto systems::get_world_transform(entt::registry const &registry, entt::entity entity,
                                  Components::Transform const &transform) -> glm::mat4 {
    auto world_matrix = transform.matrix();

    auto const *parent = registry.try_get<Components::Parent const>(entity);

    for (auto depth = 0; depth < 64 && parent != nullptr; ++depth) {
        auto const parent_entity = parent->entity;

        if (!registry.valid(parent_entity)) {
            break;
        }

        if (auto const *parent_transform = registry.try_get<Components::Transform const>(parent_entity)) {
            world_matrix = parent_transform->matrix() * world_matrix;
        }

        parent = registry.try_get<Components::Parent const>(parent_entity);
    }

    return world_matrix;
}
