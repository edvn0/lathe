#include "scene.hxx"

#include "components.hxx"
#include "physics_world.hxx"

Scene::Scene() { connect_light_signals(); }

Scene::~Scene() = default;

auto Scene::on_scene_start() -> void {
    physics_world = std::make_unique<PhysicsWorld>(physics_settings);
    physics_world->populate_from(registry);
}

auto Scene::on_scene_stop() -> void { physics_world.reset(); }

auto Scene::step(float delta_time) -> void { physics_world->step(get_registry(), std::min(delta_time, 0.25F)); }

auto Scene::mark_lights_dirty(entt::registry &, entt::entity) -> void { lights_dirty = true; }

auto Scene::on_transform_changed(entt::registry &reg, entt::entity entity) -> void {
    if (reg.any_of<Components::PointLight, Components::SpotLight>(entity)) {
        lights_dirty = true;
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
        physics.remove_body(entity);
        registry.destroy(entity);
    }
}
