#include "scene.hxx"

#include "physics_world.hxx"

Scene::Scene() = default;

// Out-of-line even though it's a plain default: physics_world's deleter
// needs PhysicsWorld's complete definition, which scene.hxx deliberately
// doesn't include.
Scene::~Scene() = default;

auto Scene::on_scene_start() -> void {
    physics_world = std::make_unique<PhysicsWorld>(physics_settings);
    physics_world->populate_from(registry);
}

auto Scene::on_scene_stop() -> void { physics_world.reset(); }
