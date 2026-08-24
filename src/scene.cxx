#include "scene.hxx"

#include "components.hxx"
#include "physics_world.hxx"

#include "player_camera.hxx"
#include "player_controller.hxx"

Scene::Scene(Renderer &renderer) {
    connect_light_signals();
    entt::sink{lights_changed_signal_}.connect<&Renderer::mark_lights_dirty>(renderer);
}

Scene::~Scene() = default;

auto Scene::on_scene_start() -> void {
    physics_world = std::make_unique<PhysicsWorld>(physics_settings, Renderer::thread_pool());
    physics_world->populate_from(registry);
}

auto Scene::on_scene_stop() -> void { physics_world.reset(); }

auto Scene::step(float delta_time) -> void { physics_world->step(get_registry(), std::min(delta_time, 0.25F)); }

auto Scene::mark_lights_dirty(entt::registry &, entt::entity) -> void { lights_changed_signal_.publish(); }

auto Scene::on_transform_changed(entt::registry &reg, entt::entity entity) -> void {
    if (reg.any_of<Components::PointLight, Components::SpotLight>(entity)) {
        lights_changed_signal_.publish();
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

auto systems::player_movement(entt::registry &registry, PhysicsWorld &physics_world, entt::entity player_entity,
                              PlayerController &controller, PlayerCamera &camera, float delta_time) -> void {
    if (!registry.valid(player_entity) ||
        !registry.all_of<Components::Transform, Components::RigidBody>(player_entity)) {
        return;
    }

    auto const &transform = registry.get<Components::Transform>(player_entity);
    auto const &body = registry.get<Components::RigidBody>(player_entity);

    auto const capsule_half_height = body.capsule_height * 0.5F;
    auto const is_grounded = physics_world.is_grounded(player_entity, capsule_half_height, body.capsule_radius);

    if (controller.consumes_jump() && is_grounded) {
        constexpr float jump_velocity = 6.5F; // Adjust to match gravity
        physics_world.jump(player_entity, jump_velocity);
    }

    auto const desired_velocity = controller.desired_horizontal_velocity();
    physics_world.set_velocity(player_entity, desired_velocity);

    auto const speed_factor =
            controller.move_speed() > 0.0F ? glm::length(desired_velocity) / controller.move_speed() : 0.0F;

    camera.update(transform.position, controller.yaw_degrees(), controller.pitch_degrees(), speed_factor, delta_time);
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
