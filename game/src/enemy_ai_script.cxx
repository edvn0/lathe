#include "enemy_ai_script.hxx"

#include "scene/components.hxx"

#include <cmath>

auto EnemyAIScript::on_update(ScriptEntity entity, float delta_time) -> void {
    if (!entity.has<Components::Transform>() || !entity.has<Components::CircularMotion>()) {
        return;
    }

    auto &motion = entity.get<Components::CircularMotion>();
    motion.angle += motion.angular_speed * delta_time;

    auto &transform = entity.get<Components::Transform>();
    transform.position = motion.center + glm::vec3{motion.radius * std::cos(motion.angle), 0.0F,
                                                    motion.radius * std::sin(motion.angle)};
}
