#pragma once

#include <entt/entt.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "material.hxx"
#include "renderer.hxx"

class btRigidBody;
class btCollisionShape;

namespace Components {
    struct Lifetime {
        float remaining_seconds{0.0F};
    };

    struct Transform {
        glm::vec3 position{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        glm::vec3 scale{1.0F};

        [[nodiscard]] auto matrix() const -> glm::mat4 {
            return glm::translate(glm::mat4{1.0F}, position) * glm::mat4_cast(rotation) *
                   glm::scale(glm::mat4{1.0F}, scale);
        }
    };

    struct PointLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
    };

    struct SpotLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        float inner_cone_degrees = 20.0F;
        float outer_cone_degrees = 30.0F;
    };


    struct MaterialOverride {
        MaterialHandle material{};
    };

    struct Model {
        ModelHandle model{};
    };

    enum class BodyShape : std::uint8_t {
        box,
        capsule,
    };

    struct PlayerTag {};

    struct Parent {
        entt::entity entity{entt::null};
    };

    struct RigidBody {
        glm::vec3 velocity{0.0F}; // initial velocity, applied when the body is created
        glm::vec3 half_extents{0.5F}; // btBoxShape half-extents (shape == box)
        float capsule_radius = 0.4F; // btCapsuleShape radius (shape == capsule)
        float capsule_height = 1.0F; // btCapsuleShape cylinder height, excludes end caps (shape == capsule)
        float restitution = 0.4F;
        float mass = 1.0F; // ignored (treated as immovable) when is_static
        bool is_static = false;
        bool lock_rotation = false; // zero angular factor -- for player capsules, so collisions don't tip them
        BodyShape shape = BodyShape::box;

        static auto from_model_bounds(auto &&bounds) -> RigidBody {
            auto &&[min, max] = std::tuple(std::get<0>(bounds), std::get<1>(bounds));
            return RigidBody{.half_extents = (max - min) * 0.5F};
        }

        static auto make_capsule(float radius, float height, float mass = 1.0F) -> RigidBody {
            return RigidBody{
                    .capsule_radius = radius,
                    .capsule_height = height,
                    .mass = mass,
                    .lock_rotation = true,
            };
        }
    };

    // The live Bullet handle for an entity's RigidBody, added by
    // PhysicsWorld::add_body once the body exists in the simulation and
    // removed by PhysicsWorld::remove_body. Neither pointer is owned here --
    // PhysicsWorld's arena owns the storage; this is just how PhysicsWorld
    // finds an entity's body without keeping its own entity-keyed map (which
    // used to mean every per-frame transform writeback walked a hash map
    // instead of an entt::view).
    struct PhysicsBody {
        btRigidBody *rigid_body = nullptr;
        btCollisionShape *shape = nullptr;
    };
} // namespace Components
