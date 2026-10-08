#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

class btRigidBody;
class btCollisionShape;

namespace Components {
    enum class BodyShape : std::uint8_t {
        box,
        capsule,
        heightfield,
        compound,
        sphere,
    };

    struct CompoundBoxChild {
        glm::vec3 local_centre{0.0F};
        glm::vec3 half_extents{0.5F};
    };

    struct HeightfieldShape {
        std::shared_ptr<std::vector<float> const> heights;
        std::uint32_t width = 2;
        std::uint32_t length = 2;
        float min_height = 0.0F;
        float max_height = 0.0F;
        float cell_size_x = 1.0F;
        float cell_size_z = 1.0F;
    };

    struct RigidBody {
        glm::vec3 velocity{0.0F};
        glm::vec3 half_extents{0.5F};
        float capsule_radius = 0.4F;
        float capsule_height = 1.0F;
        float sphere_radius = 0.5F;
        float restitution = 0.4F;
        float mass = 1.0F;
        bool is_static = false;
        bool lock_rotation = false;
        BodyShape shape = BodyShape::box;
        std::shared_ptr<HeightfieldShape const> heightfield{};
        std::shared_ptr<std::vector<CompoundBoxChild> const> compound_boxes{};

        static auto from_model_bounds(auto &&bounds) -> RigidBody {
            auto &&[min, max] = std::tuple(std::get<0>(bounds), std::get<1>(bounds));
            return RigidBody{.half_extents = (max - min) * 0.5F};
        }

        static auto from_submesh_boxes(std::vector<std::pair<glm::vec3, glm::vec3>> const &boxes) -> RigidBody {
            auto children = std::make_shared<std::vector<CompoundBoxChild>>();
            children->reserve(boxes.size());

            for (auto const &[min, max]: boxes) {
                children->push_back(CompoundBoxChild{
                        .local_centre = (min + max) * 0.5F,
                        .half_extents = (max - min) * 0.5F,
                });
            }

            return RigidBody{
                    .is_static = true,
                    .shape = BodyShape::compound,
                    .compound_boxes = std::move(children),
            };
        }

        static auto make_capsule(float radius, float height, float mass = 1.0F) -> RigidBody {
            return RigidBody{
                    .capsule_radius = radius,
                    .capsule_height = height,
                    .mass = mass,
                    .lock_rotation = true,
                    .shape = BodyShape::capsule,
            };
        }

        static auto make_sphere(float radius, float mass = 1.0F, float restitution = 0.5F) -> RigidBody {
            return RigidBody{
                    .sphere_radius = radius,
                    .restitution = restitution,
                    .mass = mass,
                    .shape = BodyShape::sphere,
            };
        }

        static auto make_heightfield(std::shared_ptr<HeightfieldShape const> shape) -> RigidBody {
            return RigidBody{
                    .is_static = true,
                    .shape = BodyShape::heightfield,
                    .heightfield = std::move(shape),
            };
        }
    };

    struct PhysicsBody {
        btRigidBody *rigid_body = nullptr;
        btCollisionShape *shape = nullptr;
    };
}
