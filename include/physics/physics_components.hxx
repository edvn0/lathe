#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

class btRigidBody;
class btCollisionShape;

// Kept apart from components.hxx so physics doesn't depend on the assets layer.
namespace Components {
    enum class BodyShape : std::uint8_t {
        box,
        capsule,
        heightfield,
        compound,
    };

    // One box of a compound collider, axis-aligned in the RigidBody's local space.
    struct CompoundBoxChild {
        glm::vec3 local_centre{0.0F};
        glm::vec3 half_extents{0.5F};
    };

    // Heightfield collider data, shaped like TerrainMeshResult::heights. Shared because Bullet keeps a raw pointer
    // into `heights`.
    struct HeightfieldShape {
        std::shared_ptr<std::vector<float> const> heights; // row-major, size == width * length
        std::uint32_t width = 2; // samples along local X
        std::uint32_t length = 2; // samples along local Z
        float min_height = 0.0F;
        float max_height = 0.0F;
        float cell_size_x = 1.0F;
        float cell_size_z = 1.0F;
    };

    struct RigidBody {
        glm::vec3 velocity{0.0F}; // applied when the body is created
        glm::vec3 half_extents{0.5F}; // shape == box
        float capsule_radius = 0.4F; // shape == capsule
        float capsule_height = 1.0F; // shape == capsule, excluding the caps
        float restitution = 0.4F;
        float mass = 1.0F; // ignored when is_static
        bool is_static = false;
        bool lock_rotation = false; // zero angular factor so collisions don't tip the body over
        BodyShape shape = BodyShape::box;
        std::shared_ptr<HeightfieldShape const> heightfield{}; // shape == heightfield
        std::shared_ptr<std::vector<CompoundBoxChild> const> compound_boxes{}; // shape == compound

        static auto from_model_bounds(auto &&bounds) -> RigidBody {
            auto &&[min, max] = std::tuple(std::get<0>(bounds), std::get<1>(bounds));
            return RigidBody{.half_extents = (max - min) * 0.5F};
        }

        // Static collider approximating a large model with one box per submesh; pass
        // Renderer::model_submesh_bounds() straight in.
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

        // Always static; Bullet heightfields can't be dynamic.
        static auto make_heightfield(std::shared_ptr<HeightfieldShape const> shape) -> RigidBody {
            return RigidBody{
                    .is_static = true,
                    .shape = BodyShape::heightfield,
                    .heightfield = std::move(shape),
            };
        }
    };

    // The entity's live Bullet body, added by PhysicsWorld::add_body and removed by remove_body. Owned by
    // PhysicsWorld's arena.
    struct PhysicsBody {
        btRigidBody *rigid_body = nullptr;
        btCollisionShape *shape = nullptr;
    };
} // namespace Components
