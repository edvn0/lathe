#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

class btRigidBody;
class btCollisionShape;

// Physics-specific ECS components, split out of components.hxx so
// physics_world.hxx/.cxx can depend on this instead of the full Components
// namespace (which also pulls in MaterialHandle/ModelHandle from the assets
// layer) -- physics sits below assets in the module layering.
namespace Components {
    enum class BodyShape : std::uint8_t {
        box,
        capsule,
        heightfield,
        compound,
    };

    // One box child of a BodyShape::compound collider -- local_centre/
    // half_extents are in the owning RigidBody's local space (no rotation:
    // each box is already axis-aligned in that space, since
    // Renderer::model_submesh_bounds folds every submesh's own AABB through
    // its node transform conservatively rather than actually rotating the
    // box -- see that function's comment).
    struct CompoundBoxChild {
        glm::vec3 local_centre{0.0F};
        glm::vec3 half_extents{0.5F};
    };

    // Static, axis-aligned heightfield collider backing BodyShape::heightfield
    // -- see PhysicsWorld::add_body and make_terrain_mesh (terrain_mesh.hxx),
    // whose TerrainMeshResult::heights is shaped to be handed here directly.
    // Held by shared_ptr because btHeightfieldTerrainShape keeps a raw
    // pointer into `heights` for as long as the shape exists (it never
    // copies the data) -- the RigidBody component staying alive for the
    // entity's lifetime is what keeps that pointer valid.
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
        glm::vec3 velocity{0.0F}; // initial velocity, applied when the body is created
        glm::vec3 half_extents{0.5F}; // btBoxShape half-extents (shape == box)
        float capsule_radius = 0.4F; // btCapsuleShape radius (shape == capsule)
        float capsule_height = 1.0F; // btCapsuleShape cylinder height, excludes end caps (shape == capsule)
        float restitution = 0.4F;
        float mass = 1.0F; // ignored (treated as immovable) when is_static
        bool is_static = false;
        bool lock_rotation = false; // zero angular factor -- for player capsules, so collisions don't tip them
        BodyShape shape = BodyShape::box;
        std::shared_ptr<HeightfieldShape const> heightfield{}; // shape == heightfield
        std::shared_ptr<std::vector<CompoundBoxChild> const> compound_boxes{}; // shape == compound

        static auto from_model_bounds(auto &&bounds) -> RigidBody {
            auto &&[min, max] = std::tuple(std::get<0>(bounds), std::get<1>(bounds));
            return RigidBody{.half_extents = (max - min) * 0.5F};
        }

        // Static collider approximating a large multi-submesh model (e.g. a
        // Sponza-like glTF scene, one primitive per wall/column) with one
        // small btBoxShape per submesh instead of a single box over the
        // whole model's bounds -- pass Renderer::model_submesh_bounds()'s
        // result straight through. Always static, same reasoning as
        // make_heightfield: a compound of many boxes baked from level
        // geometry has no sensible dynamic (falling, tumbling) behaviour.
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

        // Always static -- a heightfield can't meaningfully be a dynamic
        // rigid body (Bullet treats btHeightfieldTerrainShape as a concave,
        // world-static shape).
        static auto make_heightfield(std::shared_ptr<HeightfieldShape const> shape) -> RigidBody {
            return RigidBody{
                    .is_static = true,
                    .shape = BodyShape::heightfield,
                    .heightfield = std::move(shape),
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
