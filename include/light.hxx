#pragma once

#include <glm/vec3.hpp>

// Punctual light ECS components. Position comes from the entity's Transform
// (see submit_scene() in main.cxx); SpotLight's direction comes from
// Transform::rotation applied to -Y (a spot light "points down" by
// convention, matching glTF KHR_lights_punctual). Plain, default-constructible,
// copyable value types -- clone_registry round-trips components through
// std::any (see scene.hxx), so anything non-trivial (handles, pointers) would
// break Play/Stop.
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
