#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

struct ModelCpuData;

// What a ray hit on a model's surface: everything is in the model's own space (node transforms applied), which is the
// space of the entity's transform.
struct ModelPickHit {
    float distance = 0.0F;
    glm::vec3 position{0.0F};
    // Unit, facing the ray's origin.
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    std::uint32_t triangle = 0;
    std::array<glm::vec3, 3> corners{};
};

// A model's triangles with a bounding-volume hierarchy over them, kept on the CPU for ray queries (editor picking and
// placing things on a surface). The renderer does not keep triangles after the upload, so this is a copy made from the
// parsed model; build it off the main thread.
class ModelPickGeometry {
public:
    [[nodiscard]] static auto build(ModelCpuData const &model) -> std::shared_ptr<ModelPickGeometry const>;
    [[nodiscard]] static auto build(std::vector<glm::vec3> positions, std::vector<std::uint32_t> indices)
            -> std::shared_ptr<ModelPickGeometry const>;

    // The nearest triangle the ray hits within `max_distance` (`distance` is in units of `direction`'s length, so a unit direction gives model units).
    // Triangles are hit from either side.
    [[nodiscard]] auto raycast(glm::vec3 const &origin, glm::vec3 const &direction, float max_distance) const
            -> std::optional<ModelPickHit>;

    [[nodiscard]] auto triangle_count() const noexcept -> std::size_t { return indices_.size() / 3; }

private:
    struct Node {
        glm::vec3 low{0.0F};
        glm::vec3 high{0.0F};
        // A leaf holds triangles [first, first + count) of the reordered list; otherwise count is 0 and `first` and
        // `right` are the child nodes.
        std::uint32_t first = 0;
        std::uint32_t count = 0;
        std::uint32_t right = 0;
    };

    ModelPickGeometry() = default;

    auto build_node(std::uint32_t first, std::uint32_t count, std::vector<glm::vec3> const &centroids,
                    std::vector<std::uint32_t> &order) -> std::uint32_t;

    std::vector<glm::vec3> positions_;
    std::vector<std::uint32_t> indices_;
    std::vector<Node> nodes_;
};
