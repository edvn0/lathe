#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Animation {
    inline constexpr std::int32_t no_parent = -1;

    struct JointTransform {
        glm::vec3 translation{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        glm::vec3 scale{1.0F};
    };

    struct PoseView {
        std::span<glm::vec3> translation;
        std::span<glm::quat> rotation;
        std::span<glm::vec3> scale;

        [[nodiscard]] auto size() const -> std::size_t { return translation.size(); }
    };

    struct ConstPoseView {
        std::span<glm::vec3 const> translation;
        std::span<glm::quat const> rotation;
        std::span<glm::vec3 const> scale;

        ConstPoseView() = default;
        ConstPoseView(std::span<glm::vec3 const> t, std::span<glm::quat const> r, std::span<glm::vec3 const> s)
            : translation{t}, rotation{r}, scale{s} {}
        // NOLINTNEXTLINE(google-explicit-constructor): a mutable view is always usable as a const one.
        ConstPoseView(PoseView const &view) : translation{view.translation}, rotation{view.rotation}, scale{view.scale} {}

        [[nodiscard]] auto size() const -> std::size_t { return translation.size(); }
    };

    class Pose {
    public:
        Pose() = default;
        explicit Pose(std::size_t joint_count)
            : translation_(joint_count, glm::vec3{0.0F}),
              rotation_(joint_count, glm::quat{1.0F, 0.0F, 0.0F, 0.0F}),
              scale_(joint_count, glm::vec3{1.0F}) {}

        [[nodiscard]] auto size() const -> std::size_t { return translation_.size(); }
        [[nodiscard]] auto view() -> PoseView { return {translation_, rotation_, scale_}; }
        [[nodiscard]] auto view() const -> ConstPoseView { return {translation_, rotation_, scale_}; }

        [[nodiscard]] auto joint(std::size_t index) const -> JointTransform {
            return {translation_[index], rotation_[index], scale_[index]};
        }
        void set_joint(std::size_t index, JointTransform const &transform) {
            translation_[index] = transform.translation;
            rotation_[index] = transform.rotation;
            scale_[index] = transform.scale;
        }

    private:
        std::vector<glm::vec3> translation_;
        std::vector<glm::quat> rotation_;
        std::vector<glm::vec3> scale_;
    };

    class Skeleton {
    public:
        Skeleton(std::vector<std::string> names, std::vector<std::int32_t> parents, Pose bind_pose,
                 std::vector<glm::mat4> inverse_bind = {});

        [[nodiscard]] auto joint_count() const -> std::size_t { return parents_.size(); }
        [[nodiscard]] auto names() const -> std::span<std::string const> { return names_; }
        [[nodiscard]] auto parents() const -> std::span<std::int32_t const> { return parents_; }
        [[nodiscard]] auto bind_pose() const -> Pose const & { return bind_pose_; }
        [[nodiscard]] auto inverse_bind() const -> std::span<glm::mat4 const> { return inverse_bind_; }
        [[nodiscard]] auto find_joint(std::string_view name) const -> std::optional<std::size_t>;

    private:
        std::vector<std::string> names_;
        std::vector<std::int32_t> parents_;
        Pose bind_pose_;
        std::vector<glm::mat4> inverse_bind_;
    };

    [[nodiscard]] auto local_matrix(glm::vec3 translation, glm::quat rotation, glm::vec3 scale) -> glm::mat4;

    void compute_model_matrices(Skeleton const &skeleton, ConstPoseView pose, std::span<glm::mat4> out);

    void compute_skinning_palette(Skeleton const &skeleton, ConstPoseView pose, std::span<glm::mat4> out);

    void copy_pose(ConstPoseView source, PoseView destination);

    void blend_poses(ConstPoseView a, ConstPoseView b, float t, PoseView out);
}
