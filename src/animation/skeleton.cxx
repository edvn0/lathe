#include "animation/skeleton.hxx"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cassert>
#include <utility>

namespace Animation {
    auto local_matrix(glm::vec3 const translation, glm::quat const rotation, glm::vec3 const scale) -> glm::mat4 {
        auto const rotation_matrix = glm::mat3_cast(rotation);
        return glm::mat4{glm::vec4{rotation_matrix[0] * scale.x, 0.0F}, glm::vec4{rotation_matrix[1] * scale.y, 0.0F},
                         glm::vec4{rotation_matrix[2] * scale.z, 0.0F}, glm::vec4{translation, 1.0F}};
    }

    Skeleton::Skeleton(std::vector<std::string> names, std::vector<std::int32_t> parents, Pose bind_pose,
                       std::vector<glm::mat4> inverse_bind)
        : names_{std::move(names)}, parents_{std::move(parents)}, bind_pose_{std::move(bind_pose)},
          inverse_bind_{std::move(inverse_bind)} {
        assert(names_.size() == parents_.size());
        assert(bind_pose_.size() == parents_.size());
        for (std::size_t i = 0; i < parents_.size(); ++i) {
            assert(parents_[i] < static_cast<std::int32_t>(i));
        }

        if (inverse_bind_.empty()) {
            inverse_bind_.resize(parents_.size());
            compute_model_matrices(*this, bind_pose_.view(), inverse_bind_);
            for (auto &matrix : inverse_bind_) {
                matrix = glm::inverse(matrix);
            }
        }
        assert(inverse_bind_.size() == parents_.size());
    }

    auto Skeleton::find_joint(std::string_view const name) const -> std::optional<std::size_t> {
        auto const found = std::ranges::find(names_, name);
        if (found == names_.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(found - names_.begin());
    }

    void compute_model_matrices(Skeleton const &skeleton, ConstPoseView const pose, std::span<glm::mat4> const out) {
        auto const parents = skeleton.parents();
        assert(out.size() == parents.size() && pose.size() == parents.size());
        for (std::size_t i = 0; i < parents.size(); ++i) {
            auto const local = local_matrix(pose.translation[i], pose.rotation[i], pose.scale[i]);
            out[i] = parents[i] == no_parent ? local : out[static_cast<std::size_t>(parents[i])] * local;
        }
    }

    void compute_skinning_palette(Skeleton const &skeleton, ConstPoseView const pose, std::span<glm::mat4> const out) {
        compute_model_matrices(skeleton, pose, out);
        auto const inverse_bind = skeleton.inverse_bind();
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] *= inverse_bind[i];
        }
    }

    void copy_pose(ConstPoseView const source, PoseView const destination) {
        std::ranges::copy(source.translation, destination.translation.begin());
        std::ranges::copy(source.rotation, destination.rotation.begin());
        std::ranges::copy(source.scale, destination.scale.begin());
    }

    void blend_poses(ConstPoseView const a, ConstPoseView const b, float const t, PoseView const out) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out.translation[i] = glm::mix(a.translation[i], b.translation[i], t);
            out.rotation[i] = glm::normalize(glm::slerp(a.rotation[i], b.rotation[i], t));
            out.scale[i] = glm::mix(a.scale[i], b.scale[i], t);
        }
    }
}
