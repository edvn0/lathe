#include "animation/batch.hxx"

#include <algorithm>
#include <cassert>

namespace Animation {
    AnimationBatch::AnimationBatch(Skeleton const &skeleton, AnimStateMachine const &machine, std::size_t const count,
                                   BS::priority_thread_pool *const pool, BatchSettings const settings)
        : skeleton_{skeleton}, machine_{machine}, pool_{pool}, settings_{settings}, count_{count},
          joint_count_{skeleton.joint_count()}, states_(count), pending_dt_(count, 0.0F),
          translation_(count * joint_count_), rotation_(count * joint_count_), scale_(count * joint_count_),
          scratch_translation_(count * joint_count_), scratch_rotation_(count * joint_count_),
          scratch_scale_(count * joint_count_), palette_(count * joint_count_) {
        auto const bind = skeleton.bind_pose().view();
        for (std::size_t i = 0; i < count; ++i) {
            auto const slice = std::span{palette_}.subspan(i * joint_count_, joint_count_);
            compute_skinning_palette(skeleton, bind, slice);
        }
    }

    auto AnimationBatch::palette(std::size_t const character) const -> std::span<glm::mat4 const> {
        return std::span{palette_}.subspan(character * joint_count_, joint_count_);
    }

    void AnimationBatch::update(std::span<AnimInputs const> const inputs, float const dt) {
        assert(inputs.size() == count_);
        auto const chunk = std::max<std::size_t>(settings_.chunk_size, 1);
        if (pool_ == nullptr || count_ <= chunk) {
            update_range(0, count_, inputs, dt);
            return;
        }
        auto const blocks = (count_ + chunk - 1) / chunk;
        auto future = pool_->submit_blocks(
                std::size_t{0}, count_,
                [this, inputs, dt](std::size_t const begin, std::size_t const end) { update_range(begin, end, inputs, dt); },
                blocks);
        future.wait();
    }

    void AnimationBatch::update_range(std::size_t const begin, std::size_t const end,
                                      std::span<AnimInputs const> const inputs, float const dt) {
        for (auto i = begin; i < end; ++i) {
            auto const lod = inputs[i].lod;
            if (lod == Lod::Hold) {
                pending_dt_[i] = 0.0F;
                continue;
            }
            pending_dt_[i] += dt;
            if (lod == Lod::Reduced && pending_dt_[i] < settings_.reduced_period) {
                continue;
            }
            auto const step = pending_dt_[i];
            pending_dt_[i] = 0.0F;

            auto const offset = i * joint_count_;
            auto const pose = PoseView{std::span{translation_}.subspan(offset, joint_count_),
                                       std::span{rotation_}.subspan(offset, joint_count_),
                                       std::span{scale_}.subspan(offset, joint_count_)};
            auto const scratch = PoseView{std::span{scratch_translation_}.subspan(offset, joint_count_),
                                          std::span{scratch_rotation_}.subspan(offset, joint_count_),
                                          std::span{scratch_scale_}.subspan(offset, joint_count_)};
            machine_.update(states_[i], inputs[i], step, pose, scratch);
            compute_skinning_palette(skeleton_, pose, std::span{palette_}.subspan(offset, joint_count_));
        }
    }
}
