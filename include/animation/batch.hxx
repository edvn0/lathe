#pragma once

#include "animation/state_machine.hxx"

#include <BS_thread_pool.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace Animation {
    struct BatchSettings {
        std::size_t chunk_size{128};    // characters per pool task
        float reduced_period{1.0F / 15.0F}; // seconds between evaluations at Lod::Reduced
    };

    // N characters sharing one Skeleton and state machine. All per-character data lives in contiguous arrays
    // and each pool chunk touches only its own slices, so no locking is needed.
    class AnimationBatch {
    public:
        // `pool` may be null, which evaluates serially on the calling thread.
        AnimationBatch(Skeleton const &skeleton, AnimStateMachine const &machine, std::size_t count,
                       BS::priority_thread_pool *pool, BatchSettings settings = {});
        AnimationBatch(AnimationBatch const &) = delete;
        auto operator=(AnimationBatch const &) -> AnimationBatch & = delete;
        AnimationBatch(AnimationBatch &&) = delete;
        auto operator=(AnimationBatch &&) -> AnimationBatch & = delete;
        ~AnimationBatch() = default;

        [[nodiscard]] auto count() const -> std::size_t { return count_; }

        // Advances every character by `dt`. `inputs.size()` must equal count().
        void update(std::span<AnimInputs const> inputs, float dt);

        // joint_count() matrices per character, characters back to back: one buffer for a single GPU upload.
        [[nodiscard]] auto palette() const -> std::span<glm::mat4 const> { return palette_; }
        [[nodiscard]] auto palette(std::size_t character) const -> std::span<glm::mat4 const>;
        [[nodiscard]] auto state(std::size_t character) const -> StateMachineState const & { return states_[character]; }
        [[nodiscard]] auto joint_count() const -> std::size_t { return joint_count_; }

    private:
        void update_range(std::size_t begin, std::size_t end, std::span<AnimInputs const> inputs, float dt);

        Skeleton const &skeleton_;
        AnimStateMachine const &machine_;
        BS::priority_thread_pool *pool_;
        BatchSettings settings_;
        std::size_t count_;
        std::size_t joint_count_;

        std::vector<StateMachineState> states_;
        std::vector<float> pending_dt_; // time accumulated while a character was skipped at reduced rate
        // Pose scratch, `joint_count_` entries per character (current pose, and the outgoing pose for crossfades).
        std::vector<glm::vec3> translation_;
        std::vector<glm::quat> rotation_;
        std::vector<glm::vec3> scale_;
        std::vector<glm::vec3> scratch_translation_;
        std::vector<glm::quat> scratch_rotation_;
        std::vector<glm::vec3> scratch_scale_;
        std::vector<glm::mat4> palette_;
    };
} // namespace Animation
