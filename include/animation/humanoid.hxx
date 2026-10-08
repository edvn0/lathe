#pragma once

#include "animation/state_machine.hxx"

#include <memory>

namespace Animation::Humanoid {
    // 14-joint humanoid, Y-up, about 1.8 m tall, bind pose = standing with arms hanging down.
    // The character faces +Z; its left side is +X. Rotations use glm's right-handed convention.
    enum Joint : std::uint32_t {
        Pelvis, // root, at hip height
        Spine,
        Chest,
        Head,
        UpperArmL,
        ForearmL,
        UpperArmR,
        ForearmR,
        ThighL,
        ShinL,
        FootL,
        ThighR,
        ShinR,
        FootR,
        JointCount,
    };

    // Gait numbers shared by the clips and the state table.
    inline constexpr float walk_stride = 1.6F; // metres per full cycle (two steps)
    inline constexpr float run_stride = 3.6F;

    // Skeleton plus the procedural clips. Clips are referenced by pointer from the state table, so the rig is
    // heap-allocated and never moves.
    struct Rig {
        Rig(Skeleton skeleton_in);
        Rig(Rig const &) = delete;
        auto operator=(Rig const &) -> Rig & = delete;
        Rig(Rig &&) = delete;
        auto operator=(Rig &&) -> Rig & = delete;
        ~Rig() = default;

        Skeleton skeleton;
        std::unique_ptr<ProceduralClip> idle;
        std::unique_ptr<ProceduralClip> walk;
        std::unique_ptr<ProceduralClip> run;
        std::unique_ptr<ProceduralClip> jump_rise;
        std::unique_ptr<ProceduralClip> jump_apex;
        std::unique_ptr<ProceduralClip> jump_fall;
        std::unique_ptr<ProceduralClip> lying_down;
        std::unique_ptr<ProceduralClip> prone;
        std::unique_ptr<ProceduralClip> getting_up;
    };

    [[nodiscard]] auto make_skeleton() -> Skeleton;
    [[nodiscard]] auto make_rig() -> std::unique_ptr<Rig>;
    // Table wiring every state to the rig's clips; the rig must outlive the returned machine.
    [[nodiscard]] auto make_state_table(Rig const &rig) -> StateTable;
} // namespace Animation::Humanoid
