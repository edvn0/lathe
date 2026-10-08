#pragma once

#include "animation/state_machine.hxx"

#include <memory>

namespace Animation::Humanoid {
    enum Joint : std::uint32_t {
        Pelvis,
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

    inline constexpr float walk_stride = 1.6F;
    inline constexpr float run_stride = 3.6F;

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
    [[nodiscard]] auto make_state_table(Rig const &rig) -> StateTable;
}
