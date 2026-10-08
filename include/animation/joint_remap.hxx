#pragma once

#include "animation/skeleton.hxx"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Animation {
    inline constexpr std::int32_t no_joint = -1;

    // Lower-cases, strips "mixamorig"/"mixamorig:" prefixes and every non-alphanumeric character, then maps
    // bone-naming aliases (upleg/thigh, forearm/lowerarm, spine1/chest, ...) to one canonical token.
    [[nodiscard]] auto canonical_joint_name(std::string_view name) -> std::string;

    struct JointRemap {
        std::vector<std::int32_t> source_to_target; // per source joint: target index or no_joint
        std::vector<std::int32_t> target_to_source; // per target joint: source index or no_joint
        std::size_t mapped_count{0};
    };

    // Matches source joints to target names by canonical name (a target is claimed by one source only).
    // Unmatched sources can then be dropped or re-parented by the caller.
    [[nodiscard]] auto remap_joints(Skeleton const &source, std::span<std::string const> target_names) -> JointRemap;

    // The names of the 14-joint Humanoid rig, in Humanoid::Joint order.
    [[nodiscard]] auto humanoid_joint_names() -> std::span<std::string const>;

    // Mixamo/Quaternius-style source onto the Humanoid rig: Hips->Pelvis, Spine->Spine, Spine2->Chest,
    // Head, LeftArm->UpperArmL, LeftForeArm->ForearmL, LeftUpLeg->ThighL, LeftLeg->ShinL, LeftFoot->FootL, ...
    [[nodiscard]] auto remap_to_humanoid(Skeleton const &source) -> JointRemap;
} // namespace Animation
