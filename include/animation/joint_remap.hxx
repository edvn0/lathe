#pragma once

#include "animation/skeleton.hxx"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Animation {
    inline constexpr std::int32_t no_joint = -1;

    [[nodiscard]] auto canonical_joint_name(std::string_view name) -> std::string;

    struct JointRemap {
        std::vector<std::int32_t> source_to_target;
        std::vector<std::int32_t> target_to_source;
        std::size_t mapped_count{0};
    };

    [[nodiscard]] auto remap_joints(Skeleton const &source, std::span<std::string const> target_names) -> JointRemap;

    [[nodiscard]] auto humanoid_joint_names() -> std::span<std::string const>;

    [[nodiscard]] auto remap_to_humanoid(Skeleton const &source) -> JointRemap;
}
