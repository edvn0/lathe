#pragma once

#include "animation/clip.hxx"
#include "animation/skeleton.hxx"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct SkinVertex {
    std::array<std::uint16_t, 4> joints{};
    std::array<std::uint16_t, 4> weights{};

    friend auto operator==(SkinVertex const &, SkinVertex const &) -> bool = default;
};
static_assert(sizeof(SkinVertex) == 16);

struct ImportedClip {
    std::string name;
    float duration{0.0F};
    Animation::Pose base;
    std::vector<Animation::Vec3Track> translations;
    std::vector<Animation::QuatTrack> rotations;
    std::vector<Animation::Vec3Track> scales;

    [[nodiscard]] auto make_clip() const -> std::unique_ptr<Animation::KeyframeClip>;
};

struct ModelAnimationData {
    Animation::Skeleton skeleton;
    std::vector<ImportedClip> clips;

    [[nodiscard]] auto find_clip(std::string_view name) const -> ImportedClip const *;
};

using AssetSkin = ModelAnimationData;
