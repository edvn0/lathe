#pragma once

#include "animation/clip.hxx"
#include "animation/skeleton.hxx"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// CPU-side glTF skin + animation import results. Pure data (no Vulkan); see docs/gpu-skinning-design.md.

// Per-vertex skinning stream, parallel to a primitive's vertices (same count and order, also after tangent
// generation reorders them). Kept separate from ModelVertex so the render vertex layout is unchanged.
// `joints` index Animation::Skeleton joints (the importer's topological order, NOT the glTF skin order).
// `weights` are unorm16 and sum to exactly 65535; unused influences are 0 with joint 0.
struct SkinVertex {
    std::array<std::uint16_t, 4> joints{};
    std::array<std::uint16_t, 4> weights{};

    friend auto operator==(SkinVertex const &, SkinVertex const &) -> bool = default;
};
static_assert(sizeof(SkinVertex) == 16);

// Imported animation: keyframe tracks per joint. STEP channels are expanded into hold keys (KeyframeClip only
// interpolates), CUBICSPLINE channels are approximated by their key values (tangents dropped).
struct ImportedClip {
    std::string name;
    float duration{0.0F};
    Animation::Pose base; // bind pose; joints without a track keep it
    std::vector<Animation::Vec3Track> translations;
    std::vector<Animation::QuatTrack> rotations;
    std::vector<Animation::Vec3Track> scales;

    [[nodiscard]] auto make_clip() const -> std::unique_ptr<Animation::KeyframeClip>;
};

// Skeleton (joints parent-before-child) plus every animation of the first glTF skin.
struct ModelAnimationData {
    Animation::Skeleton skeleton;
    std::vector<ImportedClip> clips;

    [[nodiscard]] auto find_clip(std::string_view name) const -> ImportedClip const *; // exact or "|name" suffix
};

using AssetSkin = ModelAnimationData;
