#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include <glm/glm.hpp>

// CPU mirror of the Hi-Z occlusion test in assets/shaders/hiz_occlusion.slang and the pyramid layout built by
// assets/shaders/hiz_build.slang, so the conservativeness of the GPU test can be unit-tested
// (test/hiz_occlusion_test.cxx). Keep the two in lockstep. See docs/occlusion-culling.md.
//
// Depth convention: the projection is a standard [0, 1] one and reverse-Z comes from the viewport (minDepth 1,
// maxDepth 0), so stored depth = 1 - ndc.z: larger is nearer and 0 is the far clear. The pyramid keeps the
// farthest, i.e. the MIN, depth of each footprint. The viewport also flips Y: pixel row 0 is ndc.y = +1.
//
// Pyramid layout: logical level L is ceil(W / 2^(L+1)) x ceil(H / 2^(L+1)) texels, and texel t of level L covers depth
// pixels [t * 2^(L+1), (t + 1) * 2^(L+1)) clipped to the depth extent, so a pixel maps to a texel with a shift.
// Vulkan halves mip extents with floor, so the image itself is a power of two (hiz_image_extent()) and only the
// logical region of each level is written and read.

struct HizExtent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    auto operator==(HizExtent const &) const -> bool = default;
};

// Inclusive pixel bounds of a projected box, clamped to the depth extent, and its nearest stored depth.
struct HizPixelRect {
    std::int32_t x0 = 0;
    std::int32_t y0 = 0;
    std::int32_t x1 = 0;
    std::int32_t y1 = 0;
    float nearest_depth = 0.0F;
};

// Inclusive texel bounds within one pyramid level; at most 2 x 2 texels for the level select_hiz_level() picks.
struct HizTexelRect {
    std::int32_t x0 = 0;
    std::int32_t y0 = 0;
    std::int32_t x1 = 0;
    std::int32_t y1 = 0;
};

// The image's mip count is capped here, which covers depth extents up to 65536 pixels per side.
inline constexpr std::uint32_t hiz_max_mip_count = 16;

[[nodiscard]] constexpr auto hiz_level_extent(HizExtent depth, std::uint32_t level) noexcept -> HizExtent {
    auto const shift = level + 1U;

    if (shift >= 32U) {
        return HizExtent{.width = 1, .height = 1};
    }

    auto const divisor = std::uint32_t{1} << shift;

    return HizExtent{
            .width = std::max((depth.width + divisor - 1U) / divisor, 1U),
            .height = std::max((depth.height + divisor - 1U) / divisor, 1U),
    };
}

// ceil(log2(max(W, H))), at least 1: the last level is 1 x 1. Capped at hiz_max_mip_count.
[[nodiscard]] constexpr auto hiz_mip_count(HizExtent depth) noexcept -> std::uint32_t {
    auto const largest = std::max({depth.width, depth.height, 1U});
    auto const levels = static_cast<std::uint32_t>(std::bit_width(largest - 1U));

    return std::clamp(levels, 1U, hiz_max_mip_count);
}

// The image's base extent: level 0's logical extent rounded up to powers of two, so every floor-halved mip still
// holds its level's logical extent.
[[nodiscard]] constexpr auto hiz_image_extent(HizExtent depth) noexcept -> HizExtent {
    auto const level_zero = hiz_level_extent(depth, 0);

    return HizExtent{
            .width = std::bit_ceil(level_zero.width),
            .height = std::bit_ceil(level_zero.height),
    };
}

// Projects a world-space box with the view-projection the pyramid was built with. nullopt when the test cannot
// decide (a corner behind the eye or nearer than the near plane, or the box entirely off screen, which is the
// frustum test's job); callers then treat the box as visible. `guard_pixels` grows the rect on every side.
[[nodiscard]] inline auto project_aabb_to_hiz_rect(glm::mat4 const &view_projection, glm::vec3 world_min,
                                                   glm::vec3 world_max, HizExtent depth, float guard_pixels) noexcept
        -> std::optional<HizPixelRect> {
    if (depth.width == 0 || depth.height == 0) {
        return std::nullopt;
    }

    constexpr auto big = std::numeric_limits<float>::max();

    glm::vec2 ndc_min{big};
    glm::vec2 ndc_max{-big};
    float nearest = 0.0F;

    for (std::uint32_t corner = 0; corner < 8U; ++corner) {
        glm::vec3 const position{
                (corner & 1U) != 0U ? world_max.x : world_min.x,
                (corner & 2U) != 0U ? world_max.y : world_min.y,
                (corner & 4U) != 0U ? world_max.z : world_min.z,
        };

        glm::vec4 const clip = view_projection * glm::vec4(position, 1.0F);

        if (clip.w <= 0.0F || clip.z < 0.0F) {
            return std::nullopt;
        }

        glm::vec3 const ndc = glm::vec3(clip) / clip.w;

        ndc_min = glm::min(ndc_min, glm::vec2(ndc));
        ndc_max = glm::max(ndc_max, glm::vec2(ndc));
        nearest = std::max(nearest, 1.0F - ndc.z);
    }

    auto const width = static_cast<float>(depth.width);
    auto const height = static_cast<float>(depth.height);

    auto const x0 = ((ndc_min.x * 0.5F) + 0.5F) * width - guard_pixels;
    auto const x1 = ((ndc_max.x * 0.5F) + 0.5F) * width + guard_pixels;
    auto const y0 = (0.5F - (ndc_max.y * 0.5F)) * height - guard_pixels;
    auto const y1 = (0.5F - (ndc_min.y * 0.5F)) * height + guard_pixels;

    if (x1 < 0.0F || y1 < 0.0F || x0 >= width || y0 >= height) {
        return std::nullopt;
    }

    // Clamped as floats before the conversion, which would be undefined for out-of-range values.
    auto const to_pixel = [](float value, float extent) -> std::int32_t {
        return static_cast<std::int32_t>(std::clamp(std::floor(value), 0.0F, extent - 1.0F));
    };

    return HizPixelRect{
            .x0 = to_pixel(x0, width),
            .y0 = to_pixel(y0, height),
            .x1 = to_pixel(x1, width),
            .y1 = to_pixel(y1, height),
            .nearest_depth = nearest,
    };
}

// The finest level at which the rect spans at most 2 x 2 texels: with span < 2^(L+1), [x0, x1] crosses at most one
// texel boundary of level L. The top level is 1 x 1, so the clamp keeps that true.
[[nodiscard]] constexpr auto select_hiz_level(HizPixelRect const &rect, std::uint32_t mip_count) noexcept
        -> std::uint32_t {
    auto const span = static_cast<std::uint32_t>(std::max({rect.x1 - rect.x0, rect.y1 - rect.y0, 0}));
    auto const level = span == 0U ? 0U : static_cast<std::uint32_t>(std::bit_width(span)) - 1U;

    return std::min(level, std::max(mip_count, 1U) - 1U);
}

[[nodiscard]] constexpr auto hiz_texel_rect(HizPixelRect const &rect, std::uint32_t level) noexcept -> HizTexelRect {
    auto const shift = static_cast<std::int32_t>(level + 1U);

    return HizTexelRect{
            .x0 = rect.x0 >> shift,
            .y0 = rect.y0 >> shift,
            .x1 = rect.x1 >> shift,
            .y1 = rect.y1 >> shift,
    };
}

// True when the box's nearest depth is strictly behind the farthest depth of everything in its footprint, by more
// than `epsilon`. Ties count as visible.
[[nodiscard]] constexpr auto depth_occluded(float nearest_depth, float hiz_farthest, float epsilon) noexcept -> bool {
    return nearest_depth + epsilon < hiz_farthest;
}
