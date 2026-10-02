#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "rendering/hiz_occlusion.hxx"

namespace {

    // One pyramid level or the depth buffer itself, row-major.
    struct DepthLevel {
        HizExtent extent{};
        std::vector<float> texels;

        [[nodiscard]] auto at(std::int32_t x, std::int32_t y) const -> float {
            return texels[(static_cast<std::size_t>(y) * extent.width) + static_cast<std::size_t>(x)];
        }
    };

    // The reduction hiz_build.slang runs: each texel is the MIN (the farthest, under reverse-Z) of its 2 x 2 children
    // inside the source's logical extent, with 1.0 (the nearest possible depth) as the neutral value.
    [[nodiscard]] auto build_hiz_reference(DepthLevel const &depth) -> std::vector<DepthLevel> {
        std::vector<DepthLevel> levels;
        auto const mip_count = hiz_mip_count(depth.extent);

        for (std::uint32_t level = 0; level < mip_count; ++level) {
            auto const &source = level == 0 ? depth : levels.back();
            auto const extent = hiz_level_extent(depth.extent, level);

            CHECK(extent == hiz_level_extent(source.extent, 0));

            DepthLevel destination{
                    .extent = extent,
                    .texels = std::vector<float>(static_cast<std::size_t>(extent.width) * extent.height),
            };

            for (std::int32_t y = 0; std::cmp_less(y, extent.height); ++y) {
                for (std::int32_t x = 0; std::cmp_less(x, extent.width); ++x) {
                    float farthest = 1.0F;

                    for (std::int32_t dy = 0; dy < 2; ++dy) {
                        for (std::int32_t dx = 0; dx < 2; ++dx) {
                            auto const source_x = (x * 2) + dx;
                            auto const source_y = (y * 2) + dy;

                            if (std::cmp_less(source_x, source.extent.width) &&
                                std::cmp_less(source_y, source.extent.height)) {
                                farthest = std::min(farthest, source.at(source_x, source_y));
                            }
                        }
                    }

                    destination.texels[(static_cast<std::size_t>(y) * extent.width) + static_cast<std::size_t>(x)] =
                            farthest;
                }
            }

            levels.push_back(std::move(destination));
        }

        return levels;
    }

    // hiz_occlusion.slang's aabb_occluded(), on a CPU pyramid.
    struct OcclusionQuery {
        glm::mat4 view_projection{1.0F};
        glm::vec3 world_min{0.0F};
        glm::vec3 world_max{0.0F};
        float guard_pixels = 1.0F;
        float depth_epsilon = 1e-6F;
    };

    [[nodiscard]] auto aabb_occluded_reference(std::vector<DepthLevel> const &pyramid, HizExtent depth,
                                               OcclusionQuery const &query) -> bool {
        auto const rect = project_aabb_to_hiz_rect(query.view_projection, query.world_min, query.world_max, depth,
                                                   query.guard_pixels);

        if (!rect) {
            return false;
        }

        auto const level = select_hiz_level(*rect, static_cast<std::uint32_t>(pyramid.size()));
        auto const texels = hiz_texel_rect(*rect, level);
        auto const &hiz = pyramid[level];

        auto const farthest = std::min({hiz.at(texels.x0, texels.y0), hiz.at(texels.x1, texels.y0),
                                        hiz.at(texels.x0, texels.y1), hiz.at(texels.x1, texels.y1)});

        return depth_occluded(rect->nearest_depth, farthest, query.depth_epsilon);
    }

    [[nodiscard]] auto random_float(std::mt19937 &engine, float low, float high) -> float {
        return std::uniform_real_distribution<float>{low, high}(engine);
    }

    [[nodiscard]] auto random_uint(std::mt19937 &engine, std::uint32_t low, std::uint32_t high) -> std::uint32_t {
        return std::uniform_int_distribution<std::uint32_t>{low, high}(engine);
    }

    // A background depth with a few nearer and farther rectangles, so boxes behind it are often occluded.
    [[nodiscard]] auto random_depth(std::mt19937 &engine, HizExtent extent) -> DepthLevel {
        DepthLevel depth{
                .extent = extent,
                .texels = std::vector<float>(static_cast<std::size_t>(extent.width) * extent.height,
                                             random_float(engine, 0.0F, 0.6F)),
        };

        auto const rectangle_count = random_uint(engine, 0, 5);

        for (std::uint32_t rectangle = 0; rectangle < rectangle_count; ++rectangle) {
            auto const x0 = random_uint(engine, 0, extent.width - 1);
            auto const y0 = random_uint(engine, 0, extent.height - 1);
            auto const x1 = random_uint(engine, x0, extent.width - 1);
            auto const y1 = random_uint(engine, y0, extent.height - 1);
            auto const value = random_float(engine, 0.0F, 1.0F);

            for (auto y = y0; y <= y1; ++y) {
                for (auto x = x0; x <= x1; ++x) {
                    depth.texels[(static_cast<std::size_t>(y) * extent.width) + x] = value;
                }
            }
        }

        return depth;
    }

    constexpr std::array<HizExtent, 7> extent_table{
            HizExtent{.width = 1, .height = 1},       HizExtent{.width = 2, .height = 1},
            HizExtent{.width = 7, .height = 1},       HizExtent{.width = 3, .height = 3},
            HizExtent{.width = 1366, .height = 768},  HizExtent{.width = 1920, .height = 1080},
            HizExtent{.width = 2560, .height = 1440},
    };

} // namespace

TEST_CASE("Hi-Z level extents halve with ceil down to 1 x 1") {
    for (auto const depth: extent_table) {
        CAPTURE(depth.width);
        CAPTURE(depth.height);

        auto const mip_count = hiz_mip_count(depth);
        auto const image = hiz_image_extent(depth);

        REQUIRE(mip_count >= 1);
        CHECK(hiz_level_extent(depth, mip_count - 1) == HizExtent{.width = 1, .height = 1});

        // A full chain on the image has at least mip_count levels.
        CHECK(static_cast<std::uint32_t>(std::bit_width(std::max(image.width, image.height))) >= mip_count);

        for (std::uint32_t level = 0; level < mip_count; ++level) {
            CAPTURE(level);

            auto const logical = hiz_level_extent(depth, level);
            auto const divisor = std::pow(2.0, static_cast<double>(level + 1));

            CHECK(logical.width == static_cast<std::uint32_t>(std::ceil(depth.width / divisor)));
            CHECK(logical.height == static_cast<std::uint32_t>(std::ceil(depth.height / divisor)));

            // Vulkan's floor-halved mip of the image holds the logical level.
            CHECK(std::max(image.width >> level, 1U) >= logical.width);
            CHECK(std::max(image.height >> level, 1U) >= logical.height);
        }

        // One level fewer would leave the top level larger than 1 x 1.
        if (mip_count > 1) {
            CHECK_FALSE(hiz_level_extent(depth, mip_count - 2) == HizExtent{.width = 1, .height = 1});
        }
    }

    CHECK(hiz_mip_count(HizExtent{.width = 1920, .height = 1080}) == 11);
    CHECK(hiz_level_extent(HizExtent{.width = 1920, .height = 1080}, 0) == HizExtent{.width = 960, .height = 540});
    CHECK(hiz_image_extent(HizExtent{.width = 1920, .height = 1080}) == HizExtent{.width = 1024, .height = 1024});
}

TEST_CASE("the selected Hi-Z level covers any pixel span with at most 2 x 2 texels") {
    constexpr HizExtent depth{.width = 256, .height = 256};
    auto const mip_count = hiz_mip_count(depth);

    for (std::int32_t x0 = 0; x0 < 256; ++x0) {
        for (std::int32_t x1 = x0; x1 < 256; ++x1) {
            HizPixelRect const rect{.x0 = x0, .y0 = 0, .x1 = x1, .y1 = 0};

            auto const level = select_hiz_level(rect, mip_count);
            auto const texels = hiz_texel_rect(rect, level);
            auto const texel_size = std::int32_t{1} << (level + 1U);
            auto const extent = hiz_level_extent(depth, level);

            bool const covered = texels.x1 - texels.x0 <= 1 && texels.x0 * texel_size <= x0 &&
                                 (texels.x1 + 1) * texel_size > x1 && std::cmp_less(texels.x1, extent.width) &&
                                 texels.y0 == 0 && texels.y1 == 0;

            if (!covered) {
                CAPTURE(x0);
                CAPTURE(x1);
                CAPTURE(level);
                CHECK(covered);
            }
        }
    }

    // Wider than every level but the top: the top level is a single texel.
    CHECK(select_hiz_level(HizPixelRect{.x0 = 0, .y0 = 0, .x1 = 255, .y1 = 255}, mip_count) == mip_count - 1);
    CHECK(hiz_level_extent(depth, mip_count - 1) == HizExtent{.width = 1, .height = 1});
}

TEST_CASE("each reference Hi-Z texel is the farthest depth of its pixel footprint") {
    std::mt19937 engine{1234U};

    for (auto const depth_extent: {HizExtent{.width = 1, .height = 1}, HizExtent{.width = 7, .height = 5},
                                   HizExtent{.width = 33, .height = 17}, HizExtent{.width = 100, .height = 61},
                                   HizExtent{.width = 257, .height = 129}}) {
        CAPTURE(depth_extent.width);
        CAPTURE(depth_extent.height);

        DepthLevel depth{
                .extent = depth_extent,
                .texels = std::vector<float>(static_cast<std::size_t>(depth_extent.width) * depth_extent.height),
        };

        for (auto &texel: depth.texels) {
            texel = random_float(engine, 0.0F, 1.0F);
        }

        auto const pyramid = build_hiz_reference(depth);
        REQUIRE(pyramid.size() == hiz_mip_count(depth_extent));

        std::uint32_t mismatches = 0;

        for (std::uint32_t level = 0; level < pyramid.size(); ++level) {
            auto const &hiz = pyramid[level];
            auto const texel_size = std::int32_t{1} << (level + 1U);

            for (std::int32_t ty = 0; std::cmp_less(ty, hiz.extent.height); ++ty) {
                for (std::int32_t tx = 0; std::cmp_less(tx, hiz.extent.width); ++tx) {
                    float farthest = 1.0F;

                    for (auto y = ty * texel_size; y < (ty + 1) * texel_size && std::cmp_less(y, depth_extent.height);
                         ++y) {
                        for (auto x = tx * texel_size;
                             x < (tx + 1) * texel_size && std::cmp_less(x, depth_extent.width); ++x) {
                            farthest = std::min(farthest, depth.at(x, y));
                        }
                    }

                    mismatches += hiz.at(tx, ty) == farthest ? 0U : 1U;
                }
            }
        }

        CHECK(mismatches == 0);
    }
}

TEST_CASE("an occluded box has no pixel in its projection at or behind its nearest depth") {
    std::mt19937 engine{20260101U};

    std::uint32_t occluded_count = 0;
    std::uint32_t visible_count = 0;
    std::uint32_t violations = 0;

    HizExtent extent{};
    DepthLevel depth{};
    std::vector<DepthLevel> pyramid;

    for (std::uint32_t iteration = 0; iteration < 10'000; ++iteration) {
        if (iteration % 100 == 0) {
            extent = HizExtent{.width = random_uint(engine, 1, 96), .height = random_uint(engine, 1, 96)};
            depth = random_depth(engine, extent);
            pyramid = build_hiz_reference(depth);
        }

        auto const aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        auto const projection = glm::perspectiveLH_ZO(glm::radians(random_float(engine, 30.0F, 90.0F)), aspect,
                                                      random_float(engine, 0.05F, 0.5F), 200.0F);

        glm::vec3 const eye{random_float(engine, -2.0F, 2.0F), random_float(engine, -2.0F, 2.0F),
                            random_float(engine, -2.0F, 2.0F)};
        glm::vec3 const target = eye + glm::vec3{random_float(engine, -0.3F, 0.3F), random_float(engine, -0.3F, 0.3F),
                                                 1.0F};
        auto const view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F});
        auto const view_projection = projection * view;

        glm::vec3 const centre = eye + glm::vec3{random_float(engine, -6.0F, 6.0F), random_float(engine, -6.0F, 6.0F),
                                                 random_float(engine, 0.2F, 60.0F)};
        glm::vec3 const half_size{random_float(engine, 0.01F, 3.0F), random_float(engine, 0.01F, 3.0F),
                                  random_float(engine, 0.01F, 3.0F)};

        OcclusionQuery const query{
                .view_projection = view_projection,
                .world_min = centre - half_size,
                .world_max = centre + half_size,
        };

        if (!aabb_occluded_reference(pyramid, extent, query)) {
            ++visible_count;
            continue;
        }

        ++occluded_count;

        // The exact screen rectangle and nearest depth, in double precision. Every pixel whose square touches the
        // rectangle could receive a sample of the box.
        glm::dvec2 screen_min{1e300};
        glm::dvec2 screen_max{-1e300};
        double nearest = 0.0;

        for (std::uint32_t corner = 0; corner < 8; ++corner) {
            glm::dvec3 const position{
                    (corner & 1U) != 0U ? query.world_max.x : query.world_min.x,
                    (corner & 2U) != 0U ? query.world_max.y : query.world_min.y,
                    (corner & 4U) != 0U ? query.world_max.z : query.world_min.z,
            };

            glm::dvec4 const clip = glm::dmat4(view_projection) * glm::dvec4(position, 1.0);
            glm::dvec3 const ndc = glm::dvec3(clip) / clip.w;

            glm::dvec2 const screen{((ndc.x * 0.5) + 0.5) * extent.width, (0.5 - (ndc.y * 0.5)) * extent.height};
            screen_min = glm::min(screen_min, screen);
            screen_max = glm::max(screen_max, screen);
            nearest = std::max(nearest, 1.0 - ndc.z);
        }

        auto const first_x = std::max(static_cast<std::int64_t>(std::floor(screen_min.x)), std::int64_t{0});
        auto const first_y = std::max(static_cast<std::int64_t>(std::floor(screen_min.y)), std::int64_t{0});
        auto const last_x = std::min(static_cast<std::int64_t>(std::floor(screen_max.x)),
                                     static_cast<std::int64_t>(extent.width) - 1);
        auto const last_y = std::min(static_cast<std::int64_t>(std::floor(screen_max.y)),
                                     static_cast<std::int64_t>(extent.height) - 1);

        for (auto y = first_y; y <= last_y; ++y) {
            for (auto x = first_x; x <= last_x; ++x) {
                if (static_cast<double>(depth.at(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y))) <=
                    nearest) {
                    ++violations;
                }
            }
        }
    }

    CHECK(violations == 0);

    // Otherwise the property above was tested on nothing.
    CHECK(occluded_count > 500);
    CHECK(visible_count > 500);
}

TEST_CASE("boxes crossing the near plane or behind the eye are never occluded") {
    auto const projection = glm::perspectiveLH_ZO(glm::radians(60.0F), 1.0F, 0.1F, 100.0F);
    auto const view = glm::lookAtLH(glm::vec3{0.0F}, glm::vec3{0.0F, 0.0F, 1.0F}, glm::vec3{0.0F, 1.0F, 0.0F});
    auto const view_projection = projection * view;
    constexpr HizExtent depth{.width = 64, .height = 64};

    // Straddles the near plane.
    CHECK_FALSE(project_aabb_to_hiz_rect(view_projection, glm::vec3{-0.5F, -0.5F, 0.05F}, glm::vec3{0.5F, 0.5F, 1.0F},
                                         depth, 1.0F));

    // Contains the eye.
    CHECK_FALSE(project_aabb_to_hiz_rect(view_projection, glm::vec3{-1.0F}, glm::vec3{1.0F}, depth, 1.0F));

    // Entirely behind the eye.
    CHECK_FALSE(project_aabb_to_hiz_rect(view_projection, glm::vec3{-1.0F, -1.0F, -5.0F},
                                         glm::vec3{1.0F, 1.0F, -1.0F}, depth, 1.0F));

    // In front: decided, and nearer than a far one.
    auto const near_box =
            project_aabb_to_hiz_rect(view_projection, glm::vec3{-0.1F, -0.1F, 1.0F}, glm::vec3{0.1F, 0.1F, 1.2F},
                                     depth, 1.0F);
    auto const far_box = project_aabb_to_hiz_rect(view_projection, glm::vec3{-0.1F, -0.1F, 50.0F},
                                                  glm::vec3{0.1F, 0.1F, 50.2F}, depth, 1.0F);
    REQUIRE(near_box);
    REQUIRE(far_box);
    CHECK(near_box->nearest_depth > far_box->nearest_depth);

    // Off screen to the side: left to the frustum test.
    CHECK_FALSE(project_aabb_to_hiz_rect(view_projection, glm::vec3{40.0F, -0.1F, 5.0F},
                                         glm::vec3{41.0F, 0.1F, 6.0F}, depth, 1.0F));
}

TEST_CASE("NDC y = +1 maps to pixel row 0 and x maps left to right") {
    // Identity view-projection: NDC is the world position.
    constexpr HizExtent depth{.width = 100, .height = 100};

    auto const top_left = project_aabb_to_hiz_rect(glm::mat4{1.0F}, glm::vec3{-1.0F, 0.9F, 0.5F},
                                                   glm::vec3{-0.9F, 1.0F, 0.6F}, depth, 0.0F);
    REQUIRE(top_left);
    CHECK(top_left->x0 == 0);
    CHECK(top_left->x1 == 5);
    CHECK(top_left->y0 == 0);
    CHECK(top_left->y1 == 5);
    CHECK(top_left->nearest_depth == doctest::Approx(0.5F));

    auto const bottom_right = project_aabb_to_hiz_rect(glm::mat4{1.0F}, glm::vec3{0.9F, -1.0F, 0.25F},
                                                       glm::vec3{1.0F, -0.9F, 0.75F}, depth, 0.0F);
    REQUIRE(bottom_right);
    CHECK(bottom_right->x0 == 95);
    CHECK(bottom_right->x1 == 99);
    CHECK(bottom_right->y0 == 95);
    CHECK(bottom_right->y1 == 99);
    CHECK(bottom_right->nearest_depth == doctest::Approx(0.75F));
}

TEST_CASE("a sub-pixel box reads level 0 and at most 2 x 2 texels") {
    constexpr HizExtent depth{.width = 64, .height = 64};

    auto const rect = project_aabb_to_hiz_rect(glm::mat4{1.0F}, glm::vec3{0.001F, 0.001F, 0.5F},
                                               glm::vec3{0.002F, 0.002F, 0.5F}, depth, 0.0F);
    REQUIRE(rect);
    CHECK(rect->x0 == rect->x1);
    CHECK(rect->y0 == rect->y1);

    auto const level = select_hiz_level(*rect, hiz_mip_count(depth));
    CHECK(level == 0);

    auto const texels = hiz_texel_rect(*rect, level);
    CHECK(texels.x1 - texels.x0 <= 1);
    CHECK(texels.y1 - texels.y0 <= 1);

    // With the default 1-pixel guard the rect is 3 pixels wide, still level 1 at most.
    auto const guarded = project_aabb_to_hiz_rect(glm::mat4{1.0F}, glm::vec3{0.001F, 0.001F, 0.5F},
                                                  glm::vec3{0.002F, 0.002F, 0.5F}, depth, 1.0F);
    REQUIRE(guarded);
    CHECK(select_hiz_level(*guarded, hiz_mip_count(depth)) <= 1);
}

TEST_CASE("occlusion needs the box strictly behind the footprint by more than epsilon") {
    CHECK_FALSE(depth_occluded(0.5F, 0.75F, 0.25F));
    CHECK(depth_occluded(0.5F, 0.875F, 0.25F));
    CHECK_FALSE(depth_occluded(0.5F, 0.5F, 0.0F));
    CHECK(depth_occluded(0.25F, 0.5F, 0.0F));

    // Nearer than the footprint is never occluded.
    CHECK_FALSE(depth_occluded(0.75F, 0.5F, 0.0F));
}

TEST_CASE("a box behind a full-screen wall is occluded and one in front of it is not") {
    constexpr HizExtent extent{.width = 40, .height = 30};
    auto const projection = glm::perspectiveLH_ZO(glm::radians(60.0F), 40.0F / 30.0F, 0.1F, 100.0F);
    auto const view = glm::lookAtLH(glm::vec3{0.0F}, glm::vec3{0.0F, 0.0F, 1.0F}, glm::vec3{0.0F, 1.0F, 0.0F});
    auto const view_projection = projection * view;

    // A wall at z = 5 covering the screen.
    auto const wall = glm::vec4(view_projection * glm::vec4(0.0F, 0.0F, 5.0F, 1.0F));
    auto const wall_depth = 1.0F - (wall.z / wall.w);

    DepthLevel const depth{
            .extent = extent,
            .texels = std::vector<float>(static_cast<std::size_t>(extent.width) * extent.height, wall_depth),
    };
    auto const pyramid = build_hiz_reference(depth);

    CHECK(aabb_occluded_reference(pyramid, extent,
                                  OcclusionQuery{.view_projection = view_projection,
                                                 .world_min = glm::vec3{-0.5F, -0.5F, 8.0F},
                                                 .world_max = glm::vec3{0.5F, 0.5F, 9.0F}}));

    CHECK_FALSE(aabb_occluded_reference(pyramid, extent,
                                        OcclusionQuery{.view_projection = view_projection,
                                                       .world_min = glm::vec3{-0.5F, -0.5F, 3.0F},
                                                       .world_max = glm::vec3{0.5F, 0.5F, 4.0F}}));

    // Straddling the wall: its nearest point is in front.
    CHECK_FALSE(aabb_occluded_reference(pyramid, extent,
                                        OcclusionQuery{.view_projection = view_projection,
                                                       .world_min = glm::vec3{-0.5F, -0.5F, 4.5F},
                                                       .world_max = glm::vec3{0.5F, 0.5F, 9.0F}}));
}
