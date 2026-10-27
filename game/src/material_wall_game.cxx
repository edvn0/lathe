#include "material_wall_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <bit>
#include <format>
#include <functional>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {
    constexpr std::size_t shot_count = 6;
    constexpr float spacing = 2.2F;
    constexpr float sphere_scale = 1.0F;

    constexpr std::array<glm::vec3, 8> palette{{
            {0.90F, 0.12F, 0.10F},
            {0.95F, 0.55F, 0.10F},
            {0.95F, 0.85F, 0.15F},
            {0.15F, 0.75F, 0.25F},
            {0.10F, 0.70F, 0.80F},
            {0.15F, 0.30F, 0.90F},
            {0.60F, 0.20F, 0.85F},
            {0.90F, 0.90F, 0.90F},
    }};

    using Texel = glm::vec4;

    auto hash2(glm::ivec2 cell, std::uint32_t seed) -> float {
        auto h = static_cast<std::uint32_t>(cell.x) * 0x9E3779B1U ^ static_cast<std::uint32_t>(cell.y) * 0x85EBCA77U ^
                 seed * 0xC2B2AE3DU;
        h ^= h >> 15U;
        h *= 0x2C1B3C6DU;
        h ^= h >> 12U;
        h *= 0x297A2D39U;
        h ^= h >> 15U;
        return static_cast<float>(h & 0xFFFFFFU) / static_cast<float>(0x1000000U);
    }

    auto value_noise(glm::vec2 p, std::uint32_t seed) -> float {
        auto const cell = glm::ivec2{glm::floor(p)};
        auto const f = p - glm::floor(p);
        auto const u = f * f * (3.0F - 2.0F * f);
        auto const a = hash2(cell, seed);
        auto const b = hash2(cell + glm::ivec2{1, 0}, seed);
        auto const c = hash2(cell + glm::ivec2{0, 1}, seed);
        auto const d = hash2(cell + glm::ivec2{1, 1}, seed);
        return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
    }

    auto fbm(glm::vec2 p, std::uint32_t seed) -> float {
        float sum = 0.0F;
        float amplitude = 0.5F;
        for (int octave = 0; octave < 5; ++octave) {
            sum += amplitude * value_noise(p, seed + static_cast<std::uint32_t>(octave));
            p *= 2.03F;
            amplitude *= 0.5F;
        }
        return sum;
    }

    // Rasterises `shade(uv)` (uv in [0,1]^2, values already in the texture's encoding) with a box-filtered mip chain.
    auto upload_texture(Renderer &renderer, std::uint32_t size, VkFormat format, char const *name,
                        std::function<Texel(glm::vec2)> const &shade) -> ImageHandle {
        auto const mip_levels = static_cast<std::uint32_t>(std::bit_width(size));

        std::vector<Texel> level(static_cast<std::size_t>(size) * size);
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                level[static_cast<std::size_t>(y) * size + x] =
                        shade({(static_cast<float>(x) + 0.5F) / static_cast<float>(size),
                               (static_cast<float>(y) + 0.5F) / static_cast<float>(size)});
            }
        }

        std::vector<std::byte> bytes;
        auto width = size;
        for (std::uint32_t mip = 0; mip < mip_levels; ++mip) {
            for (auto const &texel: level) {
                auto const packed = glm::clamp(texel, 0.0F, 1.0F) * 255.0F + 0.5F;
                for (int channel = 0; channel < 4; ++channel) {
                    bytes.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(packed[channel])));
                }
            }

            if (width == 1) {
                break;
            }

            auto const next = width / 2;
            std::vector<Texel> smaller(static_cast<std::size_t>(next) * next);
            for (std::uint32_t y = 0; y < next; ++y) {
                for (std::uint32_t x = 0; x < next; ++x) {
                    auto const at = [&](std::uint32_t dx, std::uint32_t dy) {
                        return level[static_cast<std::size_t>(y * 2 + dy) * width + x * 2 + dx];
                    };
                    smaller[static_cast<std::size_t>(y) * next + x] = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25F;
                }
            }
            level = std::move(smaller);
            width = next;
        }

        auto image = renderer.image_storage().create_image(
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = size, .height = size, .depth = 1},
                        .format = format,
                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .mip_levels = mip_levels,
                        .array_layers = 1,
                        .debug_name = name,
                },
                bytes, ImageMipSource::provided);

        if (!image) {
            error("[MaterialWall] Could not create texture {}", name);
            return renderer.image_storage().white();
        }
        return *image;
    }

    // Height field -> tangent-space normal (+Y up in the image, matching the shader's tangent frame: x along the
    // decal's local X, y along its local Z).
    auto normal_from_height(std::function<float(glm::vec2)> const &height, glm::vec2 uv, float strength) -> Texel {
        constexpr float step = 1.0F / 256.0F;
        auto const dx = height(uv + glm::vec2{step, 0.0F}) - height(uv - glm::vec2{step, 0.0F});
        auto const dy = height(uv + glm::vec2{0.0F, step}) - height(uv - glm::vec2{0.0F, step});
        auto const n = glm::normalize(glm::vec3{-dx * strength, -dy * strength, 1.0F});
        return {n * 0.5F + 0.5F, 1.0F};
    }

    // Signed distance (negative inside) to a wobbly blob with a few satellite droplets.
    auto splat_distance(glm::vec2 uv) -> float {
        auto const p = uv - 0.5F;
        auto const angle = std::atan2(p.y, p.x);
        auto const wobble = fbm(glm::vec2{std::cos(angle), std::sin(angle)} * 2.2F + 7.0F, 3);
        auto const arms = 0.07F * std::sin(angle * 5.0F + 1.3F) + 0.05F * std::sin(angle * 9.0F);
        auto distance = glm::length(p) - (0.20F + 0.20F * wobble + arms);

        constexpr std::array<glm::vec3, 9> droplets{{{0.30F, 0.12F, 0.035F}, {-0.34F, 0.08F, 0.03F},
                                                     {0.1F, -0.36F, 0.04F}, {-0.2F, -0.3F, 0.025F},
                                                     {0.38F, -0.2F, 0.02F}, {-0.08F, 0.38F, 0.03F},
                                                     {-0.4F, -0.1F, 0.018F}, {0.22F, 0.34F, 0.02F},
                                                     {0.0F, -0.45F, 0.015F}}};
        for (auto const &droplet: droplets) {
            distance = std::min(distance, glm::length(p - glm::vec2{droplet.x, droplet.y}) - droplet.z);
        }
        return distance;
    }

    auto splat_thickness(glm::vec2 uv) -> float {
        auto const inside = glm::clamp(-splat_distance(uv) * 14.0F, 0.0F, 1.0F);
        return inside * (0.55F + 0.45F * fbm(uv * 14.0F, 11));
    }

    // Rounded-box signed distance, half-size b, corner radius r.
    auto rounded_box(glm::vec2 p, glm::vec2 b, float r) -> float {
        auto const q = glm::abs(p) - b + r;
        return glm::length(glm::max(q, 0.0F)) + std::min(std::max(q.x, q.y), 0.0F) - r;
    }

    auto panel_height(glm::vec2 uv) -> float {
        auto const p = uv - 0.5F;
        auto const bevel = glm::clamp(-rounded_box(p, {0.46F, 0.46F}, 0.05F) * 40.0F, 0.0F, 1.0F);
        float rivets = 0.0F;
        for (auto const sx: {-1.0F, 1.0F}) {
            for (auto const sy: {-1.0F, 1.0F}) {
                auto const d = glm::length(p - glm::vec2{sx, sy} * 0.40F);
                rivets = std::max(rivets, glm::clamp(1.0F - d / 0.03F, 0.0F, 1.0F));
            }
        }
        return 0.4F * bevel + 1.2F * std::sqrt(rivets) + 0.05F * fbm(uv * 40.0F, 5);
    }

    auto stripe_mask(glm::vec2 uv) -> float {
        auto const t = (uv.x + uv.y) * 7.0F;
        return glm::smoothstep(0.45F, 0.55F, std::abs(t - std::floor(t) - 0.5F) * 2.0F) ;
    }
}

auto MaterialWallGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    scene.environment.source = EnvironmentSource::procedural_sky;
    scene.environment.draw_skybox = true;
    scene.environment.sun.elevation_degrees = 38.0F;
    scene.environment.sun.intensity = 4.0F;
    scene.environment.exposure_ev = 0.5F;
    scene.environment.specular_intensity = 1.5F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    auto const make_material = [&](glm::vec3 const &colour, float metallic, float roughness,
                                   glm::vec3 const &emissive = glm::vec3{0.0F},
                                   float emissive_strength = 1.0F) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{colour, 1.0F},
                .emissive_factor = emissive,
                .emissive_strength = emissive_strength,
                .metallic_factor = metallic,
                .roughness_factor = roughness,
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = glm::length(emissive) > 0.0F ? images.white() : images.emissive(),
                .sampler = samplers.linear_repeat(),
                .double_sided = true,
        });
        if (!material) {
            error("[MaterialWall] Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        return *material;
    };

    auto const add_prop = [&](std::string const &name, ModelHandle model, glm::vec3 const &position,
                              glm::vec3 const &scale, MaterialHandle material) {
        auto const prop = Entity{&scene, "{}", name};
        prop.emplace<Components::Transform>(Components::Transform{.position = position, .scale = scale});
        prop.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            prop.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
    };

    // Shared backdrop and floor.
    auto const backdrop = make_material({0.45F, 0.45F, 0.47F}, 0.0F, 0.9F);
    add_prop("floor", engine_models.cube, {0.0F, -0.25F, 4.0F}, {40.0F, 0.25F, 20.0F}, backdrop);
    add_prop("backdrop", engine_models.cube, {0.0F, 7.5F, -1.5F}, {40.0F, 16.0F, 0.5F}, backdrop);

    // Key lights in front of the wall so highlights (and so roughness) read on every sphere.
    constexpr std::array<glm::vec3, 3> light_positions{{{-9.0F, 9.0F, 9.0F}, {9.0F, 9.0F, 9.0F}, {0.0F, 14.0F, 7.0F}}};
    for (std::size_t i = 0; i < light_positions.size(); ++i) {
        auto const light = Entity{&scene, "key_light_{}", i};
        light.emplace<Components::Transform>(Components::Transform{.position = light_positions[i]});
        light.emplace<Components::PointLight>(
                Components::PointLight{.colour = {1.0F, 0.95F, 0.9F}, .intensity = 400.0F, .range = 60.0F});
    }

    constexpr int columns = 9;
    constexpr int rows = 3;
    constexpr float left = -(columns - 1) * spacing * 0.5F;

    // Wall 1 (top row): roughness along x, metallic along y rows (0, 0.5, 1). Gold base.
    for (int row = 0; row < rows; ++row) {
        auto const metallic = static_cast<float>(row) / (rows - 1);
        for (int col = 0; col < columns; ++col) {
            auto const roughness = std::max(0.04F, static_cast<float>(col) / (columns - 1));
            add_prop(std::format("rm_{}_{}", row, col), engine_models.sphere,
                     {left + static_cast<float>(col) * spacing, 12.6F - static_cast<float>(row) * spacing, 0.0F},
                     glm::vec3{sphere_scale}, make_material({1.0F, 0.77F, 0.34F}, metallic, roughness));
        }
    }

    // Wall 2: tinted dielectrics (columns) at three roughness levels (rows).
    constexpr std::array<float, 3> dielectric_roughness{0.1F, 0.45F, 0.9F};
    for (std::size_t row = 0; row < dielectric_roughness.size(); ++row) {
        for (int col = 0; col < columns; ++col) {
            auto const &colour = palette[static_cast<std::size_t>(col) % palette.size()];
            add_prop(std::format("dielectric_{}_{}", row, col), engine_models.sphere,
                     {left + static_cast<float>(col) * spacing, 6.0F - static_cast<float>(row) * spacing, 0.0F},
                     glm::vec3{sphere_scale}, make_material(colour, 0.0F, dielectric_roughness[row]));
        }
    }

    // Wall 3 (bottom): emissive colours, strength rising along x. Dark base so only the emission shows.
    for (int col = 0; col < columns; ++col) {
        auto const &colour = palette[static_cast<std::size_t>(col) % palette.size()];
        auto const strength = std::pow(2.0F, static_cast<float>(col) * 0.5F);
        add_prop(std::format("emissive_{}", col), engine_models.cube,
                 {left + static_cast<float>(col) * spacing, 0.4F, 3.0F}, glm::vec3{0.8F},
                 make_material({0.02F, 0.02F, 0.02F}, 0.0F, 0.6F, colour, strength));
    }

    // Decals: textured, projected boxes laid over the floor, the emissive cubes and each other.
    auto const add_decal = [&](std::string const &name, glm::vec3 const &position, float yaw_degrees,
                               glm::vec3 const &scale, MaterialHandle material, std::int32_t sort_order = 0) {
        auto const decal = Entity{&scene, "{}", name};
        decal.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .rotation = glm::angleAxis(glm::radians(yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}),
                .scale = scale,
        });
        decal.emplace<Components::Decal>(Components::Decal{.material = material, .sort_order = sort_order});
    };

    auto const make_decal_material = [&](ImageHandle albedo, ImageHandle normal, ImageHandle metallic_roughness,
                                         ImageHandle emissive, float emissive_strength, float normal_scale) {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{1.0F},
                .emissive_factor = glm::vec3{emissive_strength > 0.0F ? 1.0F : 0.0F},
                .emissive_strength = emissive_strength,
                .metallic_factor = 1.0F,
                .roughness_factor = 1.0F,
                .normal_scale = normal_scale,
                .base_colour_texture = albedo,
                .normal_texture = normal,
                .metallic_roughness_texture = metallic_roughness,
                .occlusion_texture = images.occlusion(),
                .emissive_texture = emissive,
                .sampler = samplers.linear_clamp(),
                .double_sided = true,
        });
        if (!material) {
            error("[MaterialWall] Could not create decal material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        return *material;
    };

    constexpr std::uint32_t decal_texture_size = 512;
    auto const srgb = VK_FORMAT_R8G8B8A8_SRGB;
    auto const unorm = VK_FORMAT_R8G8B8A8_UNORM;

    // Paint splat: alpha-masked blob + droplets, paint thickness as a normal map, glossy where it is thick.
    auto const splat_albedo = upload_texture(renderer, decal_texture_size, srgb, "decal.splat.albedo", [](glm::vec2 uv) {
        auto const alpha = glm::smoothstep(0.012F, -0.012F, splat_distance(uv));
        auto const rim = glm::smoothstep(0.0F, 0.12F, splat_thickness(uv));
        auto const mottling = fbm(uv * 9.0F, 21);
        auto const colour = glm::mix(glm::vec3{0.45F, 0.02F, 0.12F}, glm::vec3{0.95F, 0.08F, 0.38F}, rim) *
                            (0.8F + 0.4F * mottling);
        return Texel{colour, alpha};
    });
    auto const splat_normal = upload_texture(renderer, decal_texture_size, unorm, "decal.splat.normal", [](glm::vec2 uv) {
        return normal_from_height(splat_thickness, uv, 5.0F);
    });
    auto const splat_mr = upload_texture(renderer, decal_texture_size, unorm, "decal.splat.mr", [](glm::vec2 uv) {
        auto const wet = glm::smoothstep(0.1F, 0.7F, splat_thickness(uv));
        return Texel{0.0F, glm::mix(0.75F, 0.18F, wet), 0.0F, 1.0F};
    });
    auto const splat = make_decal_material(splat_albedo, splat_normal, splat_mr, images.emissive(), 0.0F, 1.0F);

    // Hazard panel: stripes on a bevelled plate with rivets; bare metal border, matt painted stripes.
    auto const panel_albedo = upload_texture(renderer, decal_texture_size, srgb, "decal.panel.albedo", [](glm::vec2 uv) {
        auto const p = uv - 0.5F;
        auto const alpha = glm::smoothstep(0.01F, -0.01F, rounded_box(p, {0.46F, 0.46F}, 0.05F));
        auto const inner = glm::smoothstep(0.01F, -0.01F, rounded_box(p, {0.36F, 0.36F}, 0.02F));
        auto const stripes = stripe_mask(uv);
        auto const paint = glm::mix(glm::vec3{0.95F, 0.72F, 0.05F}, glm::vec3{0.04F, 0.04F, 0.04F}, stripes);
        auto const grime = 0.7F + 0.3F * fbm(uv * 30.0F, 9);
        auto const metal = glm::vec3{0.62F, 0.63F, 0.66F};
        return Texel{glm::mix(metal, paint, inner) * grime, alpha};
    });
    auto const panel_normal = upload_texture(renderer, decal_texture_size, unorm, "decal.panel.normal", [](glm::vec2 uv) {
        return normal_from_height(panel_height, uv, 6.0F);
    });
    auto const panel_mr = upload_texture(renderer, decal_texture_size, unorm, "decal.panel.mr", [](glm::vec2 uv) {
        auto const inner = glm::smoothstep(0.01F, -0.01F, rounded_box(uv - 0.5F, {0.36F, 0.36F}, 0.02F));
        auto const scuffs = fbm(uv * 25.0F, 14);
        return Texel{0.0F, glm::mix(0.28F, 0.55F + 0.3F * scuffs, inner), glm::mix(1.0F, 0.0F, inner), 1.0F};
    });
    auto const panel = make_decal_material(panel_albedo, panel_normal, panel_mr, images.emissive(), 0.0F, 1.0F);

    // Glowing ring glyph: dark steel disc with emissive concentric rings and ticks.
    auto const ring_shade = [](glm::vec2 uv) {
        auto const p = uv - 0.5F;
        auto const r = glm::length(p);
        auto const angle = std::atan2(p.y, p.x);
        auto const ring = [&](float radius, float width) {
            return glm::smoothstep(width, width * 0.4F, std::abs(r - radius));
        };
        auto const ticks = glm::smoothstep(0.55F, 0.7F, std::sin(angle * 24.0F)) * ring(0.40F, 0.03F);
        return std::max({ring(0.45F, 0.012F), ring(0.30F, 0.008F), ring(0.14F, 0.02F), ticks * 0.8F});
    };
    auto const ring_albedo = upload_texture(renderer, decal_texture_size, srgb, "decal.ring.albedo", [](glm::vec2 uv) {
        auto const alpha = glm::smoothstep(0.01F, -0.01F, glm::length(uv - 0.5F) - 0.48F);
        return Texel{glm::vec3{0.05F, 0.06F, 0.07F} * (0.6F + 0.8F * fbm(uv * 20.0F, 2)), alpha};
    });
    auto const ring_emissive = upload_texture(renderer, decal_texture_size, srgb, "decal.ring.emissive",
                                              [&](glm::vec2 uv) {
        auto const glow = ring_shade(uv);
        return Texel{glm::vec3{0.1F, 0.7F, 1.0F} * glow, 1.0F};
    });
    auto const ring_normal = upload_texture(renderer, decal_texture_size, unorm, "decal.ring.normal", [&](glm::vec2 uv) {
        return normal_from_height([&](glm::vec2 q) { return ring_shade(q) * 0.6F; }, uv, 3.0F);
    });
    auto const ring = make_decal_material(ring_albedo, ring_normal, images.metallic_roughness(), ring_emissive, 6.0F, 1.0F);

    add_decal("decal_panel", {0.0F, 0.0F, 6.0F}, 12.0F, {5.0F, 1.0F, 5.0F}, panel);
    add_decal("decal_splat_a", {-3.5F, 0.0F, 6.5F}, 25.0F, {6.0F, 1.0F, 6.0F}, splat, 1);
    add_decal("decal_splat_b", {1.8F, 0.0F, 7.2F}, -40.0F, {4.0F, 1.0F, 4.0F}, splat, 2);
    add_decal("decal_ring", {5.5F, 0.4F, 3.5F}, 0.0F, {5.0F, 2.0F, 5.0F}, ring);

    // Shadow-casting spot lights over a few pillars standing on the decals.
    constexpr std::array<glm::vec3, 4> pillar_positions{{{-2.0F, 1.2F, 9.0F}, {2.2F, 0.8F, 8.2F},
                                                         {-6.0F, 1.0F, 7.0F}, {5.5F, 1.4F, 10.0F}}};
    for (std::size_t i = 0; i < pillar_positions.size(); ++i) {
        add_prop(std::format("pillar_{}", i), engine_models.cube, pillar_positions[i],
                 {0.35F, pillar_positions[i].y, 0.35F}, make_material({0.8F, 0.8F, 0.82F}, 0.0F, 0.6F));
    }

    struct SpotSetup {
        glm::vec3 position;
        glm::vec3 target;
        glm::vec3 colour;
    };
    constexpr std::array<SpotSetup, 2> spots{{
            {{-9.0F, 8.0F, 15.0F}, {-1.0F, 0.0F, 7.5F}, {1.0F, 0.45F, 0.35F}},
            {{9.0F, 8.0F, 15.0F}, {1.5F, 0.0F, 7.5F}, {0.35F, 0.6F, 1.0F}},
    }};
    for (std::size_t i = 0; i < spots.size(); ++i) {
        auto const light = Entity{&scene, "shadow_spot_{}", i};
        auto const direction = glm::normalize(spots[i].target - spots[i].position);
        light.emplace<Components::Transform>(Components::Transform{
                .position = spots[i].position,
                .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F}, direction),
        });
        light.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = spots[i].colour,
                .intensity = 2200.0F,
                .range = 40.0F,
                .inner_cone_degrees = 14.0F,
                .outer_cone_degrees = 24.0F,
                .casts_shadows = true,
        });
    }

    {
        auto const orbiter = Entity{&scene, "orbiter"};
        orbiter.emplace<Components::Transform>(Components::Transform{.position = orbit_centre_, .scale = glm::vec3{0.5F}});
        orbiter.emplace<Components::Model>(Components::Model{.model = engine_models.cube});
        orbiter_ = static_cast<entt::entity>(orbiter);
    }

    // A point light among the pillars: all six cube faces cast.
    {
        auto const lamp = Entity{&scene, "shadow_lamp"};
        lamp.emplace<Components::Transform>(Components::Transform{.position = {0.5F, 2.6F, 7.0F}});
        lamp.emplace<Components::PointLight>(Components::PointLight{
                .colour = {1.0F, 0.85F, 0.55F}, .intensity = 110.0F, .range = 14.0F, .casts_shadows = true});
    }

    info("[MaterialWall] Keys: 1-6 shot (5 = decals, 6 = spot shadows), B bloom, C shadow culling, V shadow caching, O orbit");
}

auto MaterialWallGame::on_update(Scene &scene, float delta_time) -> void {
    time_ += delta_time;

    if (orbiting_) {
        orbit_time_ += delta_time;
    }

    if (auto *transform = scene.get_registry().try_get<Components::Transform>(orbiter_)) {
        transform->position = orbit_centre_ + glm::vec3{std::cos(orbit_time_ * 1.5F) * 1.2F, 0.0F, std::sin(orbit_time_ * 1.5F) * 1.2F};
    }

    if (renderer_ != nullptr && ++frames_ % 60 == 0) {
        auto const &stats = renderer_->local_shadow_stats();
        info("[MaterialWall] Local shadows: {} lights ({} cached), {} tiles drawn, {}/256 atlas cells", stats.shadowed_lights,
             stats.cached_lights, stats.rendered_tiles, stats.atlas_cells_used);
    }

    switch (shot_) {
        case 0: // everything
            eye_ = {std::sin(time_ * 0.1F) * 3.0F, 6.5F, 26.0F};
            look_at_ = {0.0F, 6.0F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        case 1: // roughness x metallic
            eye_ = {0.0F, 11.0F, 14.0F};
            look_at_ = {0.0F, 10.5F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        case 2: // dielectrics
            eye_ = {0.0F, 5.0F, 14.0F};
            look_at_ = {0.0F, 4.5F, 0.0F};
            fov_degrees_ = 50.0F;
            break;
        case 4: // decals
            eye_ = {0.0F, 7.0F, 15.0F};
            look_at_ = {0.5F, 0.0F, 5.5F};
            fov_degrees_ = 45.0F;
            break;
        case 5: // shadow-casting spot lights
            eye_ = {0.0F, 9.0F, 20.0F};
            look_at_ = {0.0F, 0.0F, 7.5F};
            fov_degrees_ = 45.0F;
            break;
        default: // emissive
            eye_ = {0.0F, 2.5F, 14.0F};
            look_at_ = {0.0F, 0.8F, 3.0F};
            fov_degrees_ = 45.0F;
            break;
    }
}

auto MaterialWallGame::subscribe_input(InputContext const &ctx) -> void {
    listen(ctx, &MaterialWallGame::on_key_pressed);
}

auto MaterialWallGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (event.key >= GLFW_KEY_1 && event.key < GLFW_KEY_1 + static_cast<std::int32_t>(shot_count)) {
        shot_ = static_cast<std::size_t>(event.key - GLFW_KEY_1);
        return;
    }

    if (event.key == GLFW_KEY_C && renderer_ != nullptr) {
        auto settings = renderer_->local_shadow_settings();
        settings.gpu_culling = !settings.gpu_culling;
        renderer_->set_local_shadow_settings(settings);
        info("[MaterialWall] Local shadow GPU culling {}", settings.gpu_culling ? "on" : "off");
        return;
    }

    if (event.key == GLFW_KEY_O) {
        orbiting_ = !orbiting_;
        info("[MaterialWall] Orbiting cube {}", orbiting_ ? "moving" : "stopped");
        return;
    }

    if (event.key == GLFW_KEY_V && renderer_ != nullptr) {
        auto settings = renderer_->local_shadow_settings();
        settings.caching = !settings.caching;
        renderer_->set_local_shadow_settings(settings);
        info("[MaterialWall] Local shadow caching {}", settings.caching ? "on" : "off");
        return;
    }

    if (event.key == GLFW_KEY_B && renderer_ != nullptr) {
        auto bloom = renderer_->bloom_settings();
        bloom.enabled = !bloom.enabled;
        renderer_->set_bloom_settings(bloom);
    }
}

auto MaterialWallGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = glm::lookAtLH(eye_, look_at_, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov_degrees_), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(fov_degrees_),
    };
}
