#include "glass_game.hxx"

#include <array>
#include <cmath>
#include <format>

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

auto GlassGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    // Ctrl-R populates the same scene again: drop the last gallery and the materials it made first.
    scene.get_registry().clear();
    if (!owned_materials_.empty()) {
        if (auto const idle = renderer.wait_idle(); !idle) {
            warn("{}", describe(idle.error()));
        }
        for (auto const material: owned_materials_) {
            renderer.release_material(material);
        }
        owned_materials_.clear();
    }

    scene.environment.source = EnvironmentSource::procedural_sky;
    scene.environment.draw_skybox = true;
    scene.environment.sun.elevation_degrees = 35.0F;
    scene.environment.sun.intensity = 0.15F;

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    struct Look {
        glm::vec4 colour;
        float metallic = 0.0F;
        float roughness = 0.5F;
        glm::vec3 emissive{0.0F};
        float emissive_strength = 1.0F;
        AlphaMode alpha_mode = AlphaMode::opaque;
        bool double_sided = false;
    };

    auto const make_material = [&](Look const &look) -> MaterialHandle {
        auto material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = look.colour,
                .emissive_factor = look.emissive,
                .emissive_strength = look.emissive_strength,
                .metallic_factor = look.metallic,
                .roughness_factor = look.roughness,
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = images.emissive(),
                .sampler = samplers.linear_repeat(),
                .alpha_mode = look.alpha_mode,
                .double_sided = look.double_sided,
        });
        if (!material) {
            error("[Glass] Could not create material: {}", describe(material.error()));
            return MaterialHandle{};
        }
        owned_materials_.push_back(*material);
        return *material;
    };

    auto const add_prop = [&](std::string const &name, ModelHandle model, glm::vec3 const &position,
                              glm::vec3 const &scale, MaterialHandle material, float yaw_degrees = 0.0F) -> Entity {
        auto const prop = Entity{&scene, "{}", name};
        prop.emplace<Components::Transform>(Components::Transform{
                .position = position,
                .rotation = glm::angleAxis(glm::radians(yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}),
                .scale = scale,
        });
        prop.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            prop.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
        return prop;
    };

    auto const opaque = [&](glm::vec3 const &colour, float roughness = 0.6F, float metallic = 0.0F) {
        return make_material(Look{.colour = {colour, 1.0F}, .metallic = metallic, .roughness = roughness});
    };
    auto const blended = [&](glm::vec4 const &colour, float roughness, glm::vec3 const &emissive = glm::vec3{0.0F},
                             float emissive_strength = 1.0F) {
        return make_material(Look{.colour = colour,
                                  .roughness = roughness,
                                  .emissive = emissive,
                                  .emissive_strength = emissive_strength,
                                  .alpha_mode = AlphaMode::blend,
                                  .double_sided = true});
    };

    // Opaque stage: a long floor and a back wall of coloured stripes, so every piece of glass has something to
    // distort the view of (well, tint: blending cannot refract).
    add_prop("floor", engine_models.cube, {0.0F, -0.25F, 0.0F}, {16.0F, 0.25F, 8.0F}, opaque({0.55F, 0.55F, 0.58F}, 0.8F));

    constexpr std::array<glm::vec3, 8> stripe_colours{{
            {0.90F, 0.12F, 0.10F},
            {0.95F, 0.55F, 0.10F},
            {0.95F, 0.85F, 0.15F},
            {0.15F, 0.75F, 0.25F},
            {0.10F, 0.70F, 0.80F},
            {0.15F, 0.30F, 0.90F},
            {0.60F, 0.20F, 0.85F},
            {0.92F, 0.92F, 0.92F},
    }};
    for (std::size_t i = 0; i < stripe_colours.size(); ++i) {
        add_prop(std::format("stripe_{}", i), engine_models.cube,
                 {-14.0F + 4.0F * static_cast<float>(i) + 2.0F, 2.0F, -6.0F}, {2.0F, 2.0F, 0.1F},
                 opaque(stripe_colours[i], 0.7F));
    }

    // Row 1 (z = 0): spheres from near-clear to heavily tinted, then frosted, so alpha and roughness both show.
    struct Ball {
        glm::vec4 colour;
        float roughness;
    };
    constexpr std::array<Ball, 7> balls{{
            {{0.85F, 0.95F, 1.0F, 0.12F}, 0.02F},  // clear glass
            {{0.85F, 0.95F, 1.0F, 0.30F}, 0.02F},  // thicker clear glass
            {{0.30F, 0.80F, 0.45F, 0.45F}, 0.05F}, // green bottle glass
            {{0.95F, 0.55F, 0.15F, 0.50F}, 0.05F}, // amber
            {{0.25F, 0.40F, 0.95F, 0.55F}, 0.05F}, // cobalt
            {{0.95F, 0.95F, 1.0F, 0.40F}, 0.55F},  // frosted
            {{0.95F, 0.95F, 1.0F, 0.75F}, 0.90F},  // milky
    }};
    for (std::size_t i = 0; i < balls.size(); ++i) {
        add_prop(std::format("ball_{}", i), engine_models.sphere, {-9.0F + 3.0F * static_cast<float>(i), 1.0F, 0.0F},
                 glm::vec3{1.0F}, blended(balls[i].colour, balls[i].roughness));
    }

    // Row 2 (z = 3.5): a glass bottle (two capsules: the shell and a smaller one for the liquid) with a glowing
    // bulb in the middle, a cube of ice, and a sphere inside a sphere.
    add_prop("bottle_shell", engine_models.capsule, {-7.5F, 1.4F, 3.5F}, {0.9F, 1.1F, 0.9F},
             blended({0.80F, 0.95F, 0.90F, 0.22F}, 0.03F));
    add_prop("bottle_liquid", engine_models.capsule, {-7.5F, 1.1F, 3.5F}, {0.7F, 0.7F, 0.7F},
             blended({0.95F, 0.35F, 0.25F, 0.55F}, 0.1F));

    add_prop("bulb_glass", engine_models.sphere, {-4.0F, 1.2F, 3.5F}, glm::vec3{1.2F},
             blended({0.95F, 0.95F, 1.0F, 0.15F}, 0.02F));
    add_prop("bulb_filament", engine_models.sphere, {-4.0F, 1.2F, 3.5F}, glm::vec3{0.35F},
             make_material(Look{.colour = {1.0F, 0.8F, 0.5F, 1.0F},
                                .emissive = {1.0F, 0.7F, 0.35F},
                                .emissive_strength = 6.0F}));
    {
        auto const bulb_light = Entity{&scene, "bulb_light"};
        bulb_light.emplace<Components::Transform>(Components::Transform{.position = {-4.0F, 1.2F, 3.5F}});
        bulb_light.emplace<Components::PointLight>(Components::PointLight{
                .colour = {1.0F, 0.75F, 0.45F}, .intensity = 60.0F, .range = 10.0F, .casts_shadows = true});
    }

    add_prop("ice_cube", engine_models.cube, {-0.5F, 0.9F, 3.5F}, glm::vec3{0.9F},
             blended({0.75F, 0.9F, 1.0F, 0.35F}, 0.25F), 25.0F);
    add_prop("nested_outer", engine_models.sphere, {3.0F, 1.2F, 3.5F}, glm::vec3{1.2F},
             blended({0.9F, 0.95F, 1.0F, 0.18F}, 0.02F));
    add_prop("nested_inner", engine_models.sphere, {3.0F, 1.2F, 3.5F}, glm::vec3{0.6F},
             blended({0.95F, 0.3F, 0.5F, 0.6F}, 0.1F));
    add_prop("nested_core", engine_models.sphere, {3.0F, 1.2F, 3.5F}, glm::vec3{0.25F},
             opaque({0.95F, 0.85F, 0.2F}, 0.3F, 1.0F));

    // Stained-glass window: overlapping gel panes in a row of three, each in front of the one behind, so mixing
    // red + green + blue on top of each other tests the back-to-front sort from every camera angle.
    constexpr std::array<glm::vec4, 3> gels{{
            {0.95F, 0.15F, 0.15F, 0.45F},
            {0.15F, 0.85F, 0.25F, 0.45F},
            {0.15F, 0.30F, 0.95F, 0.45F},
    }};
    for (std::size_t i = 0; i < gels.size(); ++i) {
        auto const fi = static_cast<float>(i);
        add_prop(std::format("gel_{}", i), engine_models.cube, {6.5F + 1.0F * fi, 1.6F, 3.5F - 0.7F * fi},
                 {1.6F, 1.6F, 0.04F}, blended(gels[i], 0.05F), -15.0F);
    }

    // A pane in front of the whole first row, just to show how much a large sheet in the foreground washes out.
    add_prop("big_pane", engine_models.cube, {-3.0F, 1.5F, -3.0F}, {7.0F, 1.5F, 0.03F},
             blended({0.7F, 0.85F, 1.0F, 0.10F}, 0.02F));

    // Some opaque props behind and between the glass.
    add_prop("cube_red", engine_models.cube, {-6.0F, 0.6F, -2.0F}, glm::vec3{0.6F}, opaque({0.9F, 0.15F, 0.1F}), 15.0F);
    add_prop("cube_white", engine_models.cube, {1.5F, 0.6F, -2.0F}, glm::vec3{0.6F}, opaque({0.9F, 0.9F, 0.9F}), -20.0F);
    add_prop("metal", engine_models.sphere, {8.5F, 0.8F, -1.0F}, glm::vec3{0.8F}, opaque({0.9F, 0.75F, 0.4F}, 0.25F, 1.0F));

    constexpr std::array<glm::vec3, 2> light_positions{{{-6.0F, 8.0F, 8.0F}, {8.0F, 7.0F, 6.0F}}};
    for (std::size_t i = 0; i < light_positions.size(); ++i) {
        auto const light = Entity{&scene, "glass_lamp_{}", i};
        light.emplace<Components::Transform>(Components::Transform{
                .position = light_positions[i],
                .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F}, glm::normalize(-light_positions[i])),
        });
        light.emplace<Components::SpotLight>(Components::SpotLight{
                .colour = {1.0F, 0.95F, 0.85F},
                .intensity = 500.0F,
                .range = 40.0F,
                .inner_cone_degrees = 25.0F,
                .outer_cone_degrees = 45.0F,
                .casts_shadows = true,
        });
    }

    info("[Glass] Keys: 1-3 shots, P pause orbit");
}

auto GlassGame::on_update(Scene &, float) -> void {
    if (orbiting_) {
        ++frame_;
    }
}

auto GlassGame::subscribe_input(InputContext const &ctx) -> void {
    listen(ctx, &GlassGame::on_key_pressed);
}

auto GlassGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    switch (event.key) {
        case GLFW_KEY_1:
            shot_ = 0;
            break;
        case GLFW_KEY_2:
            shot_ = 1;
            break;
        case GLFW_KEY_3:
            shot_ = 2;
            break;
        case GLFW_KEY_P:
            orbiting_ = !orbiting_;
            break;
        default:
            break;
    }
}

auto GlassGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    // Shot 1 faces the whole gallery, shot 2 sweeps behind the glass, shot 3 looks down the gel panes.
    auto const angle = static_cast<float>(frame_) * 0.006F;
    glm::vec3 eye{};
    glm::vec3 target{};

    switch (shot_) {
        case 1:
            eye = {std::sin(angle) * 14.0F, 3.5F, -10.0F + std::cos(angle) * 2.0F};
            target = {0.0F, 1.0F, 1.0F};
            break;
        case 2:
            eye = {8.0F, 3.0F, 10.0F};
            target = {6.0F, 1.4F, 3.0F};
            break;
        default:
            eye = {std::sin(angle) * 4.0F, 5.0F, 13.0F};
            target = {0.0F, 0.9F, 1.5F};
            break;
    }

    return CameraParams{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(50.0F), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(50.0F),
    };
}
