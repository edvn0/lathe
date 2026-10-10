#include "rendering/particle_system.hxx"

#include <algorithm>
#include <cmath>
#include <format>

#include <tracy/Tracy.hpp>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>

#include "assets/material_storage.hxx"
#include "rendering/scene.hxx"

namespace {
    constexpr auto simulate_group_size = std::uint32_t{64};

    // The members in Layout, in order, are SimulatePushConstants in particles_simulate.slang. Vectors are three
    // consecutive floats there, so the layouts agree.
    struct SimulateParams {
        BufferReadWrite particles;
        std::uint32_t count = 0;
        std::uint32_t head = 0;
        std::uint32_t spawn_count = 0;
        std::uint32_t seed = 0;
        float delta_time = 0.0F;
        float lifetime = 0.0F;
        glm::vec3 gravity{0.0F};
        std::uint32_t shape = 0;
        float shape_size = 0.0F;
        float speed = 0.0F;
        float speed_variance = 0.0F;
        float cone_radians = 0.0F;
        glm::vec3 position{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};

        using Layout = PushLayout<&SimulateParams::particles, &SimulateParams::count, &SimulateParams::head,
                                  &SimulateParams::spawn_count, &SimulateParams::seed, &SimulateParams::delta_time,
                                  &SimulateParams::lifetime, &SimulateParams::gravity, &SimulateParams::shape,
                                  &SimulateParams::shape_size, &SimulateParams::speed, &SimulateParams::speed_variance,
                                  &SimulateParams::cone_radians, &SimulateParams::position, &SimulateParams::rotation>;
    };

    // Mirrors DrawPushConstants in particles_draw.slang.
    struct DrawPush {
        glm::mat4 view_projection{1.0F};
        VkDeviceAddress particles = 0;
        float size_start = 0.0F;
        float size_end = 0.0F;
        glm::vec4 colour_start{1.0F};
        glm::vec4 colour_end{1.0F};
    };

    static_assert(GameGraph::push_size<SimulateParams>(SimulateParams::Layout{}) == 92);
    static_assert(sizeof(DrawPush) == 112);

    auto finite_or(float value, float fallback) -> float { return std::isfinite(value) ? value : fallback; }

    auto finite_or(glm::vec4 value, glm::vec4 fallback) -> glm::vec4 {
        return {finite_or(value.x, fallback.x), finite_or(value.y, fallback.y), finite_or(value.z, fallback.z),
                finite_or(value.w, fallback.w)};
    }
}

auto ParticleSystem::sanitised(Components::ParticleEmitter emitter) -> Components::ParticleEmitter {
    constexpr auto defaults = Components::ParticleEmitter{};

    emitter.count = std::clamp(emitter.count, std::uint32_t{1}, Components::ParticleEmitter::max_count);
    emitter.rate = std::clamp(finite_or(emitter.rate, defaults.rate), 0.0F, 1.0e6F);
    emitter.lifetime = std::clamp(finite_or(emitter.lifetime, defaults.lifetime), 0.01F, 3600.0F);
    emitter.gravity = {finite_or(emitter.gravity.x, 0.0F), finite_or(emitter.gravity.y, 0.0F),
                       finite_or(emitter.gravity.z, 0.0F)};
    emitter.shape_size = std::clamp(finite_or(emitter.shape_size, defaults.shape_size), 0.0F, 1.0e4F);
    emitter.cone_degrees = std::clamp(finite_or(emitter.cone_degrees, defaults.cone_degrees), 0.0F, 180.0F);
    emitter.speed = std::clamp(finite_or(emitter.speed, defaults.speed), -1.0e4F, 1.0e4F);
    emitter.speed_variance = std::clamp(finite_or(emitter.speed_variance, defaults.speed_variance), 0.0F, 1.0F);
    emitter.size_start = std::clamp(finite_or(emitter.size_start, defaults.size_start), 0.0F, 1.0e3F);
    emitter.size_end = std::clamp(finite_or(emitter.size_end, defaults.size_end), 0.0F, 1.0e3F);
    emitter.colour_start = finite_or(emitter.colour_start, defaults.colour_start);
    emitter.colour_end = finite_or(emitter.colour_end, defaults.colour_end);
    return emitter;
}

auto ParticleSystem::create(GameGpu &gpu) -> std::expected<void, RendererError> {
    ZoneScopedNC("Particle system create", tracy::Color::DeepPink);

    if (ready_) {
        return {};
    }

    auto simulate = gpu.register_compute({
            .source = "assets/shaders/particles_simulate.slang",
            .debug_name = "particles_simulate",
    });
    if (!simulate) {
        return std::unexpected(simulate.error());
    }

    auto draw = gpu.register_graphics({
            .source = "assets/shaders/particles_draw.slang",
            .blending = true,
            .debug_name = "particles_draw",
    });
    if (!draw) {
        return std::unexpected(draw.error());
    }

    gpu_ = &gpu;
    use_shaders(*simulate, *draw);
    return {};
}

auto ParticleSystem::use_shaders(GameComputeShader simulate, GameGraphicsShader draw) noexcept -> void {
    simulate_ = simulate;
    draw_ = draw;
    ready_ = true;
}

auto ParticleSystem::declare(GameGraph &graph, entt::registry const &registry, MaterialStorage const *materials,
                             float delta_time) -> void {
    if (!ready_ || graph.slot() != GameSlot::frame_start) {
        return;
    }
    ZoneScopedNC("Particles declare", tracy::Color::DeepPink);

    for (auto &[entity, state]: states_) {
        state.seen = false;
    }

    delta_time = std::clamp(finite_or(delta_time, 0.0F), 0.0F, max_delta_time);

    for (auto const entity: registry.view<Components::ParticleEmitter>()) {
        auto const emitter = sanitised(registry.get<Components::ParticleEmitter>(entity));

        auto &state = states_[entity];
        state.seen = true;
        if (state.buffer.empty()) {
            state.buffer = std::format("particles.{}", entt::to_integral(entity));
        }
        if (state.count != emitter.count) {
            // The buffer is recreated, empty, at the new size.
            state.count = emitter.count;
            state.head = 0;
            state.carry = 0.0F;
        }

        auto spawn_count = std::uint32_t{0};
        if (emitter.emitting) {
            state.carry += emitter.rate * delta_time;
            auto const whole = std::min(std::floor(state.carry), static_cast<float>(emitter.count));
            spawn_count = static_cast<std::uint32_t>(whole);
            state.carry = std::min(state.carry - whole, 1.0F);
        }
        ++state.frame;

        auto position = glm::vec3{0.0F};
        auto rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        if (auto const *transform = registry.try_get<Components::Transform>(entity)) {
            auto const world = systems::get_world_transform(registry, entity, *transform);
            position = glm::vec3{world[3]};
            rotation = glm::normalize(glm::quat_cast(glm::mat3{glm::normalize(glm::vec3{world[0]}),
                                                                 glm::normalize(glm::vec3{world[1]}),
                                                                 glm::normalize(glm::vec3{world[2]})}));
        }

        auto tint = glm::vec3{1.0F};
        auto glow = glm::vec3{0.0F};
        if (materials != nullptr && emitter.material.valid()) {
            if (auto const *material = materials->get(emitter.material)) {
                tint = glm::vec3{material->base_colour_factor};
                glow = material->emissive_factor * material->emissive_strength;
            }
        }

        // One emitter that cannot be declared is dropped alone.
        graph.isolated(std::format("particles {}", entt::to_integral(entity)), [&] {
            auto particles = graph.persistent_buffer(state.buffer, {.size = emitter.count * particle_bytes});

            auto simulate = SimulateParams{
                    .particles = {particles},
                    .count = emitter.count,
                    .head = state.head,
                    .spawn_count = spawn_count,
                    .seed = state.frame * 2891336453U + entt::to_integral(entity) * 747796405U,
                    .delta_time = delta_time,
                    .lifetime = emitter.lifetime,
                    .gravity = emitter.gravity,
                    .shape = static_cast<std::uint32_t>(emitter.shape),
                    .shape_size = emitter.shape_size,
                    .speed = emitter.speed,
                    .speed_variance = emitter.speed_variance,
                    .cone_radians = glm::radians(emitter.cone_degrees),
                    .position = position,
                    .rotation = rotation,
            };
            graph.add_compute(std::format("particles_simulate.{}", entt::to_integral(entity)), simulate, simulate_,
                              Threads{.x = emitter.count, .group_x = simulate_group_size},
                              {.label = "Particles simulate"});
            particles = simulate.particles.buffer;
            state.head = (state.head + spawn_count) % emitter.count;

            auto const colour_start = glm::vec4{glm::vec3{emitter.colour_start} * tint + glow, emitter.colour_start.w};
            auto const colour_end = glm::vec4{glm::vec3{emitter.colour_end} * tint + glow, emitter.colour_end.w};
            auto const count = emitter.count;

            // The draw reads what the simulation wrote. Declaring the read is what orders the forward pass after it.
            graph.add_scene_draw(std::format("particles_draw.{}", entt::to_integral(entity)), [&](GameDrawBuilder &draw) {
                auto const state_buffer = draw.read(particles);

                return [this, state_buffer, count, size_start = emitter.size_start, size_end = emitter.size_end,
                        colour_start, colour_end](GameDrawContext &context) {
                    context.bind(draw_);
                    context.push(DrawPush{
                            .view_projection = context.view_projection(),
                            .particles = context.address(state_buffer),
                            .size_start = size_start,
                            .size_end = size_end,
                            .colour_start = colour_start,
                            .colour_end = colour_end,
                    });
                    context.draw(count * 6U);
                };
            });
        });
    }

    // An emitter that is gone takes its particles with it.
    std::erase_if(states_, [&](auto const &entry) {
        if (entry.second.seen) {
            return false;
        }
        if (gpu_ != nullptr) {
            gpu_->release_persistent_buffer(entry.second.buffer);
        }
        return true;
    });
}
