#include "particle_field.hxx"

#include <tracy/Tracy.hpp>

#include <glm/mat4x4.hpp>

#include "rendering/frame_graph/types.hxx"

namespace {
    // Particle { float4 position_life; float4 velocity_size; } in the shaders.
    constexpr auto particle_bytes = VkDeviceSize{32};

    // Mirrors SimulatePushConstants in particles_simulate.slang.
    struct SimulatePush {
        VkDeviceAddress particles = 0;
        std::uint32_t count = 0;
        std::uint32_t frame = 0;
        float delta_time = 0.0F;
        float emitter_x = 0.0F;
        float emitter_y = 0.0F;
        float emitter_z = 0.0F;
    };

    // Mirrors DrawPushConstants in particles_draw.slang.
    struct DrawPush {
        glm::mat4 view_proj{1.0F};
        VkDeviceAddress particles = 0;
    };

    constexpr auto simulate_group_size = std::uint32_t{64};
}

auto ParticleField::create(GameGpu &gpu) -> std::expected<void, RendererError> {
    ZoneScopedNC("Particles create", tracy::Color::DeepPink);

    if (particles_.valid()) {
        return {};
    }

    auto simulate = gpu.register_compute({
            .source = "assets/shaders/game/particles_simulate.slang",
            .debug_name = "game.particles_simulate",
    });
    if (!simulate) {
        return std::unexpected(simulate.error());
    }

    auto draw = gpu.register_graphics({
            .source = "assets/shaders/game/particles_draw.slang",
            .debug_name = "game.particles_draw",
    });
    if (!draw) {
        return std::unexpected(draw.error());
    }

    auto particles = gpu.create_buffer({.size = particle_count * particle_bytes, .debug_name = "game_particles"});
    if (!particles) {
        return std::unexpected(particles.error());
    }

    simulate_ = *simulate;
    draw_ = *draw;
    particles_ = *particles;
    return {};
}

auto ParticleField::declare(GameGraph &graph, float delta_time) -> void {
    if (!particles_.valid() || graph.slot() != GameSlot::frame_start) {
        return;
    }
    ZoneScopedNC("Particles declare", tracy::Color::DeepPink);

    auto particles = graph.import(particles_);
    ++frame_;

    graph.add_compute_pass(
            "game_particles_simulate",
            {.label = "Particles simulate", .color = static_cast<std::uint32_t>(tracy::Color::DeepPink)},
            [&](GameComputeBuilder &pass) {
                pass.queue(frame_graph::QueueAffinity::compute_preferred);
                auto const state = pass.read_write(particles);

                return [this, state, delta_time](GameComputeContext &context) {
                    context.bind(simulate_);
                    context.push(SimulatePush{
                            .particles = context.address(state),
                            .count = particle_count,
                            .frame = frame_,
                            .delta_time = delta_time,
                            .emitter_y = 2.0F,
                    });
                    context.dispatch_threads(particle_count, simulate_group_size);
                };
            });

    // The draw reads what the simulation wrote. Declaring the read is what orders the forward pass after it (and
    // adds a cross-queue wait if the simulation ran on the async compute queue).
    graph.add_scene_draw("game_particles_draw", [&](GameDrawBuilder &draw) {
        auto const state = draw.read(particles);

        return [this, state](GameDrawContext &context) {
            context.bind(draw_);
            context.push(DrawPush{.view_proj = context.view_projection(), .particles = context.address(state)});
            context.draw(particle_count * 6U);
        };
    });
}
