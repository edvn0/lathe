#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <entt/entt.hpp>

#include "frame_graph_test_support.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/particle_system.hxx"

using namespace frame_graph;

namespace {
    auto topologies() -> std::vector<QueueTopology> {
        auto split = QueueTopology{};
        split.family = {0, 1};
        auto same_family = QueueTopology{};
        same_family.family = {0, 0};
        same_family.queue_index = {0, 1};
        return {QueueTopology{}, same_family, split};
    }

    auto services_for(FrameGraph &graph) -> GameGraphServices {
        return GameGraphServices{
                .scene_extent = {64, 64},
                .bindless_index = [](std::uint32_t resource) { return 100U + resource; },
                .acquire_buffer = [&graph](GameBufferRequest const &request) -> std::optional<BufferId> {
                    return graph.import_buffer({
                            .entry = request.persistent ? persistent_buffer_state : ResourceState{},
                            .exit = request.persistent ? persistent_buffer_state : ResourceState{},
                            .owner = Owner::game,
                            .debug_name = request.name,
                            .buffer = {.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{1}),
                                       .address = 0xABC,
                                       .size = request.size},
                    });
                },
        };
    }

    // The part of Renderer::record_frame the particles touch: the game's slot, then a forward pass that reads what
    // the scene draws read, as the renderer does.
    auto declare_frame(FrameGraph &graph, GameGraph &game, ParticleSystem &system, entt::registry const &registry,
                       float delta_time) -> void {
        auto swapchain = graph.import_image({
                .entry = {.stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });

        game.run_slot(GameSlot::frame_start, std::nullopt, std::nullopt,
                      [&](GameGraph &g) { system.declare(g, registry, nullptr, delta_time); });

        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &pass) {
            for (auto const &read: game.forward_reads()) {
                [[maybe_unused]] auto const declared = pass.read(
                        BufferId{.index = read.buffer.index, .generation = graph.latest_version(read.buffer.index)},
                        Use::shader_read, read.stages);
            }
            auto hdr = pass.create({.format = VK_FORMAT_R16G16B16A16_SFLOAT, .extent = {64, 64, 1}, .debug_name = "hdr"});
            hdr = pass.color(hdr, LoadOp::clear, StoreOp::store);
            swapchain = pass.color(swapchain, LoadOp::dont_care, StoreOp::store);
            return RecordFn{};
        });
    }

    auto emitter_entity(entt::registry &registry, Components::ParticleEmitter emitter) -> entt::entity {
        auto const entity = registry.create();
        registry.emplace<Components::Transform>(entity, Components::Transform{.position = {1.0F, 2.0F, 3.0F}});
        registry.emplace<Components::ParticleEmitter>(entity, emitter);
        return entity;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("every emitter gets a simulation pass and a scene draw, sound on every topology") {
        for (auto const &topology: topologies()) {
            auto registry = entt::registry{};
            emitter_entity(registry, {.count = 256});
            emitter_entity(registry, {.count = 4096, .shape = Components::ParticleShape::sphere});
            auto system = ParticleSystem{};
            system.use_shaders({}, {});

            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};
            declare_frame(graph, game, system, registry, 1.0F / 60.0F);

            CHECK(game.rolled_back_slots() == 0);
            REQUIRE(graph.declaration_errors().empty());
            CHECK(game.has_scene_draws());
            CHECK(game.forward_reads().size() == 2);

            auto simulations = 0;
            for (auto const &pass: graph.description().passes) {
                if (pass.owner != Owner::game) {
                    continue;
                }
                ++simulations;
                CHECK(pass.type == PassType::compute);
                // The buffer is shared by the frames in flight, so the pass stays on the graphics queue.
                CHECK(pass.affinity == QueueAffinity::graphics);
                REQUIRE(pass.accesses.size() == 1);
                CHECK(pass.accesses.front().use == Use::shader_read_write);
            }
            CHECK(simulations == 2);

            auto const compiled = compile(graph, topology);
            REQUIRE(compiled.has_value());
            auto const problems = test::check_happens_before(graph.description(), *compiled, topology);
            for (auto const &problem: problems) {
                MESSAGE(problem);
            }
            CHECK(problems.empty());
        }
    }

    TEST_CASE("a destroyed emitter is forgotten, and a system without shaders declares nothing") {
        auto registry = entt::registry{};
        auto const kept = emitter_entity(registry, {});
        auto const dropped = emitter_entity(registry, {});
        auto system = ParticleSystem{};

        {
            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};
            declare_frame(graph, game, system, registry, 0.016F);
            CHECK(graph.description().passes.size() == 1);
            CHECK(system.emitter_count() == 0);
        }

        system.use_shaders({}, {});
        for (auto const expected: {2U, 1U}) {
            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};
            declare_frame(graph, game, system, registry, 0.016F);
            CHECK(system.emitter_count() == expected);
            if (registry.valid(dropped)) {
                registry.destroy(dropped);
            }
        }
        CHECK(registry.valid(kept));
    }

    TEST_CASE("a game slot other than frame_start declares nothing") {
        auto registry = entt::registry{};
        emitter_entity(registry, {});
        auto system = ParticleSystem{};
        system.use_shaders({}, {});

        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        game.run_slot(GameSlot::after_lighting, std::nullopt, std::nullopt,
                      [&](GameGraph &g) { system.declare(g, registry, nullptr, 0.016F); });

        CHECK(graph.description().passes.empty());
        CHECK_FALSE(game.has_scene_draws());
    }

    TEST_CASE("an emitter is brought into the range the shaders and buffers support") {
        constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
        constexpr auto inf = std::numeric_limits<float>::infinity();
        auto const defaults = Components::ParticleEmitter{};

        auto const wild = ParticleSystem::sanitised({
                .count = 0,
                .rate = nan,
                .lifetime = -4.0F,
                .gravity = {inf, 0.0F, nan},
                .shape_size = -1.0F,
                .cone_degrees = 900.0F,
                .speed_variance = 5.0F,
                .size_start = nan,
                .colour_start = {nan, 0.0F, 0.0F, 0.0F},
        });
        CHECK(wild.count == 1);
        CHECK(wild.rate == defaults.rate);
        CHECK(wild.lifetime > 0.0F);
        CHECK(wild.gravity == glm::vec3{0.0F});
        CHECK(wild.shape_size == 0.0F);
        CHECK(wild.cone_degrees == 180.0F);
        CHECK(wild.speed_variance == 1.0F);
        CHECK(wild.size_start == defaults.size_start);
        CHECK(wild.colour_start == glm::vec4{defaults.colour_start.x, 0.0F, 0.0F, 0.0F});

        CHECK(ParticleSystem::sanitised({.count = 0xFFFFFFFFU}).count == Components::ParticleEmitter::max_count);
        CHECK(ParticleSystem::sanitised(defaults).count == defaults.count);
    }
}
