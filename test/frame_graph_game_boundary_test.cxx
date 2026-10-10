#include <doctest/doctest.h>

#include <functional>
#include <optional>

#include "frame_graph_test_support.hxx"
#include "gpu/image.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/lint.hxx"
#include "rendering/game_graph.hxx"

using namespace frame_graph;

namespace {
    constexpr auto compute = stages_of(ShaderStage::compute);
    constexpr auto fragment = stages_of(ShaderStage::fragment);

    auto sampled_image(std::string_view name, VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT) -> TransientImageDesc {
        return {
                .format = format,
                .extent = {64, 64, 1},
                .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                .debug_name = name,
        };
    }

    auto topologies() -> std::vector<QueueTopology> {
        auto split = QueueTopology{};
        split.family = {0, 1};
        auto same_family = QueueTopology{};
        same_family.family = {0, 0};
        same_family.queue_index = {0, 1};
        return {QueueTopology{}, same_family, split};
    }

    struct EngineImages {
        ImageId depth{};
        ImageId hdr{};
    };

    // depth prepass -> forward -> composition, with the game's slots in between, like Renderer::record_frame.
    auto declare_frame(FrameGraph &graph, GameGraph &game, std::function<void(GameGraph &)> const &hook) -> EngineImages {
        auto images = EngineImages{};
        auto swapchain = graph.import_image({
                .entry = {.stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });

        game.run_slot(GameSlot::frame_start, std::nullopt, std::nullopt, hook);

        graph.add_pass("depth_prepass", PassType::raster, {}, [&](PassBuilder &pass) {
            images.depth = pass.create(sampled_image("depth", VK_FORMAT_D32_SFLOAT));
            images.depth = pass.write_depth(images.depth, LoadOp::clear, StoreOp::store);
            return RecordFn{};
        });
        game.run_slot(GameSlot::after_depth, images.depth, std::nullopt, hook);

        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &pass) {
            images.hdr = pass.create(sampled_image("hdr"));
            images.hdr = pass.color(images.hdr, LoadOp::clear, StoreOp::store);
            return RecordFn{};
        });
        game.run_slot(GameSlot::after_lighting, images.depth, images.hdr, hook);
        game.run_slot(GameSlot::before_composite, images.depth, images.hdr, hook);

        auto const colour = game.scene_colour().value_or(images.hdr);
        graph.add_pass("composition", PassType::raster, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const sampled = pass.read(colour, Use::sampled, fragment);
            swapchain = pass.color(swapchain, LoadOp::dont_care, StoreOp::store);
            return RecordFn{};
        });
        return images;
    }

    auto services_for(FrameGraph &graph) -> GameGraphServices {
        return GameGraphServices{
                .scene_extent = {64, 64},
                .bindless_index = [](std::uint32_t resource) { return 100U + resource; },
                .import_buffer = [&graph](GameBufferHandle) -> std::optional<BufferId> {
                    return graph.import_buffer({
                            .owner = Owner::game,
                            .debug_name = "game_buffer",
                            .buffer = {.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{1}),
                                       .address = 0xABC,
                                       .size = 256},
                    });
                },
        };
    }

    // One game compute pass in every slot: samples what the slot exports and writes an image of its own.
    auto game_pass_in_every_slot(GameGraph &game) -> void {
        auto const name = std::string{"game_"} + std::to_string(static_cast<int>(game.slot()));
        game.add_compute_pass(name, {.label = "game"}, [&](GameComputeBuilder &pass) {
            pass.side_effect();
            pass.queue(QueueAffinity::compute_preferred);
            if (auto const depth = game.scene_depth()) {
                [[maybe_unused]] auto const sampled = pass.sample(*depth);
            }
            if (auto const hdr = game.scene_hdr()) {
                [[maybe_unused]] auto const sampled = pass.sample(*hdr);
            }
            auto out = pass.create_image({.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "game_out"});
            [[maybe_unused]] auto const written = pass.write(out);
            if (game.slot() == GameSlot::before_composite) {
                game.replace_scene_colour(out);
            }
            return [](GameComputeContext &) {};
        });
    }

    auto problems_of(FrameGraph const &graph, QueueTopology const &topology) -> std::vector<std::string> {
        auto const compiled = compile(graph, topology);
        REQUIRE(compiled.has_value());
        return test::check_happens_before(graph.description(), *compiled, topology);
    }
}

TEST_SUITE("unit") {
    TEST_CASE("a game pass cannot write an engine-owned image") {
        auto graph = FrameGraph{};
        auto const hdr = graph.import_image({.debug_name = "hdr"});
        graph.add_pass("game_post", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const forged = pass.write(hdr, Use::storage_write, compute);
            return RecordFn{};
        });

        REQUIRE(graph.declaration_errors().size() == 1);
        CHECK(graph.declaration_errors().front().type == FrameGraphErrorType::foreign_write);
    }

    TEST_CASE("an engine pass may still write engine images, and a game pass may read them") {
        auto graph = FrameGraph{};
        auto hdr = graph.import_image({.debug_name = "hdr"});
        graph.add_pass("engine", PassType::compute, {}, [&](PassBuilder &pass) {
            hdr = pass.write(hdr, Use::storage_write, compute);
            return RecordFn{};
        });
        graph.add_pass("game_read", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            pass.side_effect();
            [[maybe_unused]] auto const sampled = pass.read(hdr, Use::sampled, compute);
            return RecordFn{};
        });

        CHECK(graph.declaration_errors().empty());
    }

    TEST_CASE("a game pass cannot use a transient in a way its views and format do not allow") {
        auto graph = FrameGraph{};
        graph.add_pass("game_storage", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            auto image = pass.create(sampled_image("sampled_only"));
            [[maybe_unused]] auto const written = pass.write(image, Use::storage_write, compute);
            auto depth = pass.create({.format = VK_FORMAT_D32_SFLOAT,
                                      .extent = {8, 8, 1},
                                      .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                                      .debug_name = "depth"});
            [[maybe_unused]] auto const depth_written = pass.write(depth, Use::storage_write, compute);
            return RecordFn{};
        });

        REQUIRE(graph.declaration_errors().size() == 2);
        CHECK(graph.declaration_errors()[0].type == FrameGraphErrorType::unsupported_usage);
        CHECK(graph.declaration_errors()[1].type == FrameGraphErrorType::unsupported_usage);
    }

    TEST_CASE("rolling back a game declaration restores the engine graph exactly") {
        auto graph = FrameGraph{};
        auto const target = graph.import_image({.debug_name = "target"});
        graph.add_pass("engine", PassType::compute, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const sampled = pass.read(target, Use::sampled, compute);
            pass.side_effect();
            return RecordFn{};
        });
        auto const before = declaration_hash(graph.description(), QueueTopology{}, {});

        auto const mark = graph.checkpoint();
        graph.add_pass("game", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const forged = pass.write(target, Use::storage_write, compute);
            [[maybe_unused]] auto const own = pass.create(sampled_image("own"));
            return RecordFn{};
        });
        REQUIRE_FALSE(graph.declaration_errors().empty());

        auto const dropped = graph.rollback(mark);
        CHECK(dropped.size() == 1);
        CHECK(graph.declaration_errors().empty());
        CHECK(graph.records().size() == 1);
        CHECK(declaration_hash(graph.description(), QueueTopology{}, {}) == before);
        CHECK(compile(graph, QueueTopology{}).has_value());
    }

    TEST_CASE("a game slot that forges a write is dropped and the engine frame still compiles") {
        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        auto hdr = std::optional<ImageId>{};

        declare_frame(graph, game, [&](GameGraph &g) {
            if (g.slot() != GameSlot::after_lighting) {
                return;
            }
            graph.add_pass("game_forged", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
                hdr = graph.find_image("hdr");
                [[maybe_unused]] auto const forged = pass.write(*hdr, Use::storage_write, compute);
                return RecordFn{};
            });
        });

        CHECK(game.rolled_back_slots() == 1);
        CHECK(graph.declaration_errors().empty());
        for (auto const &pass: graph.description().passes) {
            CHECK(pass.owner == Owner::engine);
        }
        CHECK(compile(graph, QueueTopology{}).has_value());
    }

    TEST_CASE("the frame with a game pass in every slot is sound on every topology") {
        for (auto const &topology: topologies()) {
            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};

            declare_frame(graph, game, game_pass_in_every_slot);

            CHECK(game.rolled_back_slots() == 0);
            REQUIRE(graph.declaration_errors().empty());
            auto const problems = problems_of(graph, topology);
            for (auto const &problem: problems) {
                MESSAGE(problem);
            }
            CHECK(problems.empty());
        }
    }

    TEST_CASE("composition declares its read of the scene colour, so the barrier and lifetime exist") {
        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [](GameGraph &) {});

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const &desc = graph.description();
        auto const hdr = graph.find_image("hdr")->index;
        auto found = false;
        for (auto const &batch: compiled->batches) {
            for (auto const &pass: batch.passes) {
                if (desc.passes[pass.pass].name != "composition") {
                    continue;
                }
                for (auto const &barrier: pass.before.images) {
                    found = found || (barrier.resource == hdr &&
                                      barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
            }
        }
        CHECK(found);
    }

    TEST_CASE("the scene colour a game replaces is what composition reads") {
        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [&](GameGraph &g) {
            if (g.slot() == GameSlot::before_composite) {
                game_pass_in_every_slot(g);
            }
        });

        REQUIRE(game.scene_colour().has_value());
        auto const &composition = graph.description().passes.back();
        REQUIRE(composition.name == "composition");
        CHECK(composition.accesses.front().resource == game.scene_colour()->index);
    }

    TEST_CASE("lint reports what a game pass silently loses") {
        auto graph = FrameGraph{};
        graph.add_pass("game_culled", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            auto image = pass.create({.format = VK_FORMAT_R8G8B8A8_UNORM,
                                      .extent = {8, 8, 1},
                                      .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                                      .debug_name = "tmp"});
            image = pass.write(image, Use::storage_write, compute);
            return RecordFn{};
        });
        graph.add_pass("game_unread", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            pass.side_effect();
            auto image = pass.create({.format = VK_FORMAT_R8G8B8A8_UNORM,
                                      .extent = {8, 8, 1},
                                      .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                                      .debug_name = "unread"});
            image = pass.write(image, Use::storage_write, compute);
            return RecordFn{};
        });
        graph.add_pass("game_empty", PassType::compute, {}, Owner::game, [&](PassBuilder &pass) {
            pass.side_effect();
            return RecordFn{};
        });
        REQUIRE(graph.declaration_errors().empty());

        auto const compiled = compile(graph, QueueTopology{});
        REQUIRE(compiled.has_value());
        auto const findings = lint(graph.description(), *compiled);
        REQUIRE(findings.size() == 3);
        CHECK(findings[0].kind == LintKind::culled_game_pass);
        CHECK(findings[1].kind == LintKind::unread_game_write);
        CHECK(findings[2].kind == LintKind::empty_game_pass);
    }

    TEST_CASE("a record function can only look up what its own pass declared") {
        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        auto leaked_image = std::optional<DeclaredImage>{};
        auto leaked_buffer = std::optional<DeclaredBuffer>{};
        auto own_index = std::uint32_t{0};
        auto own_address = VkDeviceAddress{0};

        declare_frame(graph, game, [&](GameGraph &g) {
            if (g.slot() != GameSlot::after_depth) {
                return;
            }
            auto buffer = g.import(GameBufferHandle{.index = 0, .generation = 1});
            g.add_compute_pass("game_owner", {.label = "owner"}, [&](GameComputeBuilder &pass) {
                pass.side_effect();
                auto image = pass.create_image({.format = VK_FORMAT_R16G16B16A16_SFLOAT, .name = "owned"});
                auto const declared_image = pass.write(image);
                auto const declared_buffer = pass.read_write(buffer);
                leaked_image = declared_image;
                leaked_buffer = declared_buffer;
                return [&, declared_image, declared_buffer](GameComputeContext &context) {
                    own_index = context.storage_index(declared_image);
                    own_address = context.address(declared_buffer);
                };
            });
            g.add_compute_pass("game_thief", {.label = "thief"}, [&](GameComputeBuilder &pass) {
                pass.side_effect();
                [[maybe_unused]] auto const depth = pass.sample(*g.scene_depth());
                return [&](GameComputeContext &context) {
                    CHECK(context.sampled_index(*leaked_image) == 0);
                    CHECK_FALSE(context.violation().empty());
                    CHECK(context.address(*leaked_buffer) == 0);
                };
            });
        });
        REQUIRE(game.rolled_back_slots() == 0);

        auto resources = physical_resources_of(graph.description());
        for (auto &buffer: resources.buffers) {
            buffer.address = 0xABC;
            buffer.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{1});
        }
        auto const &passes = graph.description().passes;
        for (auto index = std::size_t{0}; index < passes.size(); ++index) {
            if (passes[index].owner != Owner::game) {
                continue;
            }
            auto context = PassContext{.resources = &resources, .accesses = passes[index].accesses};
            graph.records()[index](context);
        }

        CHECK(own_index >= 100U);
        CHECK(own_address == 0xABC);
        CHECK(game.problems().size() == 1);
    }

    TEST_CASE("dispatch sizes are rounded up and refused beyond the device limit") {
        CHECK(dispatch_group_count(65536, 64, 65535) == 1024U);
        CHECK(dispatch_group_count(65, 64, 65535) == 2U);
        CHECK_FALSE(dispatch_group_count(0, 64, 65535).has_value());
        CHECK_FALSE(dispatch_group_count(64, 0, 65535).has_value());
        CHECK_FALSE(dispatch_group_count(64U * 65536U, 64, 65535).has_value());
    }

    TEST_CASE("forward reads are only accepted before the forward pass is declared") {
        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};

        declare_frame(graph, game, [&](GameGraph &g) {
            auto buffer = g.import(GameBufferHandle{.index = 0, .generation = 1});
            g.add_compute_pass("game_writer_" + std::to_string(static_cast<int>(g.slot())), {.label = "writer"}, [&](GameComputeBuilder &pass) {
                [[maybe_unused]] auto const state = pass.read_write(buffer);
                return [](GameComputeContext &) {};
            });
            g.scene_overlay_reads(buffer, stages_of(ShaderStage::vertex));
            g.scene_overlay_reads(buffer, stages_of(ShaderStage::fragment));
        });

        // frame_start and after_depth are fine (merged into one read); after_lighting and before_composite are not.
        CHECK(game.rolled_back_slots() == 2);
        REQUIRE(game.forward_reads().size() == 1);
        CHECK(game.forward_reads().front().stages ==
              (stages_of(ShaderStage::vertex) | stages_of(ShaderStage::fragment)));
    }
}
