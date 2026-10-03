#include <doctest/doctest.h>

#include "rendering/frame_graph/compiler.hxx"

using namespace frame_graph;

namespace {

    constexpr auto fragment = static_cast<ShaderStages>(ShaderStage::fragment);
    constexpr auto compute_stage = static_cast<ShaderStages>(ShaderStage::compute);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto make_target(FrameGraph &graph, char const *name = "target", bool swapchain = false) -> ImageId {
        return graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
                .exit = {.layout =
                                 swapchain ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         .stages = swapchain ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         .access = swapchain ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .swapchain = swapchain,
                .debug_name = name,
        });
    }

    auto graphics_only() -> QueueTopology { return QueueTopology{}; }

    auto find_pass(CompiledGraph const &compiled, std::uint32_t pass) -> CompiledPass const * {
        for (auto const &batch: compiled.batches) {
            for (auto const &p: batch.passes) {
                if (p.pass == pass) {
                    return &p;
                }
            }
        }
        return nullptr;
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("RAW colour to sampled gives one image barrier") {
        auto graph = FrameGraph{};
        auto const target = make_target(graph, "target", true);
        auto const scratch = graph.add_pass("noop", PassType::compute, {}, [&](PassBuilder &) { return noop(); });
        (void) scratch;

        auto image = ImageId{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
            image = p.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("post", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const sampled = p.read(image, Use::sampled, fragment);
            [[maybe_unused]] auto const out = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *post = find_pass(*compiled, 2);
        REQUIRE(post != nullptr);
        REQUIRE(post->before.images.size() >= 1);
        auto const &barrier = post->before.images.front();
        CHECK(barrier.src_stages == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
        CHECK((barrier.src_access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0);
        CHECK(barrier.dst_stages == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        CHECK(barrier.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        CHECK(barrier.old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(barrier.new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    TEST_CASE("a second read in the same layout and stages costs no barrier") {
        auto graph = FrameGraph{};
        auto target = make_target(graph, "target", true);
        auto image = ImageId{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
            image = p.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("read_a", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, fragment);
            target = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("read_b", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, fragment);
            target = p.color(target, LoadOp::load, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *read_b = find_pass(*compiled, 2);
        REQUIRE(read_b != nullptr);
        for (auto const &barrier: read_b->before.images) {
            CHECK(barrier.resource != image.index);
        }
    }

    TEST_CASE("a later read from a new stage gets a layout-preserving barrier") {
        auto graph = FrameGraph{};
        auto const target = make_target(graph, "target", true);
        auto image = ImageId{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
            image = p.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("read_fragment", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, fragment);
            [[maybe_unused]] auto const out = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("read_compute", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *read_compute = find_pass(*compiled, 2);
        REQUIRE(read_compute != nullptr);
        auto found = false;
        for (auto const &barrier: read_compute->before.images) {
            if (barrier.resource == image.index) {
                found = true;
                CHECK(barrier.old_layout == barrier.new_layout);
                CHECK(barrier.dst_stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                // The first read's layout transition is itself a write, so the new stage chains off that read's stage.
                CHECK(barrier.src_stages == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
            }
        }
        CHECK(found);
    }

    TEST_CASE("WAR sampled then storage write is execution only plus the layout change") {
        auto graph = FrameGraph{};
        auto image = ImageId{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
            image = p.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("sample", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
            p.side_effect();
            return noop();
        });
        graph.add_pass("overwrite", PassType::compute, {}, [&](PassBuilder &p) {
            image = p.write(image, Use::storage_write, compute_stage);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *overwrite = find_pass(*compiled, 2);
        REQUIRE(overwrite != nullptr);
        REQUIRE(overwrite->before.images.size() == 1);
        auto const &barrier = overwrite->before.images.front();
        CHECK(barrier.src_access == VK_ACCESS_2_NONE);
        CHECK(barrier.old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
        CHECK(barrier.new_layout == VK_IMAGE_LAYOUT_GENERAL);
        CHECK((barrier.src_stages & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0);
    }

    TEST_CASE("buffer WAW gives a write to write barrier and is whole buffer") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({
                .entry = {.stages = VK_PIPELINE_STAGE_2_NONE},
                .exit = {.stages = VK_PIPELINE_STAGE_2_NONE},
                .debug_name = "buffer",
        });
        auto current = buffer;
        graph.add_pass("first", PassType::compute, {}, [&](PassBuilder &p) {
            current = p.write(current, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("second", PassType::compute, {}, [&](PassBuilder &p) {
            current = p.write(current, Use::shader_write, compute_stage);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *second = find_pass(*compiled, 1);
        REQUIRE(second != nullptr);
        REQUIRE(second->before.buffers.size() == 1);
        auto const &barrier = second->before.buffers.front();
        CHECK(barrier.src_access == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        CHECK(barrier.dst_access == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    TEST_CASE("the swapchain goes UNDEFINED to colour attachment then to present") {
        auto graph = FrameGraph{};
        auto const target = make_target(graph, "swapchain", true);
        graph.add_pass("present_draw", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const out = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const &batch = compiled->batches.front();
        REQUIRE(batch.passes.size() == 1);
        REQUIRE(batch.passes.front().before.images.size() == 1);
        CHECK(batch.passes.front().before.images.front().old_layout == VK_IMAGE_LAYOUT_UNDEFINED);
        CHECK(batch.passes.front().before.images.front().new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        REQUIRE(batch.epilogue.images.size() == 1);
        CHECK(batch.epilogue.images.front().old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(batch.epilogue.images.front().new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        CHECK(batch.waits_swapchain_acquire);
        CHECK(batch.signals_render_finished);
    }

    TEST_CASE("an import with a LOAD keeps its contents across the entry layout") {
        auto graph = FrameGraph{};
        auto const atlas = graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .exit = {.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT},
                .debug_name = "atlas",
        });
        graph.add_pass("shadow", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const next = p.write_depth(atlas, LoadOp::load, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const &pass = compiled->batches.front().passes.front();
        REQUIRE(pass.before.images.size() == 1);
        CHECK(pass.before.images.front().old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(pass.before.images.front().new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }

    TEST_CASE("unconsumed transients are culled transitively") {
        auto graph = FrameGraph{};
        auto const target = make_target(graph, "swapchain", true);
        auto dead = ImageId{};
        graph.add_pass("dead_a", PassType::raster, {}, [&](PassBuilder &p) {
            dead = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "dead"});
            dead = p.color(dead, LoadOp::clear, StoreOp::store);
            return noop();
        });
        auto dead_b = ImageId{};
        graph.add_pass("dead_b", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(dead, Use::sampled, fragment);
            dead_b = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "dead_b"});
            dead_b = p.color(dead_b, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("live", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const out = p.color(target, LoadOp::clear, StoreOp::store);
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        CHECK(compiled->pass_culled[0]);
        CHECK(compiled->pass_culled[1]);
        CHECK_FALSE(compiled->pass_culled[2]);
        CHECK(compiled->timestamp_passes[0].size() == 1);
        CHECK(find_pass(*compiled, 0) == nullptr);
    }

    TEST_CASE("side_effect keeps an otherwise dead pass and its producers") {
        auto graph = FrameGraph{};
        auto image = ImageId{};
        graph.add_pass("producer", PassType::raster, {}, [&](PassBuilder &p) {
            image = p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .debug_name = "image"});
            image = p.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("readback", PassType::transfer, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const s = p.read(image, Use::transfer_src);
            p.side_effect();
            return noop();
        });

        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        CHECK_FALSE(compiled->pass_culled[0]);
        CHECK_FALSE(compiled->pass_culled[1]);
        CHECK(compiled->timestamp_passes[0] == std::vector<std::uint32_t>{0, 1});
    }

    TEST_CASE("writing an import's final version roots a pass") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({.debug_name = "stats"});
        graph.add_pass("writer", PassType::compute, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const next = p.write(buffer, Use::shader_write, compute_stage);
            return noop();
        });
        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        CHECK_FALSE(compiled->pass_culled[0]);
    }

    TEST_CASE("tokens produce memory barriers only") {
        auto graph = FrameGraph{};
        auto const token = graph.import_token("overlay_prepare", {}, {});
        auto current = token;
        graph.add_pass("prepare", PassType::compute, {}, [&](PassBuilder &p) {
            current = p.write(current, Use::token_write);
            return noop();
        });
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &p) {
            [[maybe_unused]] auto const t = p.read(current, Use::token_read);
            p.side_effect();
            return noop();
        });
        auto const compiled = compile(graph, graphics_only());
        REQUIRE(compiled.has_value());
        auto const *draw = find_pass(*compiled, 1);
        REQUIRE(draw != nullptr);
        CHECK(draw->before.images.empty());
        CHECK(draw->before.buffers.empty());
        CHECK(draw->before.memory.size() == 1);
    }

    TEST_CASE("validation errors") {
        SUBCASE("duplicate pass name") {
            auto graph = FrameGraph{};
            graph.add_pass("same", PassType::compute, {}, [](PassBuilder &) { return noop(); });
            graph.add_pass("same", PassType::compute, {}, [](PassBuilder &) { return noop(); });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::duplicate_pass_name);
        }
        SUBCASE("stale version") {
            auto graph = FrameGraph{};
            auto const image = make_target(graph);
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const next = p.write(image, Use::storage_write, compute_stage);
                return noop();
            });
            graph.add_pass("b", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const old = p.read(image, Use::sampled, compute_stage);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::stale_version);
            CHECK(result.error().pass == "b");
            CHECK(result.error().resource == "target");
        }
        SUBCASE("transient read before write") {
            auto graph = FrameGraph{};
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                auto const image =
                        p.create({.format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .debug_name = "t"});
                [[maybe_unused]] auto const s = p.read(image, Use::sampled, compute_stage);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::read_before_write);
        }
        SUBCASE("write to read-only import") {
            auto graph = FrameGraph{};
            auto const image = graph.import_image({.read_only = true, .debug_name = "ro"});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const n = p.write(image, Use::storage_write, compute_stage);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::write_to_read_only_import);
        }
        SUBCASE("image use on a buffer") {
            auto graph = FrameGraph{};
            auto const buffer = graph.import_buffer({.debug_name = "b"});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const n = p.write(buffer, Use::storage_write, compute_stage);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::wrong_resource_kind);
        }
        SUBCASE("shader use without stages") {
            auto graph = FrameGraph{};
            auto const buffer = graph.import_buffer({.debug_name = "b"});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const n = p.read(buffer, Use::shader_read);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::missing_shader_stages);
        }
        SUBCASE("one resource used twice in a pass") {
            auto graph = FrameGraph{};
            auto const buffer = graph.import_buffer({.debug_name = "b"});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const r = p.read(buffer, Use::shader_read, compute_stage);
                [[maybe_unused]] auto const w = p.write(buffer, Use::shader_write, compute_stage);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::conflicting_use);
        }
        SUBCASE("attachment in a compute pass") {
            auto graph = FrameGraph{};
            auto const image = make_target(graph);
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const n = p.color(image, LoadOp::clear, StoreOp::store);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::attachment_in_non_raster_pass);
        }
        SUBCASE("raster pass with compute affinity") {
            auto graph = FrameGraph{};
            graph.add_pass("a", PassType::raster, {}, [&](PassBuilder &p) {
                p.queue(QueueAffinity::compute_required);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::raster_pass_with_compute_affinity);
        }
        SUBCASE("token on a compute pass") {
            auto graph = FrameGraph{};
            auto const token = graph.import_token("t", {}, {});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                p.queue(QueueAffinity::compute_preferred);
                [[maybe_unused]] auto const n = p.write(token, Use::token_write);
                return noop();
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::token_on_compute_pass);
        }
        SUBCASE("import exit on compute") {
            auto graph = FrameGraph{};
            [[maybe_unused]] auto const image = graph.import_image({
                    .exit = {.queue = LogicalQueue::compute},
                    .debug_name = "bad_exit",
            });
            auto const result = compile(graph, graphics_only());
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().type == FrameGraphErrorType::import_exit_not_on_graphics);
        }
    }

    TEST_CASE("the hash ignores record functions and tracks declarations") {
        auto const build = [](bool extra_read, int captured) {
            auto graph = FrameGraph{};
            auto const buffer = graph.import_buffer({.debug_name = "b"});
            graph.add_pass("a", PassType::compute, {}, [&](PassBuilder &p) {
                [[maybe_unused]] auto const n = p.write(buffer, Use::shader_write, compute_stage);
                return RecordFn{[captured](PassContext &) { (void) captured; }};
            });
            if (extra_read) {
                graph.add_pass("b", PassType::compute, {}, [&](PassBuilder &p) {
                    [[maybe_unused]] auto const r =
                            p.read(BufferId{.index = buffer.index, .generation = 2}, Use::shader_read, compute_stage);
                    p.side_effect();
                    return noop();
                });
            }
            return compile(graph, graphics_only())->hash;
        };
        CHECK(build(false, 1) == build(false, 2));
        CHECK(build(false, 1) != build(true, 1));
    }

    TEST_CASE("serialize mode replaces precise barriers with full ones") {
        auto graph = FrameGraph{};
        auto const buffer = graph.import_buffer({.debug_name = "b"});
        auto current = buffer;
        graph.add_pass("first", PassType::compute, {}, [&](PassBuilder &p) {
            current = p.write(current, Use::shader_write, compute_stage);
            return noop();
        });
        graph.add_pass("second", PassType::compute, {}, [&](PassBuilder &p) {
            current = p.write(current, Use::shader_write, compute_stage);
            return noop();
        });
        auto const compiled = compile(graph, graphics_only(), {.serialize = true});
        REQUIRE(compiled.has_value());
        auto const *second = find_pass(*compiled, 1);
        REQUIRE(second != nullptr);
        REQUIRE(second->before.buffers.size() == 1);
        CHECK(second->before.buffers.front().src_stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }
}
