#include <doctest/doctest.h>

#include "rendering/frame_graph/compiler.hxx"

using namespace frame_graph;

namespace {

    constexpr auto fragment = static_cast<ShaderStages>(ShaderStage::fragment);

    auto noop() -> RecordFn { return RecordFn{}; }

    auto transient(PassBuilder &pass, char const *name, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                   VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) -> ImageId {
        return pass.create({.format = format, .extent = {8, 8, 1}, .samples = samples, .debug_name = name});
    }

    auto swapchain_import(FrameGraph &graph) -> ImageId {
        return graph.import_image({
                .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });
    }

    auto clear_color(float value) -> VkClearValue {
        auto clear = VkClearValue{};
        clear.color.float32[0] = value;
        return clear;
    }

    // One pass that draws into a transient and a second that presents it, so everything is live.
    auto declare_frame(FrameGraph &graph, VkClearValue clear, LoadOp present_load = LoadOp::dont_care) -> void {
        auto const swapchain = swapchain_import(graph);
        auto image = ImageId{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &pass) {
            image = transient(pass, "hdr");
            image = pass.color(image, LoadOp::clear, StoreOp::store, clear);
            return noop();
        });
        graph.add_pass("present", PassType::raster, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const sampled = pass.read(image, Use::sampled, fragment);
            [[maybe_unused]] auto const out = pass.color(swapchain, present_load, StoreOp::store);
            return noop();
        });
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("colour and depth attachments are recorded with their ops and clear values") {
        auto graph = FrameGraph{};
        graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &pass) {
            auto const color = transient(pass, "color");
            auto const depth = transient(pass, "depth", VK_FORMAT_D32_SFLOAT);
            [[maybe_unused]] auto const c = pass.color(color, LoadOp::clear, StoreOp::store, clear_color(0.5F));
            [[maybe_unused]] auto const d = pass.write_depth(depth, LoadOp::load, StoreOp::dont_care);
            pass.render_area({.offset = {1, 2}, .extent = {3, 4}});
            pass.view_mask(0b11);
            return noop();
        });

        REQUIRE(graph.declaration_errors().empty());
        auto const &rendering = graph.description().passes.front().rendering;
        REQUIRE(rendering.has_value());
        REQUIRE(rendering->colors.size() == 1);
        CHECK(rendering->colors[0].load == LoadOp::clear);
        CHECK(rendering->colors[0].store == StoreOp::store);
        CHECK(rendering->colors[0].clear.color.float32[0] == doctest::Approx(0.5F));
        REQUIRE(rendering->depth.has_value());
        CHECK(rendering->depth->load == LoadOp::load);
        CHECK(rendering->depth->store == StoreOp::dont_care);
        CHECK(rendering->render_area.offset.x == 1);
        CHECK(rendering->render_area.extent.height == 4);
        CHECK(rendering->view_mask == 0b11);
    }

    TEST_CASE("a pass without attachments has no rendering description") {
        auto graph = FrameGraph{};
        graph.add_pass("compute", PassType::compute, {}, [&](PassBuilder &pass) {
            auto const image = transient(pass, "scratch");
            [[maybe_unused]] auto const written =
                    pass.write(image, Use::storage_write, static_cast<ShaderStages>(ShaderStage::compute));
            return noop();
        });
        CHECK_FALSE(graph.description().passes.front().rendering.has_value());
    }

    TEST_CASE("resolve attaches a target to the right attachment and writes it with a resolve use") {
        auto graph = FrameGraph{};
        auto resolved = ImageId{};
        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &pass) {
            auto const msaa = transient(pass, "msaa", VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_4_BIT);
            auto const depth = transient(pass, "depth", VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_4_BIT);
            resolved = transient(pass, "resolved");
            auto const color = pass.color(msaa, LoadOp::clear, StoreOp::dont_care);
            auto const z = pass.write_depth(depth, LoadOp::clear, StoreOp::dont_care);
            resolved = pass.resolve(color, resolved, VK_RESOLVE_MODE_AVERAGE_BIT);
            [[maybe_unused]] auto const unused = z;
            return noop();
        });

        REQUIRE(graph.declaration_errors().empty());
        auto const &pass = graph.description().passes.front();
        REQUIRE(pass.rendering.has_value());
        REQUIRE(pass.rendering->colors.size() == 1);
        REQUIRE(pass.rendering->colors[0].resolve.has_value());
        CHECK(pass.rendering->colors[0].resolve->resource == resolved.index);
        CHECK(pass.rendering->colors[0].resolve->mode == VK_RESOLVE_MODE_AVERAGE_BIT);
        CHECK_FALSE(pass.rendering->depth->resolve.has_value());

        auto const use =
                std::ranges::find_if(pass.accesses, [&](AccessDesc const &a) { return a.resource == resolved.index; });
        REQUIRE(use != pass.accesses.end());
        CHECK(use->use == Use::color_resolve);
        CHECK(use->discard);
        CHECK(use->produces);
    }

    TEST_CASE("a depth resolve uses the depth resolve use and mode") {
        auto graph = FrameGraph{};
        graph.add_pass("prepass", PassType::raster, {}, [&](PassBuilder &pass) {
            auto const msaa = transient(pass, "depth_msaa", VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_4_BIT);
            auto const single = transient(pass, "depth_resolved", VK_FORMAT_D32_SFLOAT);
            auto const depth = pass.write_depth(msaa, LoadOp::clear, StoreOp::store);
            [[maybe_unused]] auto const out = pass.resolve(depth, single, VK_RESOLVE_MODE_MIN_BIT);
            return noop();
        });

        REQUIRE(graph.declaration_errors().empty());
        auto const &rendering = graph.description().passes.front().rendering;
        REQUIRE(rendering.has_value());
        REQUIRE(rendering->depth.has_value());
        REQUIRE(rendering->depth->resolve.has_value());
        CHECK(rendering->depth->resolve->mode == VK_RESOLVE_MODE_MIN_BIT);
        CHECK(graph.description().passes.front().accesses.back().use == Use::depth_resolve);
    }

    TEST_CASE("resolving something that is not an attachment of the pass is an error") {
        auto graph = FrameGraph{};
        graph.add_pass("bad", PassType::raster, {}, [&](PassBuilder &pass) {
            auto const loose = transient(pass, "loose");
            auto const target = transient(pass, "target");
            [[maybe_unused]] auto const out = pass.resolve(loose, target);
            return noop();
        });
        REQUIRE_FALSE(graph.declaration_errors().empty());
        CHECK(graph.declaration_errors().front().type == FrameGraphErrorType::conflicting_use);
        CHECK_FALSE(graph.description().passes.front().rendering.has_value());
    }

    TEST_CASE("an attachment the builder rejects is not recorded") {
        auto graph = FrameGraph{};
        graph.add_pass("twice", PassType::raster, {}, [&](PassBuilder &pass) {
            auto const image = transient(pass, "image");
            [[maybe_unused]] auto const first = pass.color(image, LoadOp::clear, StoreOp::store);
            [[maybe_unused]] auto const second = pass.color(image, LoadOp::clear, StoreOp::store);
            return noop();
        });
        REQUIRE_FALSE(graph.declaration_errors().empty());
        CHECK(graph.description().passes.front().rendering->colors.size() == 1);
    }

    TEST_CASE("the plan cache skips the compile when the declaration repeats") {
        auto cache = PlanCache{};
        auto const topology = QueueTopology{};

        for (int frame = 0; frame < 3; ++frame) {
            auto graph = FrameGraph{};
            declare_frame(graph, clear_color(0.0F));
            auto const plan = cache.compile(graph, topology);
            REQUIRE(plan.has_value());
            CHECK((*plan)->batches.size() >= 1);
        }

        CHECK(cache.misses() == 1);
        CHECK(cache.hits() == 2);
    }

    TEST_CASE("clear values do not invalidate the plan cache, but load ops that discard do") {
        auto cache = PlanCache{};
        auto const topology = QueueTopology{};

        auto first = FrameGraph{};
        declare_frame(first, clear_color(0.0F));
        REQUIRE(cache.compile(first, topology).has_value());

        auto other_clear = FrameGraph{};
        declare_frame(other_clear, clear_color(1.0F));
        REQUIRE(cache.compile(other_clear, topology).has_value());
        CHECK(cache.hits() == 1);

        // Loading the swapchain instead of discarding it changes whether its contents are needed: a different plan.
        auto loaded = FrameGraph{};
        declare_frame(loaded, clear_color(0.0F), LoadOp::load);
        REQUIRE(cache.compile(loaded, topology).has_value());
        CHECK(cache.misses() == 2);
    }

    TEST_CASE("options and topology are part of the hash") {
        auto graph = FrameGraph{};
        declare_frame(graph, clear_color(0.0F));
        auto const topology = QueueTopology{};

        auto const base = declaration_hash(graph.description(), topology, {});
        CHECK(base == declaration_hash(graph.description(), topology, {}));
        CHECK(base != declaration_hash(graph.description(), topology, {.serialize = true}));
        CHECK(base != declaration_hash(graph.description(), topology, {.async_compute = false}));

        auto other = QueueTopology{};
        other.family[static_cast<std::size_t>(LogicalQueue::compute)] = 2;
        CHECK(base != declaration_hash(graph.description(), other, {}));
    }

    TEST_CASE("pinning a pass changes the hash") {
        auto const topology = QueueTopology{};
        auto make = [&](bool pinned) {
            auto graph = FrameGraph{};
            auto const swapchain = swapchain_import(graph);
            graph.add_pass("draw", PassType::raster, {}, [&](PassBuilder &pass) {
                if (pinned) {
                    pass.pinned();
                }
                [[maybe_unused]] auto const out = pass.color(swapchain, LoadOp::clear, StoreOp::store);
                return noop();
            });
            return declaration_hash(graph.description(), topology, {});
        };
        CHECK(make(false) != make(true));
    }

    TEST_CASE("a graph with a declaration error is never served from the cache") {
        auto cache = PlanCache{};
        auto const topology = QueueTopology{};

        auto good = FrameGraph{};
        declare_frame(good, clear_color(0.0F));
        REQUIRE(cache.compile(good, topology).has_value());

        auto bad = FrameGraph{};
        declare_frame(bad, clear_color(0.0F));
        bad.add_pass("draw", PassType::raster, {}, [&](PassBuilder &) { return noop(); }); // duplicate name
        CHECK_FALSE(cache.compile(bad, topology).has_value());

        // The failure dropped the cached plan, so the good graph compiles afresh.
        REQUIRE(cache.compile(good, topology).has_value());
        CHECK(cache.misses() == 2);
    }

    TEST_CASE("serialize widens every derived barrier to ALL_COMMANDS") {
        auto graph = FrameGraph{};
        auto a = ImageId{};
        auto const swapchain = swapchain_import(graph);
        graph.add_pass("first", PassType::raster, {}, [&](PassBuilder &pass) {
            a = pass.color(transient(pass, "a"), LoadOp::clear, StoreOp::store);
            return noop();
        });
        graph.add_pass("final", PassType::raster, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const ra = pass.read(a, Use::sampled, fragment);
            [[maybe_unused]] auto const out = pass.color(swapchain, LoadOp::dont_care, StoreOp::store);
            return noop();
        });

        auto all_commands_sources = [](CompiledGraph const &plan) {
            auto total = std::size_t{0};
            for (auto const &batch: plan.batches) {
                for (auto const &pass: batch.passes) {
                    for (auto const &barrier: pass.before.images) {
                        total += barrier.src_stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT ? 1U : 0U;
                    }
                }
            }
            return total;
        };

        auto const normal = compile(graph, QueueTopology{});
        auto const serialized = compile(graph, QueueTopology{}, {.serialize = true});
        REQUIRE(normal.has_value());
        REQUIRE(serialized.has_value());
        CHECK(all_commands_sources(*normal) == 0);
        CHECK(all_commands_sources(*serialized) >= 1);
    }
}
