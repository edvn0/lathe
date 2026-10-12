#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

#include "frame_graph_test_support.hxx"
#include "gpu/image.hxx"
#include "rendering/effect_system.hxx"
#include "rendering/frame_graph/compiler.hxx"

using namespace frame_graph;

namespace {
    constexpr auto tint_manifest = R"({
        "shader": "effects/tint.slang",
        "group_size": [8, 8],
        "bindings": [
            {"name": "source", "image": "in", "source": "scene_colour"},
            {"name": "result", "image": "out", "format": "rgba16f", "replaces": "scene_colour"}
        ],
        "params": [
            {"name": "strength", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"name": "tint", "type": "float3", "default": [1.0, 0.5, 0.25]}
        ],
        "dispatch": {"per_pixel_of": "result"}
    })";

    constexpr auto histogram_manifest = R"({
        "shader": "effects/histogram.slang",
        "bindings": [
            {"name": "source", "image": "in", "source": "scene_depth"},
            {"name": "bins", "buffer": "read_write", "elements": 64},
            {"name": "shared", "buffer": "read"}
        ],
        "dispatch": {"elements_of": "bins"}
    })";

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

    auto make_system() -> EffectSystem {
        auto system = EffectSystem{};
        system.use_registrar([](EffectManifest const &) -> std::expected<GameComputeShader, std::string> {
            return GameComputeShader{};
        });
        return system;
    }

    // depth prepass -> forward -> composition, with the game's four slots, like Renderer::record_frame.
    auto declare_frame(FrameGraph &graph, GameGraph &game, std::function<void(GameGraph &)> const &hook) -> void {
        auto swapchain = graph.import_image({
                .entry = {.stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
                .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
                .swapchain = true,
                .debug_name = "swapchain",
        });
        auto const sampled = [](std::string_view name, VkFormat format) {
            return TransientImageDesc{.format = format,
                                      .extent = {64, 64, 1},
                                      .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                                      .debug_name = name};
        };

        auto depth = ImageId{};
        auto hdr = ImageId{};
        game.run_slot(GameSlot::frame_start, std::nullopt, std::nullopt, hook);
        graph.add_pass("depth_prepass", PassType::raster, {}, [&](PassBuilder &pass) {
            depth = pass.create(sampled("depth", VK_FORMAT_D32_SFLOAT));
            depth = pass.write_depth(depth, LoadOp::clear, StoreOp::store);
            return RecordFn{};
        });
        game.run_slot(GameSlot::after_depth, depth, std::nullopt, hook);
        graph.add_pass("forward", PassType::raster, {}, [&](PassBuilder &pass) {
            for (auto const &read: game.forward_reads()) {
                [[maybe_unused]] auto const declared = pass.read(
                        BufferId{.index = read.buffer.index, .generation = graph.latest_version(read.buffer.index)},
                        Use::shader_read, read.stages);
            }
            hdr = pass.create(sampled("hdr", VK_FORMAT_R16G16B16A16_SFLOAT));
            hdr = pass.color(hdr, LoadOp::clear, StoreOp::store);
            return RecordFn{};
        });
        game.run_slot(GameSlot::after_lighting, depth, hdr, hook);
        game.run_slot(GameSlot::before_composite, depth, hdr, hook);

        auto const colour = game.scene_colour().value_or(hdr);
        graph.add_pass("composition", PassType::raster, {}, [&](PassBuilder &pass) {
            [[maybe_unused]] auto const read = pass.read(colour, Use::sampled, stages_of(ShaderStage::fragment));
            swapchain = pass.color(swapchain, LoadOp::dont_care, StoreOp::store);
            return RecordFn{};
        });
    }

    auto effect_passes(FrameGraph const &graph) -> std::vector<PassDesc const *> {
        auto result = std::vector<PassDesc const *>{};
        for (auto const &pass: graph.description().passes) {
            if (pass.owner == Owner::game) {
                result.push_back(&pass);
            }
        }
        return result;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("an effect is a pass declared from its manifest, and its output replaces the scene colour") {
        for (auto const &topology: topologies()) {
            auto system = make_system();
            auto const shader = system.define("tint", tint_manifest);
            REQUIRE(shader.has_value());
            auto const effect = system.instance(*shader);
            REQUIRE(effect.has_value());
            REQUIRE(system.set(*effect, "strength", {.numbers = {0.25}}).has_value());
            REQUIRE(system.set(*effect, "tint", {.numbers = {1.0, 2.0, 3.0}}).has_value());
            REQUIRE(system.add(*effect, GameSlot::before_composite).has_value());

            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};
            declare_frame(graph, game, [&](GameGraph &g) { system.declare(g); });

            CHECK(game.rolled_back_slots() == 0);
            CHECK(game.rejected_units() == 0);
            REQUIRE(graph.declaration_errors().empty());
            CHECK(system.problem(*effect).empty());

            auto const passes = effect_passes(graph);
            REQUIRE(passes.size() == 1);
            CHECK(passes[0]->type == PassType::compute);
            CHECK(passes[0]->affinity == QueueAffinity::compute_preferred);
            REQUIRE(passes[0]->accesses.size() == 2);
            CHECK(passes[0]->accesses[0].use == Use::sampled);
            CHECK(passes[0]->accesses[1].use == Use::storage_write);
            REQUIRE(game.scene_colour().has_value());
            CHECK(game.scene_colour()->index == passes[0]->accesses[1].resource);

            auto const compiled = compile(graph, topology);
            REQUIRE(compiled.has_value());
            auto const problems = test::check_happens_before(graph.description(), *compiled, topology);
            for (auto const &problem: problems) {
                MESSAGE(problem);
            }
            CHECK(problems.empty());
        }
    }

    TEST_CASE("effects chain: the next one reads what the last replaced the scene colour with") {
        auto system = make_system();
        auto const shader = *system.define("tint", tint_manifest);
        auto const first = *system.instance(shader);
        auto const second = *system.instance(shader);
        REQUIRE(system.add(first, GameSlot::after_lighting).has_value());
        REQUIRE(system.add(second, GameSlot::before_composite).has_value());

        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [&](GameGraph &g) { system.declare(g); });

        auto const passes = effect_passes(graph);
        REQUIRE(passes.size() == 2);
        // The second reads the first's output, not the forward pass's image.
        CHECK(passes[1]->accesses[0].resource == passes[0]->accesses[1].resource);
        CHECK(game.scene_colour()->index == passes[1]->accesses[1].resource);
    }

    TEST_CASE("buffers: an effect's own persist, and one a script made is shared by name") {
        for (auto const &topology: topologies()) {
            auto system = make_system();
            auto const shader = *system.define("histogram", histogram_manifest);
            auto const effect = *system.instance(shader);
            auto const buffer = *system.create_buffer(128);

            // 'shared' has no size of its own, so it needs a buffer from the script.
            CHECK_FALSE(system.add(effect, GameSlot::after_depth).has_value());
            REQUIRE(system.set(effect, "shared", {.buffer = buffer}).has_value());
            REQUIRE(system.add(effect, GameSlot::after_depth).has_value());

            auto graph = FrameGraph{};
            auto memory = GameGraphMemory{};
            auto game = GameGraph{graph, memory, services_for(graph)};
            declare_frame(graph, game, [&](GameGraph &g) { system.declare(g); });

            CHECK(game.rejected_units() == 0);
            REQUIRE(graph.declaration_errors().empty());
            auto const &resources = graph.description().resources;
            CHECK(std::ranges::count_if(resources, [](ResourceDesc const &r) { return r.name.starts_with("effect.") && r.name.ends_with(".bins"); }) == 1);
            CHECK(std::ranges::count_if(resources, [&](ResourceDesc const &r) { return r.name == "effect_buffer." + std::to_string(buffer); }) == 1);

            auto const passes = effect_passes(graph);
            REQUIRE(passes.size() == 1);
            // Persistent buffers are shared by frames in flight, so the pass runs on the graphics queue.
            CHECK(passes[0]->affinity == QueueAffinity::graphics);

            auto const problems = test::check_happens_before(graph.description(), *compile(graph, topology), topology);
            CHECK(problems.empty());
        }
    }

    TEST_CASE("what a script can name is checked against the manifest") {
        auto system = make_system();
        auto const shader = *system.define("tint", tint_manifest);
        auto const effect = *system.instance(shader);

        auto const bad = [&](std::string_view name, EffectValue value, std::string_view mentions) {
            CAPTURE(name);
            auto const result = system.set(effect, name, value);
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().find(mentions) != std::string::npos);
        };

        bad("nonsense", {.numbers = {1.0}}, "strength");
        bad("strength", {.numbers = {2.0}}, "between");
        bad("strength", {.numbers = {1.0, 2.0}}, "needs 1");
        bad("strength", {.text = "high"}, "number");
        bad("tint", {.numbers = {1.0}}, "needs 3");
        bad("source", {.text = "the_moon"}, "scene_colour");
        bad("source", {.numbers = {1.0}}, "scene_colour");
        bad("result", {.text = "scene_colour"}, "output");

        CHECK(system.set(effect, "source", {.text = "scene_depth"}).has_value());

        // An input the slot cannot offer is refused when the effect is added, and when it is changed afterwards.
        CHECK(system.add(effect, GameSlot::frame_start).error().find("not available") != std::string::npos);
        CHECK(system.add(effect, GameSlot::after_depth).error().find("replaces the scene colour") != std::string::npos);
        CHECK(system.set(effect, "source", {.text = "scene_colour"}).has_value());
        CHECK_FALSE(system.add(effect, GameSlot::after_depth).has_value());
        REQUIRE(system.add(effect, GameSlot::after_lighting).has_value());
        CHECK_FALSE(system.add(effect, GameSlot::after_lighting).has_value());
        CHECK(system.set(effect, "source", {.text = "scene_depth"}).has_value());

        auto const histogram = *system.instance(*system.define("histogram", histogram_manifest));
        auto const small = *system.create_buffer(1);
        CHECK_FALSE(system.set(histogram, "bins", {.buffer = small}).has_value());
        CHECK_FALSE(system.set(histogram, "bins", {.buffer = 999}).has_value());
        CHECK_FALSE(system.set(histogram, "bins", {.text = "x"}).has_value());
        CHECK_FALSE(system.create_buffer(0).has_value());
        CHECK_FALSE(system.create_buffer(max_effect_buffer_elements + 1).has_value());
    }

    TEST_CASE("an effect the graph rejects is dropped alone, with the reason, and its siblings survive") {
        auto system = make_system();
        auto const shader = *system.define("tint", tint_manifest);
        auto const doomed = *system.instance(shader);
        auto const fine = *system.instance(shader);
        REQUIRE(system.add(doomed, GameSlot::before_composite).has_value());
        REQUIRE(system.add(fine, GameSlot::before_composite).has_value());

        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [&](GameGraph &g) {
            if (g.slot() == GameSlot::before_composite) {
                // A pass already has the doomed effect's name, so declaring it is rejected.
                graph.add_pass("effect." + std::to_string(doomed), PassType::compute, {}, Owner::game,
                               [](PassBuilder &pass) {
                                   pass.side_effect();
                                   return RecordFn{};
                               });
            }
            system.declare(g);
        });

        CHECK(game.rolled_back_slots() == 0);
        CHECK(game.rejected_units() == 1);
        CHECK(system.problem(doomed).find("duplicate_pass_name") != std::string::npos);
        CHECK(system.problem(fine).empty());
        CHECK(graph.declaration_errors().empty());

        auto const passes = effect_passes(graph);
        REQUIRE(passes.size() == 2);
        CHECK(passes[1]->name == "effect." + std::to_string(fine));
        CHECK(game.scene_colour()->index == passes[1]->accesses[1].resource);
        CHECK(compile(graph, QueueTopology{}).has_value());
    }

    TEST_CASE("an effect stays out of the frame once removed or destroyed") {
        auto system = make_system();
        auto const shader = *system.define("tint", tint_manifest);
        auto const removed = *system.instance(shader);
        auto const destroyed = *system.instance(shader);
        REQUIRE(system.add(removed, GameSlot::before_composite).has_value());
        REQUIRE(system.add(destroyed, GameSlot::before_composite).has_value());
        system.remove(removed);
        system.destroy(destroyed);

        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [&](GameGraph &g) { system.declare(g); });

        CHECK(effect_passes(graph).empty());
        CHECK_FALSE(system.exists(destroyed));
        CHECK(system.exists(removed));
        // It can go back in.
        CHECK(system.add(removed, GameSlot::after_lighting).has_value());
    }

    TEST_CASE("a buffer lives as long as the script or an effect holds it") {
        auto system = make_system();
        auto const shader = *system.define("histogram", histogram_manifest);
        auto const effect = *system.instance(shader);
        auto const buffer = *system.create_buffer(64);
        REQUIRE(system.set(effect, "shared", {.buffer = buffer}).has_value());

        system.release_buffer(buffer);
        CHECK(system.buffer_elements(buffer) == 64);
        system.destroy(effect);
        CHECK(system.buffer_elements(buffer) == 0);
    }

    TEST_CASE("a manifest is reloaded in place, keeping the values that still fit") {
        auto const directory = std::filesystem::temp_directory_path() / "lathe_effect_reload_test";
        std::filesystem::create_directories(directory);
        auto const file = directory / "tint.json";
        auto const write = [&](std::string_view text) {
            auto out = std::ofstream{file, std::ios::trunc};
            out << text;
        };

        write(tint_manifest);
        auto system = make_system();
        auto const shader = system.load_file(file, "tint.json");
        REQUIRE(shader.has_value());
        CHECK(system.load_file(file, "tint.json") == shader);
        auto const effect = *system.instance(*shader);
        REQUIRE(system.set(effect, "strength", {.numbers = {0.75}}).has_value());
        REQUIRE(system.set(effect, "tint", {.numbers = {1.0, 1.0, 1.0}}).has_value());
        REQUIRE(system.add(effect, GameSlot::before_composite).has_value());

        // A manifest that does not parse changes nothing.
        write("{ not json");
        CHECK_FALSE(system.reload(*shader).has_value());
        CHECK(system.manifest(*shader)->params.size() == 2);

        // strength keeps 0.75 (still a float within its new range), tint changes type and goes back to its default,
        // and a new param appears.
        write(R"({
            "shader": "effects/tint.slang",
            "bindings": [
                {"name": "source", "image": "in", "source": "scene_colour"},
                {"name": "result", "image": "out", "replaces": "scene_colour"}],
            "params": [
                {"name": "strength", "type": "float", "default": 0.1, "max": 1.0},
                {"name": "tint", "type": "float4", "default": [0.5, 0.5, 0.5, 1.0]},
                {"name": "bias", "type": "float", "default": 3.0, "max": 5.0}],
            "dispatch": {"per_pixel_of": "result"}})");
        REQUIRE(system.reload(*shader).has_value());
        CHECK(system.manifest(*shader)->params.size() == 3);

        auto graph = FrameGraph{};
        auto memory = GameGraphMemory{};
        auto game = GameGraph{graph, memory, services_for(graph)};
        declare_frame(graph, game, [&](GameGraph &g) { system.declare(g); });
        CHECK(game.rejected_units() == 0);
        CHECK(system.slot_of(effect) == GameSlot::before_composite);
        CHECK(effect_passes(graph).size() == 1);

        // A reload that makes the effect impossible in its slot takes it out, with the reason.
        write(R"({
            "shader": "effects/tint.slang",
            "bindings": [{"name": "source", "image": "in", "source": "scene_colour"},
                         {"name": "needs_buffer", "buffer": "read"}],
            "dispatch": {"threads": 64}})");
        REQUIRE(system.reload(*shader).has_value());
        CHECK_FALSE(system.slot_of(effect).has_value());
        CHECK(system.problem(effect).find("needs a buffer") != std::string::npos);

        std::filesystem::remove_all(directory);
    }
}
