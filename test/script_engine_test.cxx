#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "core/fly_string.hxx"
#include "core/transform.hxx"
#include "rendering/entity.hxx"
#include "scripting/script_engine.hxx"

namespace {
    // A bare registry plus a revision counter the test controls: the seam that keeps these tests free of Scene,
    // Renderer and Vulkan.
    struct TestWorld {
        entt::registry registry;
        std::uint64_t revision = 1;

        [[nodiscard]] auto world() -> ScriptWorld { return {.registry = &registry, .hierarchy_revision = revision}; }

        // Meta + Transform (+ Parent), and a revision bump, as Scene's hierarchy signals would do.
        auto spawn(std::string_view name, entt::entity parent = entt::null) -> entt::entity {
            auto const entity = registry.create();
            registry.emplace<Components::Meta>(entity, Components::Meta{.name = FlyString{name}});
            registry.emplace<Components::Transform>(entity);
            if (parent != entt::null) {
                registry.emplace<Components::Parent>(entity, Components::Parent{.entity = parent});
            }
            ++revision;
            return entity;
        }

        [[nodiscard]] auto position(entt::entity entity) const -> glm::vec3 {
            return registry.get<Components::Transform>(entity).position;
        }
    };

    // Generous enough for Debug and sanitizer builds; the timeout tests set their own.
    [[nodiscard]] auto relaxed_settings() -> ScriptEngineSettings {
        return ScriptEngineSettings{.timeout = std::chrono::milliseconds{5000}};
    }

    [[nodiscard]] auto make_engine(ScriptEngineSettings const &settings = relaxed_settings()) -> ScriptEngine {
        auto engine = ScriptEngine::create(settings);
        REQUIRE(engine.has_value());
        return std::move(*engine);
    }

    [[nodiscard]] auto error_kind(ScriptRunReport const &report) -> std::optional<ScriptErrorKind> {
        if (!report.error.has_value()) {
            return std::nullopt;
        }
        return report.error->kind;
    }

    [[nodiscard]] auto error_message(ScriptRunReport const &report) -> std::string {
        return report.error.has_value() ? report.error->message : std::string{};
    }

    [[nodiscard]] auto error_line(ScriptRunReport const &report) -> std::optional<std::int32_t> {
        return report.error.has_value() ? report.error->line : std::nullopt;
    }

    [[nodiscard]] auto contains(std::string_view text, std::string_view part) -> bool {
        return text.find(part) != std::string_view::npos;
    }

    [[nodiscard]] auto within(glm::vec3 value, float low, float high) -> bool {
        return value.x >= low && value.x <= high && value.y >= low && value.y <= high && value.z >= low &&
               value.z <= high;
    }

    struct UpdateCounter {
        int count = 0;
        auto on_update(entt::registry & /*registry*/, entt::entity /*entity*/) -> void { ++count; }
    };

    struct DestroyOnUpdate {
        entt::entity target = entt::null;
        auto on_update(entt::registry &registry, entt::entity /*entity*/) -> void {
            if (registry.valid(target)) {
                registry.destroy(target);
            }
        }
    };

    struct ReentrantRun {
        ScriptEngine *engine = nullptr;
        ScriptWorld world{};
        bool called = false;
        std::optional<ScriptErrorKind> kind;

        auto on_update(entt::registry & /*registry*/, entt::entity /*entity*/) -> void {
            called = true;
            kind = error_kind(engine->run("return 1", world));
        }
    };

    constexpr std::string_view user_example = "e = scene.get_entity(\"Helmets\"); for _, c in "
                                              "ipairs(e.get_children_or_empty()) do c.get_transform().translation = "
                                              "Vec3.random(-30, 30) end";
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("ScriptEngine: the example script moves every child of Helmets") {
        TestWorld test;
        auto const helmets = test.spawn("Helmets");
        auto const first = test.spawn("Helmet 1", helmets);
        auto const second = test.spawn("Helmet 2", helmets);
        auto const third = test.spawn("Helmet 3", helmets);
        auto const bystander = test.spawn("Bystander");
        auto engine = make_engine();

        auto const report = engine.run(user_example, test.world());

        REQUIRE_MESSAGE(report.ok(), error_message(report));
        CHECK(report.transforms_written == 3);
        CHECK(within(test.position(first), -30.0F, 30.0F));
        CHECK(within(test.position(second), -30.0F, 30.0F));
        CHECK(within(test.position(third), -30.0F, 30.0F));
        CHECK(test.position(bystander) == glm::vec3{0.0F});
        CHECK(test.position(helmets) == glm::vec3{0.0F});
    }

    TEST_CASE("ScriptEngine: a translation write fires on_update<Transform>; a read does not") {
        TestWorld test;
        auto const entity = test.spawn("A");
        UpdateCounter counter;
        test.registry.on_update<Components::Transform>().connect<&UpdateCounter::on_update>(counter);
        auto engine = make_engine();

        auto const read = engine.run("local p = scene.get_entity('A').get_transform().translation", test.world());
        REQUIRE_MESSAGE(read.ok(), error_message(read));
        CHECK(counter.count == 0);
        CHECK(read.transforms_written == 0);

        auto const write =
                engine.run("scene.get_entity('A').get_transform().translation = Vec3(1, 2, 3)", test.world());
        REQUIRE_MESSAGE(write.ok(), error_message(write));
        CHECK(counter.count == 1);
        CHECK(write.transforms_written == 1);
        CHECK(test.position(entity) == glm::vec3{1.0F, 2.0F, 3.0F});
    }

    TEST_CASE("ScriptEngine: syntax errors report their line") {
        TestWorld test;
        auto engine = make_engine();

        auto const bad_token = engine.run("x = = 1", test.world());
        CHECK(error_kind(bad_token) == ScriptErrorKind::syntax);
        CHECK(error_line(bad_token) == 1);

        auto const unfinished = engine.run("for i=1,2 do\n", test.world());
        CHECK(error_kind(unfinished) == ScriptErrorKind::syntax);
        CHECK(error_line(unfinished) == 2);
        CHECK(contains(error_message(unfinished), "<eof>"));

        auto const binary = engine.run("\x1bLua", test.world());
        CHECK(error_kind(binary) == ScriptErrorKind::syntax);
        CHECK(contains(error_message(binary), "binary"));
    }

    TEST_CASE("ScriptEngine: runtime errors report their line") {
        TestWorld test;
        auto engine = make_engine();

        auto const report = engine.run("local a = 1\nlocal b = nil\nb.c = 1", test.world());
        CHECK(error_kind(report) == ScriptErrorKind::runtime);
        CHECK(error_line(report) == 3);
    }

    TEST_CASE("ScriptEngine: an infinite loop times out and the engine stays usable") {
        TestWorld test;
        auto engine = make_engine(ScriptEngineSettings{.timeout = std::chrono::milliseconds{50}});

        auto const report = engine.run("while true do end", test.world());
        CHECK(error_kind(report) == ScriptErrorKind::timeout);
        CHECK(error_line(report) == 1);
        CHECK(report.duration >= std::chrono::milliseconds{50});
        CHECK(report.duration < std::chrono::seconds{2});

        auto const after = engine.run("return 1", test.world());
        CHECK_MESSAGE(after.ok(), error_message(after));

        auto const swallowed = engine.run("while true do pcall(function() while true do end end) end", test.world());
        CHECK(error_kind(swallowed) == ScriptErrorKind::timeout);
    }

    TEST_CASE("ScriptEngine: allocations past the memory limit fail as memory errors") {
        TestWorld test;
        auto settings = relaxed_settings();
        settings.memory_limit_bytes = 8ULL * 1024 * 1024;
        auto engine = make_engine(settings);

        auto const doubling = engine.run("local s = string.rep('x', 1024) while true do s = s .. s end", test.world());
        CHECK(error_kind(doubling) == ScriptErrorKind::memory);

        auto const huge = engine.run("return string.rep('x', 1 << 30)", test.world());
        CHECK(error_kind(huge) == ScriptErrorKind::memory);

        auto const after = engine.run("return 1", test.world());
        CHECK_MESSAGE(after.ok(), error_message(after));
        CHECK(engine.stats().lua_bytes_in_use < settings.memory_limit_bytes);
    }

    TEST_CASE("ScriptEngine: unbounded recursion is a stack overflow runtime error") {
        TestWorld test;
        auto settings = relaxed_settings();
        settings.memory_limit_bytes = 512ULL * 1024 * 1024;
        auto engine = make_engine(settings);

        auto const report = engine.run("local function f(n) return 1 + f(n + 1) end return f(1)", test.world());
        CHECK(error_kind(report) == ScriptErrorKind::runtime);
        CHECK(contains(error_message(report), "stack overflow"));
    }

    TEST_CASE("ScriptEngine: the sandbox hides dangerous globals and keeps libraries read-only") {
        TestWorld test;
        auto engine = make_engine();

        auto const hidden = engine.run(
                "assert(os == nil and io == nil and require == nil and load == nil and loadstring == nil and "
                "dofile == nil and loadfile == nil and debug == nil and collectgarbage == nil and package == nil and "
                "coroutine == nil and _G == nil)",
                test.world());
        CHECK_MESSAGE(hidden.ok(), error_message(hidden));

        auto const write = engine.run("string.rep = nil", test.world());
        CHECK(error_kind(write) == ScriptErrorKind::runtime);
        CHECK(contains(error_message(write), "read-only"));

        auto const raw_write = engine.run("rawset(string, 'rep', nil)", test.world());
        CHECK(error_kind(raw_write) == ScriptErrorKind::runtime);
        auto const still_there = engine.run("assert(string.rep('a', 2) == 'aa')", test.world());
        CHECK_MESSAGE(still_there.ok(), error_message(still_there));

        auto const string_metatable = engine.run("return getmetatable('').__index", test.world());
        CHECK(error_kind(string_metatable) == ScriptErrorKind::runtime);

        auto const exit = engine.run("os.exit()", test.world());
        CHECK(error_kind(exit) == ScriptErrorKind::runtime);
        CHECK(contains(error_message(exit), "not available in scripts"));

        auto const assign = engine.run("x = 5", test.world());
        REQUIRE_MESSAGE(assign.ok(), error_message(assign));
        auto const fresh = engine.run("assert(x == nil)", test.world());
        CHECK_MESSAGE(fresh.ok(), error_message(fresh));
    }

    TEST_CASE("ScriptEngine: an entity destroyed mid-run is an invalid_entity error") {
        TestWorld test;
        static_cast<void>(test.spawn("A"));
        auto const b = test.spawn("B");
        DestroyOnUpdate destroyer{.target = b};
        test.registry.on_update<Components::Transform>().connect<&DestroyOnUpdate::on_update>(destroyer);
        auto engine = make_engine();

        auto const report = engine.run("local a, b = scene.get_entity('A'), scene.get_entity('B'); "
                                       "a.get_transform().translation = Vec3(1,2,3); b.get_transform()",
                                       test.world());
        CHECK(error_kind(report) == ScriptErrorKind::invalid_entity);
        CHECK(error_line(report) == 1);
        CHECK(report.transforms_written == 1);
    }

    TEST_CASE("ScriptEngine: an unknown name suggests the closest entity") {
        TestWorld test;
        static_cast<void>(test.spawn("Helmets"));
        auto engine = make_engine();

        auto const report = engine.run("scene.get_entity('Helmetz')", test.world());
        CHECK(error_kind(report) == ScriptErrorKind::invalid_entity);
        CHECK(contains(error_message(report), "did you mean 'Helmets'"));
    }

    TEST_CASE("ScriptEngine: an entity without children yields an empty table") {
        TestWorld test;
        static_cast<void>(test.spawn("Leaf"));
        auto engine = make_engine();

        auto const report = engine.run("assert(#scene.get_entity('Leaf').get_children_or_empty() == 0)", test.world());
        CHECK_MESSAGE(report.ok(), error_message(report));
    }

    TEST_CASE("ScriptEngine: compiled chunks are cached with LRU eviction") {
        TestWorld test;

        auto engine = make_engine();
        auto const first = engine.run("return 1", test.world());
        auto const second = engine.run("return 1", test.world());
        CHECK_FALSE(first.chunk_cache_hit);
        CHECK(second.chunk_cache_hit);

        auto settings = relaxed_settings();
        settings.chunk_cache_capacity = 2;
        auto small = make_engine(settings);
        static_cast<void>(small.run("return 'a'", test.world()));
        static_cast<void>(small.run("return 'b'", test.world()));
        static_cast<void>(small.run("return 'c'", test.world()));
        auto const evicted = small.run("return 'a'", test.world());
        CHECK_FALSE(evicted.chunk_cache_hit);
        CHECK(small.stats().chunk_cache_misses == 4);
        CHECK(small.stats().chunk_cache_hits == 0);
    }

    TEST_CASE("ScriptEngine: the entity index is rebuilt only when the hierarchy revision changes") {
        TestWorld test;
        auto const root = test.spawn("Root");
        auto const other = test.spawn("Other");
        auto const child = test.spawn("Child 1", root);
        static_cast<void>(test.spawn("Child 2", root));
        auto engine = make_engine();

        auto const count_is = [&](int expected) {
            auto const source =
                    std::format("assert(#scene.get_entity('Root').get_children_or_empty() == {})", expected);
            auto const report = engine.run(source, test.world());
            CHECK_MESSAGE(report.ok(), error_message(report));
        };

        count_is(2);
        CHECK(engine.stats().entity_index_rebuilds == 1);

        count_is(2);
        CHECK(engine.stats().entity_index_rebuilds == 1);

        // Without a revision bump the cached index is used, so the new child isn't seen yet.
        auto const unbumped = test.revision;
        static_cast<void>(test.spawn("Child 3", root));
        test.revision = unbumped;
        count_is(2);
        CHECK(engine.stats().entity_index_rebuilds == 1);

        ++test.revision;
        count_is(3);
        CHECK(engine.stats().entity_index_rebuilds == 2);

        test.registry.replace<Components::Parent>(child, Components::Parent{.entity = other});
        ++test.revision;
        count_is(2);
        auto const moved = engine.run("assert(#scene.get_entity('Other').get_children_or_empty() == 1)", test.world());
        CHECK_MESSAGE(moved.ok(), error_message(moved));
        CHECK(engine.stats().entity_index_rebuilds == 3);
    }

    TEST_CASE("ScriptEngine: a run started from inside a run reports busy") {
        TestWorld test;
        static_cast<void>(test.spawn("A"));
        auto engine = make_engine();
        ReentrantRun reentrant{.engine = &engine, .world = test.world()};
        test.registry.on_update<Components::Transform>().connect<&ReentrantRun::on_update>(reentrant);

        auto const report =
                engine.run("scene.get_entity('A').get_transform().translation = Vec3(1, 1, 1)", test.world());
        CHECK_MESSAGE(report.ok(), error_message(report));
        CHECK(reentrant.called);
        CHECK(reentrant.kind == ScriptErrorKind::busy);
        CHECK_FALSE(engine.is_running());
    }

    TEST_CASE("ScriptEngine: a misspelt API name suggests the right one") {
        TestWorld test;
        static_cast<void>(test.spawn("A"));
        auto engine = make_engine();

        auto const report = engine.run("scene.get_entiy('A')", test.world());
        CHECK(error_kind(report) == ScriptErrorKind::runtime);
        CHECK(contains(error_message(report), "did you mean 'get_entity'"));
    }

    TEST_CASE("ScriptEngine: Vec3 arithmetic, random and tostring") {
        TestWorld test;
        auto engine = make_engine();

        auto const arithmetic = engine.run("assert((Vec3(1,2,3) + Vec3(1,1,1)).x == 2)\n"
                                           "assert((2 * Vec3(1,2,3)).z == 6)\n"
                                           "assert((Vec3(1,2,3) * 2).y == 4)\n"
                                           "assert((Vec3(3,3,3) - Vec3(1,2,3)).x == 2)\n"
                                           "assert((-Vec3(1,2,3)).y == -2)\n"
                                           "assert(Vec3.new(4,5,6).z == 6)\n"
                                           "for i = 1, 100 do\n"
                                           "  local v = Vec3.random(-1, 1)\n"
                                           "  assert(v.x >= -1 and v.x <= 1 and v.y >= -1 and v.y <= 1 and "
                                           "v.z >= -1 and v.z <= 1)\n"
                                           "end\n"
                                           "assert(tostring(Vec3(1,2,3)):sub(1, 5) == 'Vec3(')",
                                           test.world());
        CHECK_MESSAGE(arithmetic.ok(), error_message(arithmetic));

        auto const bad_arguments = engine.run("return Vec3(1, 'a')", test.world());
        CHECK(error_kind(bad_arguments) == ScriptErrorKind::runtime);
    }

    TEST_CASE("ScriptEngine: print accepts any values and the run succeeds") {
        TestWorld test;
        auto engine = make_engine();

        auto const report = engine.run("print('hello', 1, nil, Vec3(1, 2, 3))", test.world());
        CHECK_MESSAGE(report.ok(), error_message(report));
        CHECK_FALSE(engine.is_running());
        CHECK(engine.stats().runs == 1);
    }
}
