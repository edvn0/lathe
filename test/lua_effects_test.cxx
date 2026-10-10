#include <doctest/doctest.h>

#include <expected>
#include <string>

#include "app/lua_game.hxx"
#include "app/lua_game_api.hxx"
#include "rendering/effect_system.hxx"
#include "scripting/lua_runtime.hxx"

namespace {
    constexpr auto manifest = R"({
        "shader": "effects/tint.slang",
        "bindings": [
            {"name": "source", "image": "in", "source": "scene_colour"},
            {"name": "result", "image": "out", "replaces": "scene_colour"},
            {"name": "scratch", "buffer": "read_write"}
        ],
        "params": [
            {"name": "strength", "type": "float", "default": 0.5, "min": 0, "max": 1},
            {"name": "tint", "type": "float3", "default": [1, 0.5, 0.25]}
        ],
        "dispatch": {"per_pixel_of": "result"}
    })";

    struct Harness {
        EffectSystem effects;
        LuaGameHost host;
        LuaRuntime runtime = [] {
            auto created = LuaRuntime::create();
            REQUIRE(created.has_value());
            return std::move(*created);
        }();

        Harness() {
            effects.use_registrar([](EffectManifest const &) -> std::expected<GameComputeShader, std::string> {
                return GameComputeShader{};
            });
            host.host.effects = &effects;
            runtime.set_host(&host);
            open_lua_game_api(runtime);

            // `shader` is what compute.load would give for this manifest.
            auto const defined = effects.define("test", manifest);
            REQUIRE(defined.has_value());
            shader = *defined;
        }

        EffectShaderId shader = 0;

        // Runs a chunk; the error message, or an empty string when it ran.
        auto run(std::string const &source) -> std::string {
            auto *const state = runtime.state();
            auto const top = lua_gettop(state);

            if (luaL_loadstring(state, source.c_str()) != LUA_OK || lua_pcall(state, 0, 0, 0) != LUA_OK) {
                auto message = std::string{lua_tostring(state, -1)};
                lua_settop(state, top);
                return message;
            }

            lua_settop(state, top);
            return {};
        }
    };
}

TEST_SUITE("unit") {
    TEST_CASE("the compute library is there, and compute.load refuses what is not a manifest under assets") {
        auto harness = Harness{};

        CHECK(harness.run("assert(compute and compute.instance and compute.buffer and scene.add_effect and entity.add_particles and entity.set_particles and entity.remove_particles)").empty());
        CHECK(harness.run("local ok, e = pcall(compute.load, 'assets/shaders/effects/missing.json'); assert(not ok)").empty());
        CHECK(harness.run("local ok, e = pcall(compute.load, '../secret.json'); assert(not ok and e:find('under assets/'))").empty());
        CHECK(harness.run("local ok, e = pcall(compute.load, 'assets/shaders/x.txt'); assert(not ok and e:find('.json'))").empty());
        CHECK(harness.effects.exists(1) == false);
    }

    TEST_CASE("script errors are Lua errors with a reason, and never leave a half-made effect behind") {
        auto harness = Harness{};
        auto *const state = harness.runtime.state();

        // Expose the shader the way compute.load would, by id.
        struct Shader {
            EffectShaderId id;
        };
        auto *const shader = static_cast<Shader *>(lua_newuserdatauv(state, sizeof(Shader), 0));
        shader->id = harness.shader;
        luaL_setmetatable(state, "lathe.EffectShader");
        lua_setglobal(state, "shader");

        auto const fails = [&](std::string const &source, std::string const &mentions) {
            CAPTURE(source);
            auto const message = harness.run(source);
            CHECK_FALSE(message.empty());
            CHECK(message.find(mentions) != std::string::npos);
        };

        fails("compute.instance(shader, {strength = 5})", "between");
        fails("compute.instance(shader, {strenth = 0.5})", "strength");
        fails("compute.instance(shader, {tint = {1, 2}})", "needs 3");
        fails("compute.instance(shader, {tint = {1, 'x', 3}})", "numbers");
        fails("compute.instance(shader, {source = 'the_moon'})", "scene_colour");
        fails("compute.instance(shader, {source = print})", "only numbers");
        fails("compute.instance(shader, {[1] = 5})", "strings");
        fails("compute.instance(shader, 7)", "table");
        fails("compute.instance({}, {})", "EffectShader");
        fails("compute.buffer(0)", "floats");
        fails("compute.buffer(1e12)", "floats");
        fails("compute.buffer('many')", "number");
        // No instance survived those.
        for (auto id = EffectId{1}; id < 64; ++id) {
            CHECK_FALSE(harness.effects.exists(id));
        }

        CHECK(harness.run("fx = compute.instance(shader, {strength = 0.25, tint = {1, 1, 1}})").empty());
        auto id = EffectId{1};
        while (id < 64 && !harness.effects.exists(id)) {
            ++id;
        }
        REQUIRE(harness.effects.exists(id));
        fails("fx:set('strength', 2)", "between");
        fails("fx:set('result', 'x')", "output");
        fails("fx:set('scratch', 5)", "buffer");
        fails("scene.add_effect(fx, 'sideways')", "frame_start");
        fails("scene.add_effect(fx, 'before_composite')", "scratch");
        fails("scene.add_effect({}, 'before_composite')", "Effect");
        CHECK_FALSE(harness.effects.slot_of(id).has_value());

        CHECK(harness.run("buf = compute.buffer(16); fx:set('scratch', buf); scene.add_effect(fx, 'before_composite')").empty());
        CHECK(harness.effects.slot_of(id) == GameSlot::before_composite);
        fails("scene.add_effect(fx, 'before_composite')", "already");
        CHECK(harness.run("assert(fx:problem() == nil)").empty());

        CHECK(harness.run("scene.remove_effect(fx)").empty());
        CHECK_FALSE(harness.effects.slot_of(id).has_value());
    }

    TEST_CASE("an effect and its buffers go when the script's state does") {
        auto harness = Harness{};
        auto *const state = harness.runtime.state();

        struct Shader {
            EffectShaderId id;
        };
        auto *const shader = static_cast<Shader *>(lua_newuserdatauv(state, sizeof(Shader), 0));
        shader->id = harness.shader;
        luaL_setmetatable(state, "lathe.EffectShader");
        lua_setglobal(state, "shader");

        CHECK(harness.run("buf = compute.buffer(8); fx = compute.instance(shader, {scratch = buf}); buf = nil").empty());
        lua_gc(state, LUA_GCCOLLECT);
        // The effect still holds the buffer the script let go of.
        CHECK(harness.effects.buffer_elements(1) == 8);

        CHECK(harness.run("fx = nil").empty());
        lua_gc(state, LUA_GCCOLLECT);
        lua_gc(state, LUA_GCCOLLECT);
        CHECK_FALSE(harness.effects.exists(1));
        CHECK(harness.effects.buffer_elements(1) == 0);
    }
}
