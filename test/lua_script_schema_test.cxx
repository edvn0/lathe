#include <doctest/doctest.h>

#include <string>

#include "scene/components.hxx"
#include "serialisation/scene_codec.hxx"
#include "scripting/lua_runtime.hxx"
#include "scripting/lua_script_schema.hxx"

namespace {
    auto schema_of(std::string const &source) -> ScriptSchema {
        auto runtime = LuaRuntime::create();
        REQUIRE(runtime.has_value());

        auto *const state = runtime->state();

        REQUIRE(luaL_loadstring(state, source.c_str()) == LUA_OK);
        REQUIRE(runtime->protected_call(0, 1, true).ok);

        auto schema = read_script_schema(state, -1, "test");

        lua_settop(state, 0);

        return schema;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("a behaviour's properties keep their declared order, then the keyed ones by name") {
        auto const schema = schema_of(R"(return { properties = {
            { name = "speed", type = "number", default = 3, min = 0, max = 10 },
            { name = "alive", type = "boolean", default = true },
            zeta = { type = "string", default = "hi" },
            alpha = { type = "colour", default = {1, 0.5, 0.25} },
            kind = { type = "choice", options = {"a", "b"}, default = "b" },
            target = { type = "entity" },
            skin = { type = "texture", default = "scene_colour" },
        } })");

        REQUIRE(schema.error.empty());
        REQUIRE(schema.properties.size() == 7);
        CHECK(schema.properties[0].name == "speed");
        CHECK(schema.properties[1].name == "alive");
        CHECK(schema.properties[2].name == "alpha");
        CHECK(schema.properties[3].name == "kind");
        CHECK(schema.properties[6].name == "zeta");
        CHECK(std::get<double>(schema.properties[0].default_value) == 3.0);
        CHECK(schema.properties[0].has_range);
        CHECK(std::get<glm::vec4>(schema.properties[2].default_value) == glm::vec4{1.0F, 0.5F, 0.25F, 1.0F});
        CHECK(std::get<std::string>(schema.properties[3].default_value) == "b");
    }

    TEST_CASE("a script with no properties has none, and bad declarations are reported") {
        CHECK(schema_of("return {}").properties.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "x", type = "quaternion" } } })").error.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "x", type = "number", default = "no" } } })").error.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "entity", type = "number" } } })").error.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "x", type = "number", min = 5, max = 1 } } })").error.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "x", type = "choice", options = {} } } })").error.empty());
        CHECK_FALSE(schema_of(R"(return { properties = { { name = "x", type = "number" }, { name = "x", type = "number" } } })").error.empty());
    }

    TEST_CASE("values are clamped to the declared range and choices") {
        auto const schema = schema_of(R"(return { properties = {
            { name = "n", type = "number", min = 0, max = 1 },
            { name = "c", type = "choice", options = {"a", "b"} } } })");

        REQUIRE(schema.error.empty());
        CHECK(std::get<double>(script_value_sanitised(schema.properties[0], 5.0)) == 1.0);
        CHECK(std::get<std::string>(script_value_sanitised(schema.properties[1], std::string{"zzz"})) == "a");
        CHECK(std::get<double>(script_value_sanitised(schema.properties[0], true)) == 0.0);
    }

    TEST_CASE("script values are pushed as Lua data") {
        auto runtime = LuaRuntime::create();
        REQUIRE(runtime.has_value());
        auto *const state = runtime->state();

        push_script_value(state, ScriptValue{glm::vec3{1.0F, 2.0F, 3.0F}});
        lua_getfield(state, -1, "y");
        CHECK(lua_tonumber(state, -1) == 2.0);
        lua_pop(state, 1);
        lua_geti(state, -1, 3);
        CHECK(lua_tonumber(state, -1) == 3.0);
        lua_settop(state, 0);
    }

    TEST_CASE("a scene keeps a behaviour's values through encode and decode") {
        SceneDescription scene;
        scene.entities.resize(2);
        scene.behaviours.push_back(SceneBehaviourComponent{
                .entity = 1,
                .script = "spinner",
                .values = {
                        {"speed", SceneScriptValue{12.5}},
                        {"count", SceneScriptValue{std::int64_t{7}}},
                        {"on", SceneScriptValue{true}},
                        {"label", SceneScriptValue{std::string{"hello"}}},
                        {"pos", SceneScriptValue{glm::vec3{1.0F, 2.0F, 3.0F}}},
                        {"tint", SceneScriptValue{glm::vec4{0.1F, 0.2F, 0.3F, 0.4F}}},
                        {"target", SceneScriptValue{SceneEntityRef{.index = 0}}},
                }});

        REQUIRE(validate_scene(scene).has_value());

        auto const bytes = encode_scene(scene);
        auto const decoded = decode_scene(bytes);

        REQUIRE(decoded.has_value());
        REQUIRE(decoded->behaviours.size() == 1);
        CHECK(decoded->behaviours[0].script == "spinner");
        REQUIRE(decoded->behaviours[0].values.size() == 7);
        CHECK(std::get<double>(decoded->behaviours[0].values[0].value) == 12.5);
        CHECK(std::get<std::int64_t>(decoded->behaviours[0].values[1].value) == 7);
        CHECK(std::get<std::string>(decoded->behaviours[0].values[3].value) == "hello");
        CHECK(std::get<SceneEntityRef>(decoded->behaviours[0].values[6].value).index == 0);

        scene.behaviours[0].values.push_back({"bad", SceneScriptValue{SceneEntityRef{.index = 9}}});
        CHECK_FALSE(validate_scene(scene).has_value());
    }
}
