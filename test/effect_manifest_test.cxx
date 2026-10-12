#include <doctest/doctest.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "rendering/effect_manifest.hxx"

namespace {
    constexpr auto valid_manifest = R"({
        "shader": "effects/blur.slang",
        "entry": "main_cs",
        "group_size": [8, 8],
        "bindings": [
            {"name": "source", "image": "in", "source": "scene_depth"},
            {"name": "result", "image": "out", "format": "rgba8"},
            {"name": "counts", "buffer": "read_write", "elements": 256}
        ],
        "params": [
            {"name": "radius", "type": "float", "default": 2.0, "min": 0.0, "max": 16.0},
            {"name": "tint", "type": "float3", "default": [1.0, 0.5, 0.25]},
            {"name": "taps", "type": "uint", "default": 4, "max": 64}
        ],
        "dispatch": {"per_pixel_of": "result"}
    })";

    auto problem_of(std::string const &text) -> std::string {
        auto const parsed = parse_effect_manifest(text);
        REQUIRE_FALSE(parsed.has_value());
        return parsed.error();
    }

    // The valid manifest with one piece replaced.
    auto with(std::string_view from, std::string_view to) -> std::string {
        auto text = std::string{valid_manifest};
        auto const at = text.find(from);
        REQUIRE(at != std::string::npos);
        text.replace(at, from.size(), to);
        return text;
    }
}

TEST_SUITE("unit") {
    TEST_CASE("a manifest declares bindings, params and a dispatch") {
        auto const parsed = parse_effect_manifest(valid_manifest);
        REQUIRE(parsed.has_value());

        CHECK(parsed->shader == "effects/blur.slang");
        CHECK(parsed->group_size == std::array<std::uint32_t, 2>{8, 8});
        REQUIRE(parsed->bindings.size() == 3);
        CHECK(std::get<EffectImageIn>(parsed->bindings[0].what).source == EffectSource::scene_depth);
        CHECK(std::get<EffectImageOut>(parsed->bindings[1].what).format == EffectFormat::rgba8);
        CHECK(std::get<EffectBufferBinding>(parsed->bindings[2].what).elements == 256);
        REQUIRE(parsed->params.size() == 3);
        CHECK(parsed->params[1].value[2] == doctest::Approx(0.25));
        CHECK(parsed->params[2].value[0] == 4.0);
        CHECK(parsed->dispatch.kind == EffectDispatch::Kind::per_pixel);
        CHECK(parsed->dispatch.of == "result");
        CHECK_FALSE(parsed->replaces_scene_colour());

        // uint, uint, (pad to 8) address 8..16, float 16, (pad to 32) float3 32..44, uint 44..48
        CHECK(parsed->push_bytes() == 48);
    }

    TEST_CASE("the effect that ships with the engine is a valid manifest") {
        auto file = std::ifstream{TEST_ASSETS_DIR "/assets/shaders/effects/tint.json"};
        REQUIRE(file.good());
        auto text = std::ostringstream{};
        text << file.rdbuf();

        auto const parsed = parse_effect_manifest(text.str());
        REQUIRE(parsed.has_value());
        CHECK(parsed->replaces_scene_colour());
        // source, result, strength, then a float3 on a 16-byte boundary: what tint.slang's reflection says.
        CHECK(parsed->push_bytes() == 28);
    }

    TEST_CASE("a bad manifest is rejected with the reason") {
        CHECK(problem_of("{").find("JSON") != std::string::npos);
        CHECK(problem_of("[]").find("object") != std::string::npos);
        CHECK(problem_of(with("\"dispatch\"", "\"disptach\"")).find("unknown key") != std::string::npos);

        for (auto const *shader: {"", "/etc/x.slang", "../x.slang", "a/../../x.slang", "x.txt", "a//b.slang", "a\\\\b.slang"}) {
            CAPTURE(shader);
            CHECK(problem_of(with("effects/blur.slang", shader)).find("shader must be") != std::string::npos);
        }

        CHECK(problem_of(with("\"name\": \"source\"", "\"name\": \"1source\"")).find("identifier") != std::string::npos);
        CHECK(problem_of(with("\"name\": \"radius\"", "\"name\": \"source\"")).find("twice") != std::string::npos);
        CHECK(problem_of(with("\"image\": \"in\"", "\"image\": \"sideways\"")).find("image") != std::string::npos);
        CHECK(problem_of(with("\"image\": \"in\"", "\"image\": \"in\", \"buffer\": \"read\"")).find("exactly one") != std::string::npos);
        CHECK(problem_of(with("scene_depth", "the_sky")).find("source") != std::string::npos);
        CHECK(problem_of(with("rgba8", "bc7")).find("format") != std::string::npos);
        CHECK(problem_of(with("\"read_write\"", "\"append\"")).find("buffer") != std::string::npos);
        CHECK(problem_of(with("\"elements\": 256", "\"elements\": 0")).find("elements") != std::string::npos);
        CHECK(problem_of(with("\"elements\": 256", "\"elements\": 1.5")).find("elements") != std::string::npos);
        CHECK(problem_of(with("\"elements\": 256", "\"elements\": 99999999999")).find("elements") != std::string::npos);
        CHECK(problem_of(with("\"float3\"", "\"float5\"")).find("type") != std::string::npos);
        CHECK(problem_of(with("\"group_size\": [8, 8]", "\"group_size\": [0, 8]")).find("group_size") != std::string::npos);
        CHECK(problem_of(with("\"group_size\": [8, 8]", "\"group_size\": [8, 8, 8]")).find("group_size") != std::string::npos);
        CHECK(problem_of(with("\"default\": 2.0", "\"default\": 20.0")).find("default") != std::string::npos);
        CHECK(problem_of(with("\"default\": [1.0, 0.5, 0.25]", "\"default\": [1.0, 0.5]")).find("default") != std::string::npos);
        CHECK(problem_of(with("\"default\": 4", "\"default\": 4.5")).find("whole") != std::string::npos);
        CHECK(problem_of(with("\"min\": 0.0, \"max\": 16.0", "\"min\": 8.0, \"max\": 1.0")).find("min is above max") != std::string::npos);
    }

    TEST_CASE("the dispatch rule has to name something that exists, of the right kind") {
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"per_pixel_of\": \"nowhere\"}")).find("per_pixel_of") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"per_pixel_of\": \"counts\"}")).find("image") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"elements_of\": \"result\"}")).find("buffer") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"threads\": 0}")).find("threads") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"threads\": 1e12}")).find("threads") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"threads\": 8, \"elements_of\": \"counts\"}")).find("dispatch") != std::string::npos);
        CHECK(problem_of(with("{\"per_pixel_of\": \"result\"}", "{\"fill\": 1}")).find("fill") != std::string::npos);

        auto const elements = parse_effect_manifest(with("{\"per_pixel_of\": \"result\"}", "{\"elements_of\": \"counts\"}"));
        REQUIRE(elements.has_value());
        CHECK(elements->dispatch.kind == EffectDispatch::Kind::elements);
        auto const threads = parse_effect_manifest(with("{\"per_pixel_of\": \"result\"}", "{\"threads\": 4096}"));
        REQUIRE(threads.has_value());
        CHECK(threads->dispatch.threads == 4096);
    }

    TEST_CASE("push constants that do not fit are rejected") {
        auto text = std::string{R"({"shader": "x.slang", "params": [)"};
        for (auto index = 0; index < 9; ++index) {
            text += std::string{index > 0 ? "," : ""} + "{\"name\": \"v" + std::to_string(index) + "\", \"type\": \"float4\"}";
        }
        text += R"(], "dispatch": {"threads": 64}})";
        CHECK(problem_of(text).find("push constants") != std::string::npos);

        CHECK(problem_of(R"({"shader": "x.slang", "dispatch": {"threads": 64}})").find("at least one") != std::string::npos);
    }

    TEST_CASE("only one binding can replace the scene colour") {
        auto const text = R"({"shader": "x.slang", "dispatch": {"per_pixel_of": "a"}, "bindings": [
            {"name": "a", "image": "out", "replaces": "scene_colour"},
            {"name": "b", "image": "out", "replaces": "scene_colour"}]})";
        CHECK(problem_of(text).find("only one") != std::string::npos);
        CHECK(problem_of(with("rgba8\"", "rgba8\", \"replaces\": \"scene_depth\"")).find("scene_colour") != std::string::npos);
    }

    TEST_CASE("a value a script gives is checked against the param's type and range") {
        auto const parsed = parse_effect_manifest(valid_manifest);
        REQUIRE(parsed.has_value());
        auto const &radius = parsed->params[0];
        auto const &tint = parsed->params[1];
        auto const &taps = parsed->params[2];

        auto const ok = std::array{4.0};
        CHECK(check_effect_value(radius, ok).has_value());
        CHECK((*check_effect_value(radius, ok))[0] == 4.0);

        constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
        for (auto const value: {nan, std::numeric_limits<double>::infinity(), -1.0, 17.0}) {
            CAPTURE(value);
            auto const numbers = std::array{value};
            CHECK_FALSE(check_effect_value(radius, numbers).has_value());
        }

        auto const wrong_count = std::array{1.0, 2.0};
        CHECK_FALSE(check_effect_value(radius, wrong_count).has_value());
        CHECK_FALSE(check_effect_value(tint, wrong_count).has_value());
        CHECK(check_effect_value(tint, std::array{1.0, 2.0, 3.0}).has_value());

        CHECK_FALSE(check_effect_value(taps, std::array{2.5}).has_value());
        CHECK_FALSE(check_effect_value(taps, std::array{-1.0}).has_value());
        CHECK_FALSE(check_effect_value(taps, std::array{65.0}).has_value());
        CHECK(check_effect_value(taps, std::array{64.0}).has_value());
    }
}
