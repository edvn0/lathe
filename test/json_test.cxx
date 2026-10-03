#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>

#include "core/json.hxx"

TEST_CASE("json parses every value type") {
    auto const parsed = parse_json(R"( {
        "number": -12.5e1, "integer": 42, "yes": true, "no": false, "nothing": null,
        "text": "a \"quoted\" \\ line\nnext é 😀",
        "list": [1, [2, 3], {"nested": "x"}], "empty_list": [], "empty_object": {}
    } )");
    REQUIRE(parsed.has_value());

    auto const &root = *parsed;
    CHECK(root.is_object());
    CHECK(root["number"].as_number() == -125.0);
    CHECK(root["integer"].as_number() == 42.0);
    CHECK(root["yes"].as_bool());
    CHECK_FALSE(root["no"].as_bool(true));
    CHECK(root["nothing"].is_null());
    CHECK(root["text"].as_string() == "a \"quoted\" \\ line\nnext \xc3\xa9 \xf0\x9f\x98\x80");

    auto const list = root["list"].items();
    REQUIRE(list.size() == 3);
    CHECK(list[1].items()[1].as_number() == 3.0);
    CHECK(list[2]["nested"].as_string() == "x");
    CHECK(root["empty_list"].is_array());
    CHECK(root["empty_object"].is_object());

    // Missing keys chain to null instead of failing.
    CHECK(root["missing"]["deeper"].is_null());
    CHECK(root["missing"].as_number(7.0) == 7.0);
    CHECK(root.find("missing") == nullptr);
}

TEST_CASE("json rejects malformed documents with an offset") {
    for (auto const *text: {"{\"a\": 1,}", "[1 2]", "{\"a\" 1}", "\"open", "tru", "1 2", "{\"a\": 01x}", ""}) {
        CAPTURE(text);
        auto const parsed = parse_json(text);
        CHECK_FALSE(parsed.has_value());
    }

    auto const error = parse_json("[1, 2, oops]");
    REQUIRE_FALSE(error.has_value());
    CHECK(error.error().offset == 7);
}

TEST_CASE("json writer output parses back, with inline containers and non-finite numbers as null") {
    JsonWriter writer;
    std::array const values{1.0F, 2.5F};

    writer.begin_object()
            .value("name", "a \"b\"")
            .value("count", std::uint32_t{3})
            .value("signed", -4)
            .value("flag", true)
            .value("ratio", 0.125, 3)
            .value("nan", std::numeric_limits<double>::quiet_NaN())
            .null("none")
            .numbers("values", values, 1);
    writer.begin_object("inline", true).value("x", 1).begin_array("y").value({}, 2).end_array().end_object();
    writer.end_object();

    auto const &text = writer.str();
    CHECK(text.find("\"values\": [1.0, 2.5]") != std::string::npos);
    CHECK(text.find("\"inline\": {\"x\": 1, \"y\": [2]}") != std::string::npos);
    CHECK(text.ends_with("}\n"));

    auto const parsed = parse_json(text);
    REQUIRE(parsed.has_value());
    CHECK((*parsed)["name"].as_string() == "a \"b\"");
    CHECK((*parsed)["count"].as_number() == 3.0);
    CHECK((*parsed)["signed"].as_number() == -4.0);
    CHECK((*parsed)["ratio"].as_number() == 0.125);
    CHECK((*parsed)["nan"].is_null());
    CHECK((*parsed)["values"].items().size() == 2);
}

TEST_CASE("json escapes control characters") { CHECK(json_escape("a\tb\x01") == "a\\tb\\u0001"); }
