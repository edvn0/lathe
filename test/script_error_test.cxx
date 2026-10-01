#include <doctest/doctest.h>

#include <array>
#include <string>
#include <string_view>

#include "scripting/script_error.hxx"

namespace {
    std::array<std::string, 5> const api_names{
            "get_children_or_empty", "get_entity", "get_transform", "random", "translation",
    };
} // namespace

TEST_SUITE("unit") {
    TEST_CASE("ScriptError: split_lua_error_location separates the line from the text") {
        auto const located = split_lua_error_location("script:12: boom", "script");
        REQUIRE(located.line.has_value());
        CHECK(*located.line == 12);
        CHECK(located.text == "boom");
    }

    TEST_CASE("ScriptError: split_lua_error_location leaves messages without a position alone") {
        auto const plain = split_lua_error_location("not enough memory", "script");
        CHECK_FALSE(plain.line.has_value());
        CHECK(plain.text == "not enough memory");

        auto const not_digits = split_lua_error_location("script:x: y", "script");
        CHECK_FALSE(not_digits.line.has_value());
        CHECK(not_digits.text == "script:x: y");

        auto const other_chunk = split_lua_error_location("other:3: boom", "script");
        CHECK_FALSE(other_chunk.line.has_value());

        auto const no_space = split_lua_error_location("script:3:boom", "script");
        CHECK_FALSE(no_space.line.has_value());
    }

    TEST_CASE("ScriptError: closest_name suggests near misses only") {
        auto const suggestion = closest_name("get_entiy", api_names);
        REQUIRE(suggestion.has_value());
        CHECK(*suggestion == "get_entity");

        // Adjacent transposition counts as one edit.
        auto const swapped = closest_name("get_entiyt", api_names);
        REQUIRE(swapped.has_value());
        CHECK(*swapped == "get_entity");

        CHECK_FALSE(closest_name("get_entity", api_names).has_value());
        CHECK_FALSE(closest_name("xyzzy", api_names).has_value());
        CHECK_FALSE(closest_name("", api_names).has_value());
    }

    TEST_CASE("ScriptError: with_name_suggestion explains sandboxed names") {
        auto const message = with_name_suggestion("attempt to call a nil value (global 'os')", api_names);
        CHECK(message == "attempt to call a nil value (global 'os') ('os' is not available in scripts)");

        auto const indexed = with_name_suggestion("attempt to index a nil value (global 'io')", api_names);
        CHECK(indexed.find("'io' is not available in scripts") != std::string::npos);
    }

    TEST_CASE("ScriptError: with_name_suggestion suggests a close API name") {
        auto const message = with_name_suggestion("attempt to call a nil value (field 'get_entiy')", api_names);
        CHECK(message == "attempt to call a nil value (field 'get_entiy') (did you mean 'get_entity'?)");
    }

    TEST_CASE("ScriptError: with_name_suggestion returns other messages unchanged") {
        CHECK(with_name_suggestion("boom", api_names) == "boom");
        CHECK(with_name_suggestion("attempt to call a nil value (field 'xyzzy')", api_names) ==
              "attempt to call a nil value (field 'xyzzy')");
        CHECK(with_name_suggestion("attempt to perform arithmetic on a nil value (local 'a')", api_names) ==
              "attempt to perform arithmetic on a nil value (local 'a')");
    }

    TEST_CASE("ScriptError: describe names the kind and the line") {
        CHECK(describe(ScriptError{.kind = ScriptErrorKind::runtime, .message = "boom", .line = 3}) ==
              "runtime error (line 3): boom");
        CHECK(describe(ScriptError{.kind = ScriptErrorKind::timeout, .message = "too slow"}) == "timeout: too slow");
        CHECK(script_error_kind_name(ScriptErrorKind::syntax) == "syntax error");
    }
}
