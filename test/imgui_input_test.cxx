#include <doctest/doctest.h>

#include <imgui.h>

// With ImGui's default 16-bit characters, every code point past U+FFFF (every emoji) is replaced with U+FFFD as it is
// typed, so a text field stored a replacement character instead of the emoji. The build defines IMGUI_USE_WCHAR32;
// ImWchar is part of imgui.h, so a target that did not see the definition would disagree with the library.
static_assert(sizeof(ImWchar) == 4, "ImGui must be built with IMGUI_USE_WCHAR32");
static_assert(IM_UNICODE_CODEPOINT_MAX == 0x10FFFF, "ImGui must accept code points past the basic plane");

TEST_CASE("ImGui accepts characters past the basic plane") {
    CHECK(sizeof(ImWchar) == 4);
    CHECK(IM_UNICODE_CODEPOINT_MAX == 0x10FFFF);
}
