#include <doctest/doctest.h>

#include "maths/constexpr_maths.hxx"

#include <atomic>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


TEST_SUITE("unit") {
    TEST_CASE("square_root of 2") {
        constexpr auto result = maths::square_root(2.0F);
        CHECK(result * result == doctest::Approx(2.0F));
    }

    TEST_CASE("square_root of 3") {
        constexpr auto result = maths::square_root(3.0F);
        CHECK(result * result == doctest::Approx(3.0F));
    }
}
