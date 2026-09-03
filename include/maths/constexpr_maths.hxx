#pragma once

#include <concepts>
#include <cstdint>

namespace maths {

    namespace detail {
        constexpr std::uint32_t max_sqrt_iterations = 128;

        template<std::floating_point T, typename F, typename DF>
        constexpr auto newton_raphson(F &&f, DF &&df, T x) -> T {
            for (std::uint32_t i = 0; i < detail::max_sqrt_iterations; ++i) {
                auto const fx = f(x);
                auto const dfx = df(x);

                auto const next = x - fx / dfx;

                if (next == x) {
                    return next;
                }

                x = next;
            }

            return x;
        }

        template<std::floating_point T, typename F, typename DF, typename DDF>
        constexpr auto newton_raphson(F &&f, DF &&df, DDF &&ddf, T x) -> T {
            for (std::uint32_t i = 0; i < detail::max_sqrt_iterations; ++i) {
                auto const fx = f(x);
                auto const dfx = df(x);
                auto const ddfx = ddf(x);
                auto const numerator = T{2} * fx * dfx;
                auto const denominator = T{2} * dfx * dfx - fx * ddfx;

                auto const next = x - numerator / denominator;

                if (next == x) {
                    return next;
                }

                x = next;
            }

            return x;
        }
    } // namespace detail

    constexpr auto square_root = []<std::floating_point T>(T value) {
        return detail::newton_raphson([value](T x) { return x * x - value; }, [](T x) { return T{2} * x; }, value);
    };

    constexpr auto root_three = square_root(3.0F);
    constexpr auto root_two = square_root(2.0F);

} // namespace maths
