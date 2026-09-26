#include "core/random.hxx"

namespace {

    std::optional<std::uint32_t> g_fixed_seed;

} // namespace

auto set_fixed_random_seed(std::optional<std::uint32_t> seed) noexcept -> void { g_fixed_seed = seed; }

auto fixed_random_seed() noexcept -> std::optional<std::uint32_t> { return g_fixed_seed; }

auto make_random_engine(std::uint32_t stream) -> std::mt19937 {
    if (g_fixed_seed) {
        std::seed_seq seed{*g_fixed_seed, stream};
        return std::mt19937{seed};
    }

    std::random_device device;
    std::seed_seq seed{device(), device(), device(), device(), device(), device(), device(), device()};
    return std::mt19937{seed};
}
