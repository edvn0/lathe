#include "fly_string.hxx"

#include <cstdio>

FlyString::FlyString() noexcept = default;

FlyString::FlyString(std::string_view value) : value_(&pool().intern(value)) {}

[[nodiscard]]
auto FlyString::view() const noexcept -> std::string_view {
    return value_ != nullptr ? std::string_view{*value_} : std::string_view{};
}

[[nodiscard]]
auto FlyString::empty() const noexcept -> bool {
    return value_ == nullptr || value_->empty();
}

[[nodiscard]]
auto FlyString::c_str() const noexcept -> char const * {
    return value_ != nullptr ? value_->c_str() : "";
}

auto FlyString::operator==(FlyString rhs) const noexcept -> bool { return value_ == rhs.value_; }

auto FlyString::Pool::intern(std::string_view value) -> std::string const & {
    std::scoped_lock lock{mutex_};

    auto const [iterator, inserted] = strings_.emplace(value);

    ++total_requests_;
    if (inserted) {
        total_bytes_allocated_ += iterator->capacity();
    }

    return *iterator;
}

void FlyString::Pool::print_stats() const {
    std::scoped_lock lock{mutex_};

    std::size_t total_string_bytes = 0;
    for (auto const &str: strings_) {
        total_string_bytes += str.capacity();
    }

    float const hit_rate = total_requests_ > 0 ? (100.0F * static_cast<float>(total_requests_ - strings_.size()) /
                                                  static_cast<float>(total_requests_))
                                               : 0.0F;

    // This only ever runs from the atexit hook below, by which point other
    // function-local statics (including the spdlog-backed logger's) may
    // already be destroyed -- C++ does not order unrelated statics' teardown
    // against each other or against atexit callbacks. stderr has no such
    // lifetime issue, so the exit-time report bypasses logger::* entirely.
    std::fprintf(stderr, "=== FlyString Pool Stats ===\n");
    std::fprintf(stderr, "  Unique strings interned : %zu\n", strings_.size());
    std::fprintf(stderr, "  Total intern requests   : %zu\n", total_requests_);
    std::fprintf(stderr, "  Deduplication hit rate  : %.2f%%\n", static_cast<double>(hit_rate));
    std::fprintf(stderr, "  Heap payload allocated  : %zu bytes\n", total_string_bytes);
    std::fprintf(stderr, "============================\n");
}

auto FlyString::pool() -> Pool & {
    static auto *instance = []() {
        auto *p = new Pool{};
        std::atexit([]() { pool().print_stats(); });
        return p;
    }();
    return *instance;
}
