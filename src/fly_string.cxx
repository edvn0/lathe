#include "fly_string.hxx"

#include "logger.hxx"

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

    // Track allocation metrics
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

    debug("=== FlyString Pool Stats ===");
    debug("  Unique strings interned : {}", strings_.size());
    debug("  Total intern requests   : {}", total_requests_);
    debug("  Deduplication hit rate  : {:.2f}%", hit_rate);
    debug("  Heap payload allocated  : {} bytes", total_string_bytes);
    debug("============================");
}

auto FlyString::pool() -> Pool & {
    static auto *instance = []() {
        auto *p = new Pool{};
        std::atexit([]() { pool().print_stats(); });
        return p;
    }();
    return *instance;
}
