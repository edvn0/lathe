#include "serialisation/checksum.hxx"

#include <algorithm>
#include <bit>
#include <cstring>

namespace {

    constexpr std::uint64_t prime_1 = 0x9E3779B185EBCA87ULL;
    constexpr std::uint64_t prime_2 = 0xC2B2AE3D27D4EB4FULL;
    constexpr std::uint64_t prime_3 = 0x165667B19E3779F9ULL;
    constexpr std::uint64_t prime_4 = 0x85EBCA77C2B2AE63ULL;
    constexpr std::uint64_t prime_5 = 0x27D4EB2F165667C5ULL;

    [[nodiscard]] auto read_u64(std::byte const *data) noexcept -> std::uint64_t {
        std::uint64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    [[nodiscard]] auto read_u32(std::byte const *data) noexcept -> std::uint32_t {
        std::uint32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    [[nodiscard]] auto round(std::uint64_t accumulator, std::uint64_t input) noexcept -> std::uint64_t {
        accumulator += input * prime_2;
        accumulator = std::rotl(accumulator, 31);
        return accumulator * prime_1;
    }

    [[nodiscard]] auto merge_round(std::uint64_t accumulator, std::uint64_t value) noexcept -> std::uint64_t {
        accumulator ^= round(0, value);
        return accumulator * prime_1 + prime_4;
    }

} // namespace

namespace {

    // Folds the tail (< 32 bytes) into `hash` and applies the final avalanche.
    [[nodiscard]] auto finish(std::uint64_t hash, std::byte const *cursor, std::byte const *end) noexcept
            -> std::uint64_t {
        while (end - cursor >= 8) {
            hash ^= round(0, read_u64(cursor));
            hash = std::rotl(hash, 27) * prime_1 + prime_4;
            cursor += 8;
        }

        if (end - cursor >= 4) {
            hash ^= static_cast<std::uint64_t>(read_u32(cursor)) * prime_1;
            hash = std::rotl(hash, 23) * prime_2 + prime_3;
            cursor += 4;
        }

        while (cursor < end) {
            hash ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(*cursor)) * prime_5;
            hash = std::rotl(hash, 11) * prime_1;
            ++cursor;
        }

        hash ^= hash >> 33;
        hash *= prime_2;
        hash ^= hash >> 29;
        hash *= prime_3;
        hash ^= hash >> 32;

        return hash;
    }

    [[nodiscard]] auto merge_lanes(std::array<std::uint64_t, 4> const &lanes) noexcept -> std::uint64_t {
        auto hash = std::rotl(lanes[0], 1) + std::rotl(lanes[1], 7) + std::rotl(lanes[2], 12) + std::rotl(lanes[3], 18);
        hash = merge_round(hash, lanes[0]);
        hash = merge_round(hash, lanes[1]);
        hash = merge_round(hash, lanes[2]);
        hash = merge_round(hash, lanes[3]);
        return hash;
    }

} // namespace

Xxh64Stream::Xxh64Stream(std::uint64_t seed) noexcept :
    seed_(seed), lanes_{seed + prime_1 + prime_2, seed + prime_2, seed, seed - prime_1} {}

auto Xxh64Stream::update(std::span<std::byte const> bytes) noexcept -> void {
    auto const *cursor = bytes.data();
    auto const *const end = cursor + bytes.size();
    total_ += bytes.size();

    if (buffered_ > 0) {
        auto const take = std::min<std::size_t>(32 - buffered_, bytes.size());
        std::memcpy(buffer_.data() + buffered_, cursor, take);
        buffered_ += take;
        cursor += take;

        if (buffered_ < 32) {
            return;
        }

        for (std::size_t lane = 0; lane < 4; ++lane) {
            lanes_[lane] = round(lanes_[lane], read_u64(buffer_.data() + lane * 8));
        }

        buffered_ = 0;
    }

    while (end - cursor >= 32) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            lanes_[lane] = round(lanes_[lane], read_u64(cursor + lane * 8));
        }

        cursor += 32;
    }

    buffered_ = static_cast<std::size_t>(end - cursor);

    if (buffered_ > 0) {
        std::memcpy(buffer_.data(), cursor, buffered_);
    }
}

auto Xxh64Stream::digest() const noexcept -> std::uint64_t {
    auto hash = total_ >= 32 ? merge_lanes(lanes_) : seed_ + prime_5;
    hash += total_;
    return finish(hash, buffer_.data(), buffer_.data() + buffered_);
}

auto xxh64(std::span<std::byte const> bytes, std::uint64_t seed) noexcept -> std::uint64_t {
    auto const *cursor = bytes.data();
    auto const *const end = cursor + bytes.size();
    std::uint64_t hash = 0;

    if (bytes.size() >= 32) {
        auto lane_1 = seed + prime_1 + prime_2;
        auto lane_2 = seed + prime_2;
        auto lane_3 = seed;
        auto lane_4 = seed - prime_1;

        auto const *const limit = end - 32;

        do {
            lane_1 = round(lane_1, read_u64(cursor));
            lane_2 = round(lane_2, read_u64(cursor + 8));
            lane_3 = round(lane_3, read_u64(cursor + 16));
            lane_4 = round(lane_4, read_u64(cursor + 24));
            cursor += 32;
        } while (cursor <= limit);

        hash = std::rotl(lane_1, 1) + std::rotl(lane_2, 7) + std::rotl(lane_3, 12) + std::rotl(lane_4, 18);
        hash = merge_round(hash, lane_1);
        hash = merge_round(hash, lane_2);
        hash = merge_round(hash, lane_3);
        hash = merge_round(hash, lane_4);
    } else {
        hash = seed + prime_5;
    }

    hash += static_cast<std::uint64_t>(bytes.size());

    return finish(hash, cursor, end);
}

auto xxh64(std::string_view text, std::uint64_t seed) noexcept -> std::uint64_t {
    return xxh64(std::as_bytes(std::span<char const>{text.data(), text.size()}), seed);
}
