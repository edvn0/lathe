#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

static_assert(std::endian::native == std::endian::little, "LBF assumes a little-endian host");

template<typename T>
concept ByteStreamValue = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

class ByteWriter {
public:
    template<ByteStreamValue T>
    auto write(T const &value) -> void {
        auto const offset = bytes_.size();
        bytes_.resize(offset + sizeof(T));
        std::memcpy(bytes_.data() + offset, &value, sizeof(T));
    }

    template<ByteStreamValue T>
    auto write_span(std::span<T const> values) -> void {
        if (values.empty()) {
            return;
        }

        auto const offset = bytes_.size();
        bytes_.resize(offset + values.size_bytes());
        std::memcpy(bytes_.data() + offset, values.data(), values.size_bytes());
    }

    template<ByteStreamValue T>
    auto write_array(std::span<T const> values) -> void {
        write(static_cast<std::uint32_t>(values.size()));
        write_span(values);
    }

    template<ByteStreamValue T>
    auto write_array(std::vector<T> const &values) -> void {
        write_array(std::span<T const>{values});
    }

    auto write_string(std::string_view text) -> void {
        write(static_cast<std::uint32_t>(text.size()));
        write_span(std::span<char const>{text.data(), text.size()});
    }

    auto align(std::size_t alignment) -> void {
        auto const padded = (bytes_.size() + alignment - 1) & ~(alignment - 1);
        bytes_.resize(padded, std::byte{0});
    }

    auto write(ByteWriter &&subsection) -> void {
        write_span(subsection.bytes());
        subsection.clear();
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t { return bytes_.size(); }
    [[nodiscard]] auto bytes() const noexcept -> std::span<std::byte const> { return bytes_; }
    [[nodiscard]] auto take() noexcept -> std::vector<std::byte> { return std::move(bytes_); }
    auto clear() -> void { bytes_.clear(); }

private:
    std::vector<std::byte> bytes_;
};

class ByteReader {
public:
    explicit ByteReader(std::span<std::byte const> bytes) noexcept : bytes_(bytes) {}

    template<ByteStreamValue T>
    auto read(T &value) -> bool {
        if (!can_read(sizeof(T))) {
            value = T{};
            return false;
        }

        std::memcpy(&value, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return true;
    }

    template<ByteStreamValue T>
    [[nodiscard]] auto read() -> T {
        T value{};
        read(value);
        return value;
    }

    template<ByteStreamValue T>
    auto read_span(std::span<T> values) -> bool {
        if (!can_read(values.size_bytes())) {
            return false;
        }

        if (!values.empty()) {
            std::memcpy(values.data(), bytes_.data() + cursor_, values.size_bytes());
        }

        cursor_ += values.size_bytes();
        return true;
    }

    template<ByteStreamValue T>
    auto read_array(std::vector<T> &values) -> bool {
        auto const count = read<std::uint32_t>();

        if (failed_ || count > remaining() / sizeof(T)) {
            failed_ = true;
            values.clear();
            return false;
        }

        values.resize(count);
        return read_span(std::span<T>{values});
    }

    auto read_string(std::string &text) -> bool {
        auto const length = read<std::uint32_t>();

        if (failed_ || length > remaining()) {
            failed_ = true;
            text.clear();
            return false;
        }

        text.assign(reinterpret_cast<char const *>(bytes_.data() + cursor_), length);
        cursor_ += length;
        return true;
    }

    [[nodiscard]] auto read_string() -> std::string {
        std::string text;
        read_string(text);
        return text;
    }

    [[nodiscard]] auto read_bytes(std::size_t size) -> std::span<std::byte const> {
        if (!can_read(size)) {
            return {};
        }

        auto const view = bytes_.subspan(cursor_, size);
        cursor_ += size;
        return view;
    }

    auto skip(std::size_t size) -> bool { return !read_bytes(size).empty() || size == 0; }

    auto align(std::size_t alignment) -> bool {
        auto const padded = (cursor_ + alignment - 1) & ~(alignment - 1);
        return skip(padded - cursor_);
    }

    [[nodiscard]] auto failed() const noexcept -> bool { return failed_; }
    [[nodiscard]] auto ok() const noexcept -> bool { return !failed_; }
    [[nodiscard]] auto remaining() const noexcept -> std::size_t { return failed_ ? 0 : bytes_.size() - cursor_; }
    [[nodiscard]] auto cursor() const noexcept -> std::size_t { return cursor_; }

    auto fail() noexcept -> void { failed_ = true; }

private:
    [[nodiscard]] auto can_read(std::size_t size) noexcept -> bool {
        if (failed_ || size > bytes_.size() - cursor_) {
            failed_ = true;
            return false;
        }

        return true;
    }

    std::span<std::byte const> bytes_;
    std::size_t cursor_ = 0;
    bool failed_ = false;
};
