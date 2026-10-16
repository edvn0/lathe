#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

// An ELF `.strtab`-style blob: NUL-terminated strings back to back, byte 0 is a NUL so offset 0 is the empty string.
class StringTable {
public:
    StringTable() : blob_(1, '\0') {}

    // Strings must not contain a NUL (callers check first); the offset of an existing equal string is reused.
    [[nodiscard]] auto intern(std::string_view text) -> std::uint32_t {
        if (text.empty()) {
            return 0;
        }

        if (auto const found = offsets_.find(text); found != offsets_.end()) {
            return found->second;
        }

        auto const offset = static_cast<std::uint32_t>(blob_.size());
        blob_.append(text);
        blob_.push_back('\0');
        offsets_.emplace(std::string{text}, offset);
        return offset;
    }

    [[nodiscard]] auto bytes() const noexcept -> std::span<std::byte const> { return std::as_bytes(std::span{blob_}); }

private:
    struct Hash {
        using is_transparent = void;
        auto operator()(std::string_view text) const noexcept -> std::size_t {
            return std::hash<std::string_view>{}(text);
        }
    };

    std::string blob_;
    std::unordered_map<std::string, std::uint32_t, Hash, std::equal_to<>> offsets_;
};

// Read side of the same blob. Every lookup is bounds-checked against the bytes it was given.
class StringBlob {
public:
    // Valid only when non-empty and NUL-terminated, with the empty string at offset 0.
    [[nodiscard]] static auto parse(std::span<std::byte const> bytes) noexcept -> bool {
        return !bytes.empty() && bytes.front() == std::byte{0} && bytes.back() == std::byte{0};
    }

    explicit StringBlob(std::span<std::byte const> bytes) noexcept : bytes_(bytes) {}

    auto lookup(std::uint32_t offset, std::string &text) const -> bool {
        if (offset >= bytes_.size()) {
            return false;
        }

        auto const *const begin = reinterpret_cast<char const *>(bytes_.data()) + offset;
        auto const *const end = static_cast<char const *>(std::memchr(begin, '\0', bytes_.size() - offset));

        if (end == nullptr) {
            return false;
        }

        text.assign(begin, end);
        return true;
    }

private:
    std::span<std::byte const> bytes_;
};
