#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "serialisation/lbf_error.hxx"

inline constexpr std::uint32_t lbf_magic = 0x1A46424CU;
inline constexpr std::uint16_t lbf_version_major = 1;
inline constexpr std::uint16_t lbf_version_minor = 0;
inline constexpr std::size_t lbf_payload_alignment = 64;

inline constexpr std::uint64_t lbf_max_chunk_raw_size = std::uint64_t{2} << 30U;

[[nodiscard]] constexpr auto make_fourcc(char a, char b, char c, char d) noexcept -> std::uint32_t {
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(a)) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b)) << 8U) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(c)) << 16U) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(d)) << 24U);
}

namespace lbf_chunk {
    inline constexpr std::uint32_t scene = make_fourcc('S', 'C', 'E', 'N');
    inline constexpr std::uint32_t model = make_fourcc('M', 'O', 'D', 'L');
    inline constexpr std::uint32_t texture = make_fourcc('T', 'E', 'X', 'R');
    inline constexpr std::uint32_t metadata = make_fourcc('M', 'E', 'T', 'A');
    inline constexpr std::uint32_t environment = make_fourcc('E', 'N', 'V', 'M');
    inline constexpr std::uint32_t file = make_fourcc('F', 'I', 'L', 'E');
}

enum class LbfFileKind : std::uint32_t {
    scene = 1,
    asset_pack = 2,
};

enum class LbfCompression : std::uint8_t {
    none = 0,
    zstd = 1,
};

struct LbfFileHeader {
    std::uint32_t magic = lbf_magic;
    std::uint16_t version_major = lbf_version_major;
    std::uint16_t version_minor = lbf_version_minor;
    LbfFileKind kind = LbfFileKind::scene;
    std::uint32_t flags = 0;
    std::uint32_t header_size = sizeof(LbfFileHeader);
    std::uint32_t chunk_count = 0;
    std::uint64_t toc_offset = 0;
    std::uint64_t toc_size = 0;
    std::uint64_t toc_checksum = 0;
    std::uint64_t file_size = 0;
    std::uint64_t reserved = 0;
};

static_assert(sizeof(LbfFileHeader) == 64);
static_assert(std::is_trivially_copyable_v<LbfFileHeader>);

struct LbfChunkEntry {
    std::uint32_t type = 0;
    LbfCompression compression = LbfCompression::none;
    std::uint8_t reserved_0 = 0;
    std::uint16_t version = 0;
    std::uint64_t id = 0;
    std::uint64_t offset = 0;
    std::uint64_t stored_size = 0;
    std::uint64_t raw_size = 0;
    std::uint64_t checksum = 0;
};

static_assert(sizeof(LbfChunkEntry) == 48);
static_assert(std::is_trivially_copyable_v<LbfChunkEntry>);

struct LbfWriteOptions {
    int compression_level = 6;

    float minimum_savings = 0.03F;

    bool parallel = true;
};

struct LbfChunkInput {
    std::uint32_t type = 0;
    std::uint64_t id = 0;
    std::uint16_t version = 0;
    std::vector<std::byte> payload;
    bool compress = true;
};

class LbfWriter {
public:
    explicit LbfWriter(LbfFileKind kind) noexcept : kind_(kind) {}

    auto add_chunk(LbfChunkInput chunk) -> void;

    auto add_stored_chunk(LbfChunkEntry const &entry, std::vector<std::byte> stored_bytes) -> void;

    [[nodiscard]] auto contains(std::uint32_t type, std::uint64_t id) const noexcept -> bool;

    [[nodiscard]]
    auto finish(LbfWriteOptions const &options = {}) -> std::expected<std::vector<std::byte>, LbfError>;

    [[nodiscard]]
    auto write_file(std::filesystem::path const &path, LbfWriteOptions const &options = {})
            -> std::expected<std::uint64_t, LbfError>;

private:
    [[nodiscard]]
    auto emit(LbfWriteOptions const &options, std::function<bool(std::span<std::byte const>)> const &write)
            -> std::expected<LbfFileHeader, LbfError>;

    struct PendingChunk {
        LbfChunkEntry entry;
        std::vector<std::byte> bytes;
        bool compress = true;
        bool stored = false;
    };

    LbfFileKind kind_;
    std::vector<PendingChunk> chunks_;
};

struct LbfReadOptions {
    bool verify_checksums = true;
};

class LbfReader {
public:
    [[nodiscard]]
    static auto open(std::filesystem::path const &path, LbfReadOptions const &options = {})
            -> std::expected<LbfReader, LbfError>;

    [[nodiscard]]
    static auto from_memory(std::vector<std::byte> bytes, LbfReadOptions const &options = {})
            -> std::expected<LbfReader, LbfError>;

    [[nodiscard]] auto header() const noexcept -> LbfFileHeader const & { return header_; }
    [[nodiscard]] auto chunks() const noexcept -> std::span<LbfChunkEntry const> { return chunks_; }
    [[nodiscard]] auto path() const noexcept -> std::filesystem::path const & { return path_; }

    [[nodiscard]]
    auto find(std::uint32_t type, std::uint64_t id = 0) const noexcept -> LbfChunkEntry const *;

    [[nodiscard]]
    auto read_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

    [[nodiscard]]
    auto read_stored_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

private:
    LbfReader() = default;

    [[nodiscard]]
    static auto parse(LbfReader reader, std::span<std::byte const> header_bytes) -> std::expected<LbfReader, LbfError>;

    [[nodiscard]]
    auto read_range(std::uint64_t offset, std::uint64_t size) const -> std::expected<std::vector<std::byte>, LbfError>;

    [[nodiscard]]
    auto stream_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

    std::filesystem::path path_;
    std::shared_ptr<std::vector<std::byte> const> memory_;
    LbfFileHeader header_{};
    std::vector<LbfChunkEntry> chunks_;
    LbfReadOptions options_{};
    std::uint64_t source_size_ = 0;
};

[[nodiscard]]
auto lbf_chunk_type_name(std::uint32_t type) -> std::string;
