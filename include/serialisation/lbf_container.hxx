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

// The Lathe Binary Format container. Scene files (.lbf) and asset packs use the same layout:
//
//   [LbfFileHeader]                   64 bytes at offset 0
//   [chunk payload] ...               each starts on a 64-byte boundary
//   [LbfChunkEntry x chunk_count]     the table of contents, located by the header
//
// The table of contents sits at the end so the writer can stream payloads without knowing their sizes up front.
// A reader needs two small reads (header, then table) before it can fetch any chunk independently, which is what
// lets asset loads run in parallel on the thread pool and lets a scene pull in only the chunks it references.
//
// Each chunk is compressed on its own (zstd, or stored raw when that doesn't pay off) and carries an xxh64 of its
// stored bytes, checked before decompression. See docs/lathe-binary-format.md.

inline constexpr std::uint32_t lbf_magic = 0x1A46424CU; // "LBF\x1A"
inline constexpr std::uint16_t lbf_version_major = 1;
inline constexpr std::uint16_t lbf_version_minor = 0;
inline constexpr std::size_t lbf_payload_alignment = 64;

// Upper bound on one chunk's decompressed size. read_chunk() allocates raw_size before it has decoded anything, and
// raw_size comes from the file, so without a cap a 100-byte file can ask for terabytes (and allocation failure aborts).
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
} // namespace lbf_chunk

enum class LbfFileKind : std::uint32_t {
    scene = 1, // one SCEN chunk, plus whatever assets were embedded
    asset_pack = 2, // MODL/TEXR chunks only
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
    std::uint32_t type = 0; // lbf_chunk::*
    LbfCompression compression = LbfCompression::none;
    std::uint8_t reserved_0 = 0;
    std::uint16_t version = 0; // payload layout version, owned by the chunk's codec
    std::uint64_t id = 0; // AssetId::value for assets, 0 for singletons like SCEN
    std::uint64_t offset = 0; // from the start of the file
    std::uint64_t stored_size = 0;
    std::uint64_t raw_size = 0;
    std::uint64_t checksum = 0; // xxh64 of the stored bytes
};

static_assert(sizeof(LbfChunkEntry) == 48);
static_assert(std::is_trivially_copyable_v<LbfChunkEntry>);

struct LbfWriteOptions {
    // zstd level; 1-3 cooks fast, 9+ squeezes harder at the same decompression speed.
    int compression_level = 6;

    // Chunks that shrink by less than this fraction are stored raw: BC7 blocks barely compress, and skipping
    // decompression is worth more than a few percent of disk.
    float minimum_savings = 0.03F;

    // Compress chunks on thread_pool(). Call from outside the pool when set.
    bool parallel = true;
};

struct LbfChunkInput {
    std::uint32_t type = 0;
    std::uint64_t id = 0;
    std::uint16_t version = 0;
    std::vector<std::byte> payload;
    bool compress = true;
};

// Collects chunks, then compresses and writes them in one go.
class LbfWriter {
public:
    explicit LbfWriter(LbfFileKind kind) noexcept : kind_(kind) {}

    auto add_chunk(LbfChunkInput chunk) -> void;

    // Copies a chunk verbatim (still compressed) from another file, e.g. an unchanged asset from the pack a scene
    // was loaded from. `stored_bytes` must be the entry's stored bytes.
    auto add_stored_chunk(LbfChunkEntry const &entry, std::vector<std::byte> stored_bytes) -> void;

    [[nodiscard]] auto contains(std::uint32_t type, std::uint64_t id) const noexcept -> bool;

    // The whole file in memory (tests, small files). Consumes the chunks.
    [[nodiscard]]
    auto finish(LbfWriteOptions const &options = {}) -> std::expected<std::vector<std::byte>, LbfError>;

    // Streams the file to a temporary next to `path`, then renames it over `path`, so a failed save never leaves a
    // half-written scene behind. Chunks are compressed a few at a time ahead of the write position and each is freed
    // once written, so the file never exists in memory as a whole. Consumes the chunks.
    [[nodiscard]]
    auto write_file(std::filesystem::path const &path, LbfWriteOptions const &options = {})
            -> std::expected<std::uint64_t, LbfError>;

private:
    // Writes the file through `write` (sequential bytes from offset 0, header zeroed) and returns the header to patch
    // in at offset 0.
    [[nodiscard]]
    auto emit(LbfWriteOptions const &options, std::function<bool(std::span<std::byte const>)> const &write)
            -> std::expected<LbfFileHeader, LbfError>;

    struct PendingChunk {
        LbfChunkEntry entry;
        std::vector<std::byte> bytes; // raw until finish() compresses it, unless `stored`
        bool compress = true;
        bool stored = false;
    };

    LbfFileKind kind_;
    std::vector<PendingChunk> chunks_;
};

struct LbfReadOptions {
    bool verify_checksums = true;
};

// Random access to one .lbf file, streamed: open() reads only the header and the table of contents, and each
// read_chunk() reads just that chunk, through a fixed-size buffer (decompressing and checksumming as it goes), so
// peak memory is the chunk's decompressed size plus the buffer, whatever the file's size. Nothing is mapped.
//
// read_chunk() is const and thread-safe: every call opens its own handle (or reads the shared in-memory copy), so many
// chunks can be decoded at once.
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

    // nullptr if absent. Binary search; the writer sorts the table by (type, id).
    [[nodiscard]]
    auto find(std::uint32_t type, std::uint64_t id = 0) const noexcept -> LbfChunkEntry const *;

    // Decompressed payload.
    [[nodiscard]]
    auto read_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

    // The stored (possibly compressed) bytes, checksum-verified; see LbfWriter::add_stored_chunk.
    [[nodiscard]]
    auto read_stored_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

private:
    LbfReader() = default;

    [[nodiscard]]
    static auto parse(LbfReader reader, std::span<std::byte const> header_bytes) -> std::expected<LbfReader, LbfError>;

    [[nodiscard]]
    auto read_range(std::uint64_t offset, std::uint64_t size) const -> std::expected<std::vector<std::byte>, LbfError>;

    // read_chunk() from the file, block by block.
    [[nodiscard]]
    auto stream_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError>;

    std::filesystem::path path_;
    std::shared_ptr<std::vector<std::byte> const> memory_;
    LbfFileHeader header_{};
    std::vector<LbfChunkEntry> chunks_;
    LbfReadOptions options_{};
    std::uint64_t source_size_ = 0; // the real size of the file or buffer, which the header must agree with
};

[[nodiscard]]
auto lbf_chunk_type_name(std::uint32_t type) -> std::string;
