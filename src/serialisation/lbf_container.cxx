#include "serialisation/lbf_container.hxx"

#include <algorithm>
#include <array>
#include <deque>
#include <cstring>
#include <format>
#include <fstream>
#include <future>
#include <source_location>
#include <system_error>
#include <thread>
#include <utility>

#include <zstd.h>

#include "core/thread_pool.hxx"
#include "serialisation/checksum.hxx"

namespace {

    auto make_error(LbfErrorType type, std::string_view message = {},
                    std::source_location location = std::source_location::current()) -> LbfError {
        return LbfError{
                .type = type,
                .cause = ErrorCause{ErrorContext{
                        .message = FlyString{message},
                        .location = location,
                }},
        };
    }

    [[nodiscard]] auto align_up(std::uint64_t value, std::uint64_t alignment) noexcept -> std::uint64_t {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    [[nodiscard]] auto chunk_order(LbfChunkEntry const &left, LbfChunkEntry const &right) noexcept -> bool {
        return left.type != right.type ? left.type < right.type : left.id < right.id;
    }

    // Compresses `raw` in place when it saves at least `minimum_savings`; otherwise leaves it stored raw.
    [[nodiscard]]
    auto compress_chunk(std::vector<std::byte> &bytes, LbfChunkEntry &entry, LbfWriteOptions const &options)
            -> std::expected<void, LbfError> {
        entry.raw_size = bytes.size();
        entry.compression = LbfCompression::none;

        if (!bytes.empty()) {
            std::vector<std::byte> compressed(ZSTD_compressBound(bytes.size()));

            auto const written = ZSTD_compress(compressed.data(), compressed.size(), bytes.data(), bytes.size(),
                                               options.compression_level);

            if (ZSTD_isError(written) != 0U) {
                return std::unexpected(make_error(LbfErrorType::compression_failed, ZSTD_getErrorName(written)));
            }

            auto const threshold = static_cast<double>(bytes.size()) * (1.0 - static_cast<double>(options.minimum_savings));

            if (static_cast<double>(written) <= threshold) {
                compressed.resize(written);
                bytes = std::move(compressed);
                entry.compression = LbfCompression::zstd;
            }
        }

        entry.stored_size = bytes.size();
        entry.checksum = xxh64(bytes);

        return {};
    }

} // namespace

auto lbf_chunk_type_name(std::uint32_t type) -> std::string {
    std::string name(4, '?');

    for (std::size_t index = 0; index < 4; ++index) {
        auto const character = static_cast<char>((type >> (index * 8U)) & 0xFFU);
        name[index] = (character >= 0x20 && character < 0x7F) ? character : '?';
    }

    return name;
}

auto LbfWriter::add_chunk(LbfChunkInput chunk) -> void {
    chunks_.push_back(PendingChunk{
            .entry =
                    LbfChunkEntry{
                            .type = chunk.type,
                            .version = chunk.version,
                            .id = chunk.id,
                    },
            .bytes = std::move(chunk.payload),
            .compress = chunk.compress,
    });
}

auto LbfWriter::add_stored_chunk(LbfChunkEntry const &entry, std::vector<std::byte> stored_bytes) -> void {
    auto copy = entry;
    copy.offset = 0;

    chunks_.push_back(PendingChunk{
            .entry = copy,
            .bytes = std::move(stored_bytes),
            .stored = true,
    });
}

auto LbfWriter::contains(std::uint32_t type, std::uint64_t id) const noexcept -> bool {
    return std::ranges::any_of(chunks_, [&](PendingChunk const &chunk) {
        return chunk.entry.type == type && chunk.entry.id == id;
    });
}

auto LbfWriter::emit(LbfWriteOptions const &options, std::function<bool(std::span<std::byte const>)> const &write)
        -> std::expected<LbfFileHeader, LbfError> {
    ZoneScopedNC("LbfWriter::emit", tracy::Color::Goldenrod);

    // Duplicate (type, id) pairs would make find() ambiguous; keep the first.
    std::vector<std::size_t> order(chunks_.size());

    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }

    std::ranges::stable_sort(order, [&](std::size_t left, std::size_t right) {
        return chunk_order(chunks_[left].entry, chunks_[right].entry);
    });

    auto const duplicate = std::ranges::unique(order, [&](std::size_t left, std::size_t right) {
        return chunks_[left].entry.type == chunks_[right].entry.type && chunks_[left].entry.id == chunks_[right].entry.id;
    });

    order.erase(duplicate.begin(), duplicate.end());

    auto const process = [&options](PendingChunk &chunk) -> std::expected<void, LbfError> {
        if (chunk.stored) {
            return {};
        }

        if (!chunk.compress) {
            chunk.entry.raw_size = chunk.bytes.size();
            chunk.entry.stored_size = chunk.bytes.size();
            chunk.entry.compression = LbfCompression::none;
            chunk.entry.checksum = xxh64(chunk.bytes);
            return {};
        }

        return compress_chunk(chunk.bytes, chunk.entry, options);
    };

    // Compression runs up to `window` chunks ahead of the write position, so a big file is never held compressed in
    // full, while every core still has work.
    auto const window = options.parallel ? std::max<std::size_t>(2, std::size_t{2} * std::thread::hardware_concurrency()) : 0;
    std::deque<std::future<std::expected<void, LbfError>>> in_flight;
    std::size_t submitted = 0;

    auto const drain = [&] {
        for (auto &task: in_flight) {
            task.wait();
        }
    };

    std::array<std::byte, lbf_payload_alignment> const zeros{};
    std::uint64_t offset = 0;

    auto const put = [&](std::span<std::byte const> bytes) -> bool {
        offset += bytes.size();
        return bytes.empty() || write(bytes);
    };

    auto const pad_to = [&](std::uint64_t alignment) -> bool {
        auto const padding = align_up(offset, alignment) - offset;
        return put(std::span<std::byte const>{zeros}.first(padding));
    };

    if (!put(std::span<std::byte const>{zeros}.first(sizeof(LbfFileHeader)))) {
        return std::unexpected(make_error(LbfErrorType::io_failed, "write failed"));
    }

    std::vector<LbfChunkEntry> table;
    table.reserve(order.size());

    for (std::size_t position = 0; position < order.size(); ++position) {
        auto &chunk = chunks_[order[position]];

        if (options.parallel) {
            while (submitted < order.size() && submitted < position + window) {
                in_flight.push_back(thread_pool().submit_task([&process, next = &chunks_[order[submitted]]] {
                    return process(*next);
                }));
                ++submitted;
            }

            auto result = in_flight.front().get();
            in_flight.pop_front();

            if (!result) {
                drain();
                return std::unexpected(result.error());
            }
        } else if (auto result = process(chunk); !result) {
            return std::unexpected(result.error());
        }

        if (!pad_to(lbf_payload_alignment)) {
            drain();
            return std::unexpected(make_error(LbfErrorType::io_failed, "write failed"));
        }

        chunk.entry.offset = offset;

        if (!put(chunk.bytes)) {
            drain();
            return std::unexpected(make_error(LbfErrorType::io_failed, "write failed"));
        }

        table.push_back(chunk.entry);

        // Written; free it now rather than when the writer goes away.
        std::vector<std::byte>{}.swap(chunk.bytes);
    }

    if (!pad_to(alignof(LbfChunkEntry))) {
        return std::unexpected(make_error(LbfErrorType::io_failed, "write failed"));
    }

    auto const toc_offset = offset;
    auto const toc_bytes = std::as_bytes(std::span<LbfChunkEntry const>{table});

    if (!put(toc_bytes)) {
        return std::unexpected(make_error(LbfErrorType::io_failed, "write failed"));
    }

    chunks_.clear();

    return LbfFileHeader{
            .kind = kind_,
            .chunk_count = static_cast<std::uint32_t>(table.size()),
            .toc_offset = toc_offset,
            .toc_size = toc_bytes.size(),
            .toc_checksum = xxh64(toc_bytes),
            .file_size = offset,
    };
}

auto LbfWriter::finish(LbfWriteOptions const &options) -> std::expected<std::vector<std::byte>, LbfError> {
    std::vector<std::byte> file;

    auto header = emit(options, [&file](std::span<std::byte const> bytes) {
        file.insert(file.end(), bytes.begin(), bytes.end());
        return true;
    });

    if (!header) {
        return std::unexpected(header.error());
    }

    std::memcpy(file.data(), &*header, sizeof(LbfFileHeader));
    return file;
}

auto LbfWriter::write_file(std::filesystem::path const &path, LbfWriteOptions const &options)
        -> std::expected<std::uint64_t, LbfError> {
    std::error_code error;

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
    }

    auto temporary = path;
    temporary += ".tmp";

    std::expected<LbfFileHeader, LbfError> header = std::unexpected(LbfError{});

    {
        std::ofstream stream{temporary, std::ios::binary | std::ios::trunc};

        if (!stream) {
            return std::unexpected(
                    make_error(LbfErrorType::io_failed, std::format("cannot open '{}' for writing", temporary.string())));
        }

        header = emit(options, [&stream](std::span<std::byte const> bytes) {
            stream.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            return static_cast<bool>(stream);
        });

        if (header) {
            stream.seekp(0);
            stream.write(reinterpret_cast<char const *>(&*header), sizeof(LbfFileHeader));
            stream.flush();

            if (!stream) {
                header = std::unexpected(make_error(LbfErrorType::io_failed, std::format("short write to '{}'", temporary.string())));
            }
        }
    }

    if (!header) {
        std::filesystem::remove(temporary, error);
        return std::unexpected(header.error());
    }

    std::filesystem::rename(temporary, path, error);

    if (error) {
        std::filesystem::remove(temporary, error);
        return std::unexpected(make_error(LbfErrorType::io_failed, std::format("cannot replace '{}'", path.string())));
    }

    return header->file_size;
}

auto LbfReader::open(std::filesystem::path const &path, LbfReadOptions const &options)
        -> std::expected<LbfReader, LbfError> {
    std::error_code error;

    if (!std::filesystem::is_regular_file(path, error)) {
        return std::unexpected(make_error(LbfErrorType::file_not_found, path.string()));
    }

    LbfReader reader;
    reader.path_ = path;
    reader.options_ = options;

    auto header_bytes = reader.read_range(0, sizeof(LbfFileHeader));

    if (!header_bytes) {
        return std::unexpected(make_error(LbfErrorType::not_an_lbf_file, path.string()));
    }

    auto const file_size = std::filesystem::file_size(path, error);

    if (error) {
        return std::unexpected(make_error(LbfErrorType::io_failed, path.string()));
    }

    auto parsed = parse(std::move(reader), *header_bytes);

    if (parsed && parsed->header_.file_size != file_size) {
        return std::unexpected(make_error(LbfErrorType::corrupt_header, "file size does not match its header (truncated?)"));
    }

    return parsed;
}

auto LbfReader::from_memory(std::vector<std::byte> bytes, LbfReadOptions const &options)
        -> std::expected<LbfReader, LbfError> {
    LbfReader reader;
    reader.memory_ = std::make_shared<std::vector<std::byte> const>(std::move(bytes));
    reader.options_ = options;

    auto const memory = reader.memory_;

    if (memory->size() < sizeof(LbfFileHeader)) {
        return std::unexpected(make_error(LbfErrorType::not_an_lbf_file, "shorter than an LBF header"));
    }

    auto parsed = parse(std::move(reader), std::span<std::byte const>{*memory}.first(sizeof(LbfFileHeader)));

    if (parsed && parsed->header_.file_size != memory->size()) {
        return std::unexpected(make_error(LbfErrorType::corrupt_header, "size does not match its header (truncated?)"));
    }

    return parsed;
}

auto LbfReader::parse(LbfReader reader, std::span<std::byte const> header_bytes) -> std::expected<LbfReader, LbfError> {
    std::memcpy(&reader.header_, header_bytes.data(), sizeof(LbfFileHeader));

    auto const &header = reader.header_;

    if (header.magic != lbf_magic) {
        return std::unexpected(make_error(LbfErrorType::not_an_lbf_file, "bad magic"));
    }

    // Minor versions only add things a reader may ignore; a different major changes the layout.
    if (header.version_major != lbf_version_major) {
        return std::unexpected(make_error(LbfErrorType::unsupported_version,
                                          std::format("file is v{}.{}, this build reads v{}.x", header.version_major,
                                                      header.version_minor, lbf_version_major)));
    }

    if (header.header_size < sizeof(LbfFileHeader) ||
        header.toc_size != static_cast<std::uint64_t>(header.chunk_count) * sizeof(LbfChunkEntry) ||
        header.toc_offset > header.file_size || header.toc_size > header.file_size - header.toc_offset) {
        return std::unexpected(make_error(LbfErrorType::corrupt_header));
    }

    auto toc_bytes = reader.read_range(header.toc_offset, header.toc_size);

    if (!toc_bytes) {
        return std::unexpected(toc_bytes.error());
    }

    if (reader.options_.verify_checksums && xxh64(*toc_bytes) != header.toc_checksum) {
        return std::unexpected(make_error(LbfErrorType::checksum_mismatch, "table of contents"));
    }

    reader.chunks_.resize(header.chunk_count);

    if (!toc_bytes->empty()) {
        std::memcpy(reader.chunks_.data(), toc_bytes->data(), toc_bytes->size());
    }

    for (auto const &entry: reader.chunks_) {
        if (entry.offset > header.toc_offset || entry.stored_size > header.toc_offset - entry.offset) {
            return std::unexpected(make_error(LbfErrorType::corrupt_table_of_contents,
                                              std::format("chunk {} {:016x} points outside the file",
                                                          lbf_chunk_type_name(entry.type), entry.id)));
        }
    }

    if (!std::ranges::is_sorted(reader.chunks_, chunk_order)) {
        std::ranges::sort(reader.chunks_, chunk_order);
    }

    return reader;
}

auto LbfReader::find(std::uint32_t type, std::uint64_t id) const noexcept -> LbfChunkEntry const * {
    LbfChunkEntry const key{.type = type, .id = id};
    auto const it = std::ranges::lower_bound(chunks_, key, chunk_order);

    if (it == chunks_.end() || it->type != type || it->id != id) {
        return nullptr;
    }

    return &*it;
}

auto LbfReader::read_range(std::uint64_t offset, std::uint64_t size) const
        -> std::expected<std::vector<std::byte>, LbfError> {
    if (memory_ != nullptr) {
        if (offset > memory_->size() || size > memory_->size() - offset) {
            return std::unexpected(make_error(LbfErrorType::io_failed, "read past the end"));
        }

        auto const begin = memory_->begin() + static_cast<std::ptrdiff_t>(offset);
        return std::vector<std::byte>{begin, begin + static_cast<std::ptrdiff_t>(size)};
    }

    std::ifstream stream{path_, std::ios::binary};

    if (!stream) {
        return std::unexpected(make_error(LbfErrorType::io_failed, std::format("cannot open '{}'", path_.string())));
    }

    std::vector<std::byte> bytes(size);
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));

    if (!stream || static_cast<std::uint64_t>(stream.gcount()) != size) {
        return std::unexpected(make_error(LbfErrorType::io_failed, std::format("short read from '{}'", path_.string())));
    }

    return bytes;
}

auto LbfReader::read_stored_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError> {
    auto stored = read_range(entry.offset, entry.stored_size);

    if (!stored) {
        return std::unexpected(stored.error());
    }

    if (options_.verify_checksums && xxh64(*stored) != entry.checksum) {
        return std::unexpected(make_error(LbfErrorType::checksum_mismatch,
                                          std::format("chunk {} {:016x}", lbf_chunk_type_name(entry.type), entry.id)));
    }

    return stored;
}

auto LbfReader::stream_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError> {
    constexpr std::uint64_t block_size = 1U << 20U;

    std::ifstream stream{path_, std::ios::binary};

    if (!stream) {
        return std::unexpected(make_error(LbfErrorType::io_failed, std::format("cannot open '{}'", path_.string())));
    }

    stream.seekg(static_cast<std::streamoff>(entry.offset));

    std::vector<std::byte> raw(entry.raw_size);
    std::vector<std::byte> block(entry.compression == LbfCompression::none ? 0 : std::min(block_size, entry.stored_size));
    Xxh64Stream checksum;

    auto const read_into = [&](std::byte *destination, std::uint64_t size) -> bool {
        stream.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(size));
        return stream && static_cast<std::uint64_t>(stream.gcount()) == size;
    };

    auto const chunk_name = [&] { return std::format("chunk {} {:016x}", lbf_chunk_type_name(entry.type), entry.id); };

    switch (entry.compression) {
        case LbfCompression::none: {
            if (entry.stored_size != entry.raw_size) {
                return std::unexpected(make_error(LbfErrorType::corrupt_table_of_contents, "raw/stored size mismatch"));
            }

            // Straight into the result, a block at a time so the checksum stays cache-hot.
            for (std::uint64_t done = 0; done < entry.stored_size;) {
                auto const size = std::min(block_size, entry.stored_size - done);

                if (!read_into(raw.data() + done, size)) {
                    return std::unexpected(make_error(LbfErrorType::io_failed, "short read; " + chunk_name()));
                }

                checksum.update(std::span<std::byte const>{raw}.subspan(done, size));
                done += size;
            }
            break;
        }

        case LbfCompression::zstd: {
            std::unique_ptr<ZSTD_DStream, decltype(&ZSTD_freeDStream)> decoder{ZSTD_createDStream(), &ZSTD_freeDStream};

            if (decoder == nullptr || ZSTD_isError(ZSTD_initDStream(decoder.get())) != 0U) {
                return std::unexpected(make_error(LbfErrorType::decompression_failed, "cannot create a zstd stream"));
            }

            ZSTD_outBuffer output{.dst = raw.data(), .size = raw.size(), .pos = 0};
            std::size_t last_result = 1;

            for (std::uint64_t done = 0; done < entry.stored_size;) {
                auto const size = std::min(block_size, entry.stored_size - done);

                if (!read_into(block.data(), size)) {
                    return std::unexpected(make_error(LbfErrorType::io_failed, "short read; " + chunk_name()));
                }

                checksum.update(std::span<std::byte const>{block}.first(size));
                done += size;

                ZSTD_inBuffer input{.src = block.data(), .size = size, .pos = 0};

                while (input.pos < input.size) {
                    last_result = ZSTD_decompressStream(decoder.get(), &output, &input);

                    if (ZSTD_isError(last_result) != 0U) {
                        return std::unexpected(make_error(LbfErrorType::decompression_failed,
                                                          std::format("{}: {}", chunk_name(),
                                                                      ZSTD_getErrorName(last_result))));
                    }

                    // Output full with input left over: the frame is bigger than the table says.
                    if (output.pos == output.size && input.pos < input.size && last_result != 0) {
                        return std::unexpected(make_error(LbfErrorType::decompression_failed,
                                                          "decompresses past its raw size; " + chunk_name()));
                    }

                    if (last_result == 0 && input.pos < input.size) {
                        break; // frame finished; trailing bytes are caught by the size check below
                    }
                }
            }

            if (last_result != 0 || output.pos != raw.size()) {
                return std::unexpected(make_error(LbfErrorType::decompression_failed,
                                                  "decompressed size mismatch; " + chunk_name()));
            }
            break;
        }

        default:
            return std::unexpected(make_error(LbfErrorType::decompression_failed, "unknown compression"));
    }

    if (options_.verify_checksums && checksum.digest() != entry.checksum) {
        return std::unexpected(make_error(LbfErrorType::checksum_mismatch, chunk_name()));
    }

    return raw;
}

auto LbfReader::read_chunk(LbfChunkEntry const &entry) const -> std::expected<std::vector<std::byte>, LbfError> {
    ZoneScopedNC("LbfReader::read_chunk", tracy::Color::Goldenrod);

    if (memory_ == nullptr) {
        return stream_chunk(entry);
    }

    auto stored = read_stored_chunk(entry);

    if (!stored) {
        return std::unexpected(stored.error());
    }

    switch (entry.compression) {
        case LbfCompression::none:
            if (stored->size() != entry.raw_size) {
                return std::unexpected(make_error(LbfErrorType::corrupt_table_of_contents, "raw/stored size mismatch"));
            }
            return stored;

        case LbfCompression::zstd: {
            std::vector<std::byte> raw(entry.raw_size);
            auto const written = ZSTD_decompress(raw.data(), raw.size(), stored->data(), stored->size());

            if (ZSTD_isError(written) != 0U || written != raw.size()) {
                return std::unexpected(make_error(
                        LbfErrorType::decompression_failed,
                        ZSTD_isError(written) != 0U ? std::string_view{ZSTD_getErrorName(written)} : "size mismatch"));
            }

            return raw;
        }
    }

    return std::unexpected(make_error(LbfErrorType::decompression_failed, "unknown compression"));
}
