#include "serialisation/resource_pack.hxx"

#include <format>

#include "core/logger.hxx"
#include "serialisation/asset_id.hxx"

namespace {
    auto resource_id(std::string_view logical) -> std::uint64_t {
        return asset_id_from_key(std::format("file:{}", logical)).value;
    }
}

auto ResourcePack::open(std::filesystem::path const &path) -> std::expected<std::shared_ptr<ResourcePack>, LbfError> {
    auto reader = LbfReader::open(path);

    if (!reader) {
        return std::unexpected(std::move(reader.error()));
    }

    return std::shared_ptr<ResourcePack>{new ResourcePack{std::move(*reader)}};
}

auto ResourcePack::write(std::map<std::string, std::vector<std::byte>> resources, std::filesystem::path const &path)
        -> std::expected<std::uint64_t, LbfError> {
    LbfWriter writer{LbfFileKind::asset_pack};

    for (auto &[logical, bytes]: resources) {
        writer.add_chunk(LbfChunkInput{
                .type = lbf_chunk::file,
                .id = resource_id(logical),
                .version = 0,
                .payload = std::move(bytes),
        });
    }

    return writer.write_file(path);
}

auto ResourcePack::find(std::string_view logical) const -> std::optional<std::vector<std::byte>> {
    auto const *entry = reader_.find(lbf_chunk::file, resource_id(logical));

    if (entry == nullptr) {
        return std::nullopt;
    }

    auto bytes = reader_.read_chunk(*entry);

    if (!bytes) {
        error("[ResourcePack] Could not read '{}' from '{}'", logical, reader_.path().string());

        return std::nullopt;
    }

    return std::move(*bytes);
}
