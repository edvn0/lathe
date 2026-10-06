#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/resources.hxx"
#include "serialisation/lbf_container.hxx"

// The resource pack of a packaged game: an asset-pack .lbf whose FILE chunks hold small data files verbatim, keyed by
// the AssetId of "file:<root-relative path>".
class ResourcePack final : public ResourceProvider {
public:
    [[nodiscard]]
    static auto open(std::filesystem::path const &path) -> std::expected<std::shared_ptr<ResourcePack>, LbfError>;

    // Writes `resources` (logical path -> bytes) to `path`; returns the file size.
    [[nodiscard]]
    static auto write(std::map<std::string, std::vector<std::byte>> resources, std::filesystem::path const &path)
            -> std::expected<std::uint64_t, LbfError>;

    [[nodiscard]] auto find(std::string_view logical) const -> std::optional<std::vector<std::byte>> override;

    [[nodiscard]] auto size() const noexcept -> std::size_t { return reader_.chunks().size(); }

private:
    explicit ResourcePack(LbfReader reader) noexcept : reader_(std::move(reader)) {}

    LbfReader reader_;
};
