#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/paths.hxx"

// Small engine-owned data files (the editor font and icons, the light gizmo texture) that a packaged game bundles into
// one resource pack instead of shipping loose. read_resource() asks the installed ResourceProvider first and falls
// back to the file under the data root, so a development checkout needs no pack.
class ResourceProvider {
public:
    virtual ~ResourceProvider() = default;

    // The bytes of the file at this root-relative, forward-slash path, if the pack has it.
    [[nodiscard]] virtual auto find(std::string_view logical) const -> std::optional<std::vector<std::byte>> = 0;
};

auto install_resource_provider(std::shared_ptr<ResourceProvider const> provider) -> void;

[[nodiscard]]
auto read_resource(DataPath const &path) -> std::optional<std::vector<std::byte>>;

// While recording, every resource read_resource() returns is kept (logical path -> bytes) so it can be written to a
// pack. Resources are deliberately not noted by Paths::start_access_recording, so they don't also ship loose.
auto start_resource_recording() -> void;
[[nodiscard]] auto finish_resource_recording() -> std::map<std::string, std::vector<std::byte>>;
