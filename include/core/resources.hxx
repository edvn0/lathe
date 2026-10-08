#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/paths.hxx"

class ResourceProvider {
public:
    virtual ~ResourceProvider() = default;

    [[nodiscard]] virtual auto find(std::string_view logical) const -> std::optional<std::vector<std::byte>> = 0;
};

auto install_resource_provider(std::shared_ptr<ResourceProvider const> provider) -> void;

[[nodiscard]]
auto read_resource(DataPath const &path) -> std::optional<std::vector<std::byte>>;

auto start_resource_recording() -> void;
[[nodiscard]] auto finish_resource_recording() -> std::map<std::string, std::vector<std::byte>>;
