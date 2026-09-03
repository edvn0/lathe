#include "gpu/shader_binary_cache.hxx"

#include <format>
#include <fstream>
#include <system_error>

#include "core/logger.hxx"

namespace {
    [[nodiscard]] auto hash_key(std::string_view key) noexcept -> std::uint64_t {
        auto hash = std::uint64_t{0xcbf29ce484222325ULL};

        for (auto const byte: key) {
            hash ^= static_cast<std::uint8_t>(byte);
            hash *= 0x100000001b3ULL;
        }

        return hash;
    }

    [[nodiscard]] auto uuid_directory_name(std::array<std::uint8_t, VK_UUID_SIZE> const &uuid, std::uint32_t version)
            -> std::string {
        std::string name;
        name.reserve((VK_UUID_SIZE * 2) + 16);

        for (auto const byte: uuid) {
            name += std::format("{:02x}", byte);
        }

        name += std::format("_v{}", version);

        return name;
    }
} // namespace

auto ShaderBinaryCache::create(std::filesystem::path directory, std::array<std::uint8_t, VK_UUID_SIZE> binary_uuid,
                               std::uint32_t binary_version)
        -> std::expected<ShaderBinaryCache, ShaderBinaryCacheError> {
    if (directory.empty()) {
        return std::unexpected(ShaderBinaryCacheError{.type = ShaderBinaryCacheErrorType::invalid_argument});
    }

    auto full_directory = std::move(directory) / uuid_directory_name(binary_uuid, binary_version);

    std::error_code error_code;
    std::filesystem::create_directories(full_directory, error_code);

    if (error_code) {
        return std::unexpected(ShaderBinaryCacheError{
                .type = ShaderBinaryCacheErrorType::directory_creation_failed,
                .context = ErrorContext{.message = FlyString{error_code.message()}},
        });
    }

    ShaderBinaryCache cache;
    cache.directory_ = std::move(full_directory);

    return cache;
}

auto ShaderBinaryCache::path_for(std::string_view cache_key) const -> std::filesystem::path {
    const auto path = std::format("{}.bin", std::to_string(hash_key(cache_key)));
    return directory_ / path;
}

auto ShaderBinaryCache::find(std::string_view cache_key) const -> std::optional<std::vector<std::byte>> {
    auto const path = path_for(cache_key);

    std::error_code error_code;
    auto const size = std::filesystem::file_size(path, error_code);

    if (error_code || size == 0) {
        return std::nullopt;
    }

    auto file = std::ifstream{path, std::ios::binary};

    if (!file.is_open()) {
        return std::nullopt;
    }

    std::vector<std::byte> data(size);
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));

    if (!file) {
        return std::nullopt;
    }

    return data;
}

auto ShaderBinaryCache::store(std::string_view cache_key, std::span<std::byte const> binary) const -> void {
    auto const path = path_for(cache_key);
    auto const temp_path = path.string() + ".tmp";

    {
        auto file = std::ofstream{temp_path, std::ios::binary | std::ios::trunc};

        if (!file.is_open()) {
            warn("ShaderBinaryCache: failed to open '{}' for write", temp_path);
            return;
        }

        file.write(reinterpret_cast<char const *>(binary.data()), static_cast<std::streamsize>(binary.size()));

        if (!file) {
            warn("ShaderBinaryCache: failed writing '{}'", temp_path);
            return;
        }
    }

    std::error_code error_code;
    std::filesystem::rename(temp_path, path, error_code);

    if (error_code) {
        warn("ShaderBinaryCache: failed to finalize '{}': {}", path.string(), error_code.message());
    }
}
