#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "assets/slang_compiler.hxx"

namespace renderer {

    class ShaderPack {
    public:
        [[nodiscard]] static auto load(std::filesystem::path const &path) -> std::expected<ShaderPack, std::string>;

        [[nodiscard]] auto save(std::filesystem::path const &path) const -> std::expected<void, std::string>;

        [[nodiscard]] auto find(std::string_view key) const -> std::optional<CompiledShader>;

        auto add(std::string key, CompiledShader const &shader) -> void;

        // Hash of the shader sources the pack was compiled from; 0 means the pack was recorded without sources and
        // is never considered stale.
        [[nodiscard]] auto source_hash() const noexcept -> std::uint64_t { return source_hash_; }
        auto set_source_hash(std::uint64_t hash) noexcept -> void { source_hash_ = hash; }

        // False when the sources in `shader_directory` differ from the ones the pack was compiled from. A missing
        // directory (an installed game ships the pack without sources) or an unhashed pack counts as current.
        [[nodiscard]] auto matches_sources(std::filesystem::path const &shader_directory) const -> bool;

        [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }

    private:
        struct Entry {
            ShaderStage stage = ShaderStage::vertex;
            std::string entry_point;
            std::vector<std::uint32_t> spirv;
        };

        std::uint64_t source_hash_ = 0;
        std::unordered_map<std::string, Entry> entries_;
    };

    // FNV-1a over every .slang file and variants.txt under the directory (relative path and contents, in path order), or nullopt when
    // the directory does not exist. Shaders import each other, so any source change invalidates the whole pack.
    [[nodiscard]] auto hash_shader_sources(std::filesystem::path const &shader_directory) -> std::optional<std::uint64_t>;

    [[nodiscard]] auto shader_request_key(ShaderCompileRequest const &request) -> std::string;

    auto install_shader_pack(std::shared_ptr<ShaderPack const> pack) -> void;
    [[nodiscard]] auto installed_shader_pack() -> std::shared_ptr<ShaderPack const>;

    auto start_shader_recording() -> void;
    [[nodiscard]] auto shader_recording() -> bool;
    auto record_compiled_shader(ShaderCompileRequest const &request, CompiledShader const &shader) -> void;
    [[nodiscard]] auto finish_shader_recording(std::filesystem::path const &path) -> std::expected<std::size_t, std::string>;

}
