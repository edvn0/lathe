#pragma once

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

        [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }

    private:
        struct Entry {
            ShaderStage stage = ShaderStage::vertex;
            std::string entry_point;
            std::vector<std::uint32_t> spirv;
        };

        std::unordered_map<std::string, Entry> entries_;
    };

    [[nodiscard]] auto shader_request_key(ShaderCompileRequest const &request) -> std::string;

    auto install_shader_pack(std::shared_ptr<ShaderPack const> pack) -> void;
    [[nodiscard]] auto installed_shader_pack() -> std::shared_ptr<ShaderPack const>;

    auto start_shader_recording() -> void;
    [[nodiscard]] auto shader_recording() -> bool;
    auto record_compiled_shader(ShaderCompileRequest const &request, CompiledShader const &shader) -> void;
    [[nodiscard]] auto finish_shader_recording(std::filesystem::path const &path) -> std::expected<std::size_t, std::string>;

}
