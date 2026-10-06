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

    // Precompiled SPIR-V keyed by shader request, so a shipped game needs neither the Slang compiler nor the shader
    // sources. Built by recording what a run compiles (see start_shader_recording), loaded at startup.
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

    // Stable across installs: the source's root-relative path, entry point, stage, defines and compile flags.
    [[nodiscard]] auto shader_request_key(ShaderCompileRequest const &request) -> std::string;

    // The pack compile() consults before invoking Slang. A request it lacks falls through to Slang when available.
    auto install_shader_pack(std::shared_ptr<ShaderPack const> pack) -> void;
    [[nodiscard]] auto installed_shader_pack() -> std::shared_ptr<ShaderPack const>;

    // While recording, every shader Slang compiles is also added to an in-memory pack.
    auto start_shader_recording() -> void;
    [[nodiscard]] auto shader_recording() -> bool;
    auto record_compiled_shader(ShaderCompileRequest const &request, CompiledShader const &shader) -> void;
    [[nodiscard]] auto finish_shader_recording(std::filesystem::path const &path) -> std::expected<std::size_t, std::string>;

} // namespace renderer
