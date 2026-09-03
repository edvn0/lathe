#pragma once

#include <slang.h>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace renderer {

    enum class ShaderStage : std::uint8_t {
        vertex,
        fragment,
        compute,
        task,
        mesh,
    };

    struct ShaderDefine {
        std::string name;
        std::string value;
    };

    struct ShaderCompileRequest {
        std::filesystem::path source_path;
        std::string entry_point;
        ShaderStage stage = ShaderStage::vertex;

        std::vector<std::filesystem::path> include_directories{};

        std::vector<ShaderDefine> defines{};

        bool generate_debug_info = true;
        bool optimize = true;
    };

    struct CompiledShader {
        ShaderStage stage = ShaderStage::vertex;
        std::string entry_point;
        std::vector<std::uint32_t> spirv;
    };

    enum class ShaderCompileErrorType : std::uint8_t {
        invalid_argument,
        source_not_found,
        source_read_failed,
        slang_global_session_failed,
        slang_session_failed,
        module_load_failed,
        entry_point_not_found,
        composition_failed,
        link_failed,
        target_code_failed,
        invalid_spirv,
    };

    struct ShaderCompileError {
        ShaderCompileErrorType type = ShaderCompileErrorType::invalid_argument;

        SlangResult result = SLANG_OK;

        std::string diagnostics;
    };

    class SlangCompiler {
    public:
        SlangCompiler() noexcept;

        ~SlangCompiler();

        SlangCompiler(SlangCompiler const &) = delete;

        auto operator=(SlangCompiler const &) -> SlangCompiler & = delete;

        SlangCompiler(SlangCompiler &&other) noexcept;

        auto operator=(SlangCompiler &&other) noexcept -> SlangCompiler &;

        [[nodiscard]]
        static auto create() -> std::expected<SlangCompiler, ShaderCompileError>;

        [[nodiscard]]
        auto compile(ShaderCompileRequest const &request) const -> std::expected<CompiledShader, ShaderCompileError>;

        [[nodiscard]]
        auto valid() const noexcept -> bool;

        auto destroy() noexcept -> void;

    private:
        struct Impl;

        explicit SlangCompiler(std::unique_ptr<Impl> impl) noexcept;

        std::unique_ptr<Impl> impl_;
    };

} // namespace renderer

template<>
struct std::formatter<renderer::ShaderCompileErrorType> : std::formatter<std::string_view> {
    constexpr auto format(renderer::ShaderCompileErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case renderer::ShaderCompileErrorType::invalid_argument:
                    return "invalid_argument";
                case renderer::ShaderCompileErrorType::source_not_found:
                    return "source_not_found";
                case renderer::ShaderCompileErrorType::source_read_failed:
                    return "source_read_failed";
                case renderer::ShaderCompileErrorType::slang_global_session_failed:
                    return "slang_global_session_failed";
                case renderer::ShaderCompileErrorType::slang_session_failed:
                    return "slang_session_failed";
                case renderer::ShaderCompileErrorType::module_load_failed:
                    return "module_load_failed";
                case renderer::ShaderCompileErrorType::entry_point_not_found:
                    return "entry_point_not_found";
                case renderer::ShaderCompileErrorType::composition_failed:
                    return "composition_failed";
                case renderer::ShaderCompileErrorType::link_failed:
                    return "link_failed";
                case renderer::ShaderCompileErrorType::target_code_failed:
                    return "target_code_failed";
                case renderer::ShaderCompileErrorType::invalid_spirv:
                    return "invalid_spirv";
            }

            return "unknown_shader_compile_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
