#include "slang_compiler.hxx"

#include <slang-com-ptr.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <slang.h>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include "slang_library.hxx"

namespace renderer {
    namespace {
        constexpr auto spirv_magic = std::uint32_t{0x07230203};

        [[nodiscard]]
        auto diagnostics_from_blob(slang::IBlob *blob) -> std::string {
            if (blob == nullptr) {
                return {};
            }

            auto const *data = static_cast<char const *>(blob->getBufferPointer());

            auto const size = blob->getBufferSize();

            if (data == nullptr || size == 0) {
                return {};
            }

            return std::string{
                    data,
                    data + size,
            };
        }

        auto append_diagnostics(std::string &destination, slang::IBlob *blob) -> void {
            auto diagnostics = diagnostics_from_blob(blob);

            if (diagnostics.empty()) {
                return;
            }

            if (!destination.empty() && destination.back() != '\n') {
                destination.push_back('\n');
            }

            destination.append(diagnostics);
        }

        [[nodiscard]]
        auto make_error(ShaderCompileErrorType type, SlangResult result, std::string diagnostics)
                -> ShaderCompileError {
            return ShaderCompileError{
                    .type = type,
                    .result = result,
                    .diagnostics = std::move(diagnostics),
            };
        }

        [[nodiscard]]
        auto to_slang_stage(ShaderStage stage) noexcept -> SlangStage {
            switch (stage) {
                case ShaderStage::vertex:
                    return SLANG_STAGE_VERTEX;

                case ShaderStage::fragment:
                    return SLANG_STAGE_FRAGMENT;

                case ShaderStage::compute:
                    return SLANG_STAGE_COMPUTE;

                case ShaderStage::task:
                    return SLANG_STAGE_AMPLIFICATION;

                case ShaderStage::mesh:
                    return SLANG_STAGE_MESH;
            }

            return SLANG_STAGE_NONE;
        }

        [[nodiscard]]
        auto read_source_file(std::filesystem::path const &source_path)
                -> std::expected<std::string, ShaderCompileError> {
            std::error_code error_code;

            auto const exists = std::filesystem::exists(source_path, error_code);

            if (error_code || !exists) {
                return std::unexpected{make_error(ShaderCompileErrorType::source_not_found, SLANG_OK,
                                                  "Shader source does not exist: " + source_path.string())};
            }

            auto const regular_file = std::filesystem::is_regular_file(source_path, error_code);

            if (error_code || !regular_file) {
                return std::unexpected{make_error(ShaderCompileErrorType::source_not_found, SLANG_OK,
                                                  "Shader source is not a regular file: " + source_path.string())};
            }

            auto file = std::ifstream{
                    source_path,
                    std::ios::binary,
            };

            if (!file.is_open()) {
                return std::unexpected{make_error(ShaderCompileErrorType::source_read_failed, SLANG_OK,
                                                  "Failed to open shader source: " + source_path.string())};
            }

            auto source = std::string{
                    std::istreambuf_iterator<char>{file},
                    std::istreambuf_iterator<char>{},
            };

            if (file.bad()) {
                return std::unexpected{make_error(ShaderCompileErrorType::source_read_failed, SLANG_OK,
                                                  "Failed while reading shader source: " + source_path.string())};
            }

            return source;
        }

        [[nodiscard]]
        auto make_integer_option(slang::CompilerOptionName name, std::int32_t value) noexcept
                -> slang::CompilerOptionEntry {
            return slang::CompilerOptionEntry{
                    .name = name,
                    .value =
                            slang::CompilerOptionValue{
                                    .kind = slang::CompilerOptionValueKind::Int,
                                    .intValue0 = value,
                                    .intValue1 = 0,
                                    .stringValue0 = nullptr,
                                    .stringValue1 = nullptr,
                            },
            };
        }

        [[nodiscard]]
        auto validate_request(ShaderCompileRequest const &request) -> std::expected<void, ShaderCompileError> {
            if (request.source_path.empty()) {
                return std::unexpected{
                        make_error(ShaderCompileErrorType::invalid_argument, SLANG_OK, "Shader source path is empty.")};
            }

            if (request.entry_point.empty()) {
                return std::unexpected{make_error(ShaderCompileErrorType::invalid_argument, SLANG_OK,
                                                  "Shader entry-point name is empty.")};
            }

            if (to_slang_stage(request.stage) == SLANG_STAGE_NONE) {
                return std::unexpected{
                        make_error(ShaderCompileErrorType::invalid_argument, SLANG_OK, "Shader stage is invalid.")};
            }

            for (auto const &define: request.defines) {
                if (define.name.empty()) {
                    return std::unexpected{make_error(ShaderCompileErrorType::invalid_argument, SLANG_OK,
                                                      "Shader define has an empty name.")};
                }
            }

            return {};
        }
    } // namespace

    struct SlangCompiler::Impl {
        SlangLibrary library;

        Slang::ComPtr<slang::IGlobalSession> global_session;
    };

    SlangCompiler::SlangCompiler() noexcept = default;

    SlangCompiler::SlangCompiler(std::unique_ptr<Impl> impl) noexcept : impl_{std::move(impl)} {}

    SlangCompiler::~SlangCompiler() { destroy(); }

    SlangCompiler::SlangCompiler(SlangCompiler &&other) noexcept : impl_{std::move(other.impl_)} {}

    auto SlangCompiler::operator=(SlangCompiler &&other) noexcept -> SlangCompiler & {
        if (this == &other) {
            return *this;
        }

        destroy();

        impl_ = std::move(other.impl_);

        return *this;
    }

    auto SlangCompiler::create() -> std::expected<SlangCompiler, ShaderCompileError> {
        auto library_result = SlangLibrary::create_from_executable_directory();
        if (!library_result) {
            auto error = std::move(library_result.error());
            return std::unexpected{ShaderCompileError{
                    .type = ShaderCompileErrorType::slang_global_session_failed,
                    .result = error.result,
                    .diagnostics = std::move(error.diagnostics),
            }};
        }
        auto impl = std::make_unique<Impl>();
        impl->library = std::move(*library_result);
        auto const result = impl->library.create_global_session(impl->global_session.writeRef());
        if (SLANG_FAILED(result) || impl->global_session == nullptr) {
            impl->library.destroy();
            return std::unexpected{ShaderCompileError{
                    .type = ShaderCompileErrorType::slang_global_session_failed,
                    .result = result,
                    .diagnostics = "Failed to create the Slang "
                                   "global session through "
                                   "slang.dll.",
            }};
        }
        return SlangCompiler{
                std::move(impl),
        };
    }

    auto SlangCompiler::compile(ShaderCompileRequest const &request) const
            -> std::expected<CompiledShader, ShaderCompileError> {
        if (!valid()) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_global_session_failed,
                                              SLANG_E_NOT_AVAILABLE, "SlangCompiler is not initialized.")};
        }

        auto validation = validate_request(request);

        if (!validation) {
            return std::unexpected{std::move(validation.error())};
        }

        auto source_result = read_source_file(request.source_path);

        if (!source_result) {
            return std::unexpected{std::move(source_result.error())};
        }

        auto source = std::move(*source_result);

        /*
         * All strings referenced by SessionDesc must remain alive until
         * createSession() returns. The vectors below provide that storage.
         */
        auto search_path_storage = std::vector<std::string>{};

        search_path_storage.reserve(request.include_directories.size() + 1);

        auto const parent_path = request.source_path.parent_path();

        if (!parent_path.empty()) {
            search_path_storage.push_back(parent_path.string());
        }

        for (auto const &include_directory: request.include_directories) {
            search_path_storage.push_back(include_directory.string());
        }

        auto search_paths = std::vector<char const *>{};

        search_paths.reserve(search_path_storage.size());

        for (auto const &search_path: search_path_storage) {
            search_paths.push_back(search_path.c_str());
        }

        /*
         * Macro descriptor strings point directly into request.defines.
         * They remain valid for the duration of this call.
         */
        auto macros = std::vector<slang::PreprocessorMacroDesc>{};

        macros.reserve(request.defines.size());

        for (auto const &define: request.defines) {
            macros.push_back(slang::PreprocessorMacroDesc{
                    .name = define.name.c_str(),
                    .value = define.value.c_str(),
            });
        }

        auto options = std::vector<slang::CompilerOptionEntry>{};

        options.reserve(5);

        options.push_back(make_integer_option(slang::CompilerOptionName::EmitSpirvDirectly, 1));

        /*
         * Preserve the selected Slang function name in OpEntryPoint.
         * This lets PipelineStorage use CompiledShader::entry_point
         * directly as VkPipelineShaderStageCreateInfo::pName.
         */
        options.push_back(make_integer_option(slang::CompilerOptionName::VulkanUseEntryPointName, 1));

        options.push_back(make_integer_option(slang::CompilerOptionName::Optimization,
                                              request.optimize ? SLANG_OPTIMIZATION_LEVEL_MAXIMAL
                                                               : SLANG_OPTIMIZATION_LEVEL_NONE));

        options.push_back(make_integer_option(slang::CompilerOptionName::DebugInformation,
                                              request.generate_debug_info ? SLANG_DEBUG_INFO_LEVEL_STANDARD
                                                                          : SLANG_DEBUG_INFO_LEVEL_NONE));

        /*
         * Keep Slang's built-in SPIR-V validation enabled.
         * SkipSPIRVValidation = 0 is explicit here so the compiler
         * contract remains obvious.
         */
        options.push_back(make_integer_option(slang::CompilerOptionName::SkipSPIRVValidation, 0));

        options.push_back(make_integer_option(slang::CompilerOptionName::MatrixLayoutColumn, 1));
        options.push_back(make_integer_option(slang::CompilerOptionName::MatrixLayoutRow, 0));

        auto target_description = slang::TargetDesc{
                .structureSize = sizeof(slang::TargetDesc),
                .format = SLANG_SPIRV,
                .profile = impl_->global_session->findProfile("spirv_1_6"),
                .flags = 0,
                .floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT,
                .lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_DEFAULT,
                .forceGLSLScalarBufferLayout = true,
                .compilerOptionEntries = nullptr,
                .compilerOptionEntryCount = 0,
        };

        if (target_description.profile == SLANG_PROFILE_UNKNOWN) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_session_failed, SLANG_E_NOT_AVAILABLE,
                                              "Slang does not recognize the "
                                              "\"spirv_1_6\" target profile.")};
        }

        auto session_description = slang::SessionDesc{
                .structureSize = sizeof(slang::SessionDesc),
                .targets = &target_description,
                .targetCount = 1,
                .flags = 0,
                .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
                .searchPaths = search_paths.empty() ? nullptr : search_paths.data(),
                .searchPathCount = static_cast<SlangInt>(search_paths.size()),
                .preprocessorMacros = macros.empty() ? nullptr : macros.data(),
                .preprocessorMacroCount = static_cast<SlangInt>(macros.size()),
                .fileSystem = nullptr,
                .enableEffectAnnotations = false,
                .allowGLSLSyntax = false,
                .compilerOptionEntries = options.data(),
                .compilerOptionEntryCount = static_cast<std::uint32_t>(options.size()),
        };

        auto session = Slang::ComPtr<slang::ISession>{};

        auto result = impl_->global_session->createSession(session_description, session.writeRef());

        if (SLANG_FAILED(result) || session == nullptr) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_session_failed, result,
                                              "IGlobalSession::createSession() "
                                              "failed.")};
        }

        auto diagnostics = std::string{};

        /*
         * loadModuleFromSourceString() lets the public API use an exact
         * filesystem path instead of requiring callers to convert paths
         * into Slang module names.
         *
         * Sessions are currently per request, so a stem-based module name
         * cannot collide with another module loaded by this compiler call.
         */
        auto module_name = request.source_path.stem().string();

        if (module_name.empty()) {
            module_name = "shader";
        }

        auto source_path = request.source_path.string();

        auto module_diagnostics = Slang::ComPtr<slang::IBlob>{};

        auto module = Slang::ComPtr<slang::IModule>{session->loadModuleFromSourceString(
                module_name.c_str(), source_path.c_str(), source.c_str(), module_diagnostics.writeRef())};

        append_diagnostics(diagnostics, module_diagnostics);

        if (module == nullptr) {
            return std::unexpected{
                    make_error(ShaderCompileErrorType::module_load_failed, SLANG_FAIL, std::move(diagnostics))};
        }

        auto entry_point = Slang::ComPtr<slang::IEntryPoint>{};

        auto entry_point_diagnostics = Slang::ComPtr<slang::IBlob>{};

        result = module->findAndCheckEntryPoint(request.entry_point.c_str(), to_slang_stage(request.stage),
                                                entry_point.writeRef(), entry_point_diagnostics.writeRef());

        append_diagnostics(diagnostics, entry_point_diagnostics);

        if (SLANG_FAILED(result) || entry_point == nullptr) {
            if (diagnostics.empty()) {
                diagnostics = "Entry point \"" + request.entry_point +
                              "\" was not found or does not "
                              "match the requested shader stage.";
            }

            return std::unexpected{
                    make_error(ShaderCompileErrorType::entry_point_not_found, result, std::move(diagnostics))};
        }

        auto components = std::array<slang::IComponentType *, 2>{
                module.get(),
                entry_point.get(),
        };

        auto composed_program = Slang::ComPtr<slang::IComponentType>{};

        auto composition_diagnostics = Slang::ComPtr<slang::IBlob>{};

        result = session->createCompositeComponentType(components.data(), static_cast<SlangInt>(components.size()),
                                                       composed_program.writeRef(), composition_diagnostics.writeRef());

        append_diagnostics(diagnostics, composition_diagnostics);

        if (SLANG_FAILED(result) || composed_program == nullptr) {
            return std::unexpected{
                    make_error(ShaderCompileErrorType::composition_failed, result, std::move(diagnostics))};
        }

        auto linked_program = Slang::ComPtr<slang::IComponentType>{};

        auto link_diagnostics = Slang::ComPtr<slang::IBlob>{};

        result = composed_program->link(linked_program.writeRef(), link_diagnostics.writeRef());

        append_diagnostics(diagnostics, link_diagnostics);

        if (SLANG_FAILED(result) || linked_program == nullptr) {
            return std::unexpected{make_error(ShaderCompileErrorType::link_failed, result, std::move(diagnostics))};
        }

        auto target_code = Slang::ComPtr<slang::IBlob>{};

        auto target_diagnostics = Slang::ComPtr<slang::IBlob>{};

        result = linked_program->getEntryPointCode(0, 0, target_code.writeRef(), target_diagnostics.writeRef());

        append_diagnostics(diagnostics, target_diagnostics);

        if (SLANG_FAILED(result) || target_code == nullptr) {
            return std::unexpected{
                    make_error(ShaderCompileErrorType::target_code_failed, result, std::move(diagnostics))};
        }

        auto const byte_size = target_code->getBufferSize();

        auto const *byte_data = static_cast<std::byte const *>(target_code->getBufferPointer());

        if (byte_data == nullptr || byte_size == 0) {
            return std::unexpected{make_error(ShaderCompileErrorType::invalid_spirv, SLANG_FAIL,
                                              "Slang returned an empty SPIR-V "
                                              "blob.")};
        }

        if ((byte_size % sizeof(std::uint32_t)) != 0) {
            return std::unexpected{make_error(ShaderCompileErrorType::invalid_spirv, SLANG_FAIL,
                                              "SPIR-V byte size is not aligned "
                                              "to four bytes.")};
        }

        auto spirv = std::vector<std::uint32_t>(byte_size / sizeof(std::uint32_t));

        std::memcpy(spirv.data(), byte_data, byte_size);

        if (spirv.empty() || spirv.front() != spirv_magic) {
            return std::unexpected{make_error(ShaderCompileErrorType::invalid_spirv, SLANG_FAIL,
                                              "Slang output does not begin with "
                                              "the SPIR-V magic number.")};
        }

        return CompiledShader{
                .stage = request.stage,
                .entry_point = request.entry_point,
                .spirv = std::move(spirv),
        };
    }

    auto SlangCompiler::valid() const noexcept -> bool { return impl_ != nullptr && impl_->global_session != nullptr; }

    auto SlangCompiler::destroy() noexcept -> void {
        if (impl_ == nullptr) {
            return;
        }

        impl_->global_session = nullptr;
        impl_->library.destroy();

        impl_.reset();
    }
} // namespace renderer
