#include "assets/slang_compiler.hxx"
#include "core/perf_events.hxx"

#include <slang-com-ptr.h>

#include <spirv-tools/optimizer.hpp>

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <future>
#include <optional>
#include <unordered_map>
#include <iterator>
#include <slang.h>
#include <span>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "assets/shader_pack.hxx"
#include "assets/slang_library.hxx"
#include "core/logger.hxx"

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
        auto make_string_option(slang::CompilerOptionName name, char const *value) noexcept
                -> slang::CompilerOptionEntry {
            return slang::CompilerOptionEntry{
                    .name = name,
                    .value =
                            slang::CompilerOptionValue{
                                    .kind = slang::CompilerOptionValueKind::String,
                                    .intValue0 = 0,
                                    .intValue1 = 0,
                                    .stringValue0 = value,
                                    .stringValue1 = nullptr,
                            },
            };
        }

        [[nodiscard]]
        auto validate_request(ShaderCompileRequest const &request) -> std::expected<void, ShaderCompileError> {
            if (request.source_path.logical().empty()) {
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
    }

    namespace spirv_opt {
        [[nodiscard]]
        auto run(std::vector<std::uint32_t> spirv, bool optimize_for_size)
                -> std::expected<std::vector<std::uint32_t>, ShaderCompileError> {
            auto optimizer = spvtools::Optimizer{SPV_ENV_VULKAN_1_3};

            optimizer.SetMessageConsumer(
                    [](spv_message_level_t, char const *source, spv_position_t const &position, char const *message) {
                        warn("spirv-opt: {} ({}:{}:{})", message, source != nullptr ? source : "<unknown>",
                             position.line, position.column);
                    });

            if (optimize_for_size) {
                optimizer.RegisterSizePasses();
            } else {
                optimizer.RegisterPerformancePasses();
            }

            auto optimized = std::vector<std::uint32_t>{};

            auto options = spvtools::OptimizerOptions{};

            options.set_run_validator(false);

            if (!optimizer.Run(spirv.data(), spirv.size(), &optimized, options)) {
                return std::unexpected{make_error(ShaderCompileErrorType::invalid_spirv, SLANG_FAIL,
                                                  "spirv-opt failed to optimize the generated SPIR-V module.")};
            }

            return optimized;
        }
    }

    struct SlangCompiler::Impl {
        using Result = std::expected<CompiledShader, ShaderCompileError>;

        SlangLibrary library;

        // Slang's global session is not safe to share between threads, so every concurrent compile leases its own.
        // Creating one costs ~140 ms and Slang serializes creation, so the count is capped and callers wait for a
        // free session instead of creating more.
        std::mutex session_mutex;
        std::condition_variable session_available;
        std::vector<Slang::ComPtr<slang::IGlobalSession>> idle_sessions;
        std::size_t session_count = 0;
        std::size_t max_sessions = 1;

        std::mutex prefetch_mutex;
        std::unordered_map<std::string, std::shared_future<Result>> prefetched;
        // Dedicated threads rather than the shared pool: pool workers block on prefetched results, and the producers
        // must never queue behind those blocked consumers.
        std::vector<std::thread> prefetch_workers;
    };

    namespace {
        using perf_clock = std::chrono::steady_clock;

        [[nodiscard]] auto elapsed_ms(perf_clock::time_point since) -> double {
            return std::chrono::duration<double, std::milli>(perf_clock::now() - since).count();
        }

        // Requests with the same group key can share one session and one loaded module.
        [[nodiscard]] auto group_key(ShaderCompileRequest const &request) -> std::string {
            auto key = std::format("{}|o{}d{}", request.source_path.absolute().string(), request.optimize ? 1 : 0,
                                   request.generate_debug_info ? 1 : 0);

            for (auto const &define: request.defines) {
                key += std::format("|{}={}", define.name, define.value);
            }

            for (auto const &directory: request.include_directories) {
                key += std::format("|I{}", directory.string());
            }

            return key;
        }

        template<typename Request>
        [[nodiscard]] auto group_requests(std::span<Request const> requests) -> std::vector<std::vector<std::size_t>> {
            auto groups = std::vector<std::vector<std::size_t>>{};
            auto group_of = std::unordered_map<std::string, std::size_t>{};

            for (std::size_t i = 0; i < requests.size(); ++i) {
                auto const [it, inserted] = group_of.try_emplace(group_key(requests[i]), groups.size());

                if (inserted) {
                    groups.emplace_back();
                }

                groups[it->second].push_back(i);
            }

            return groups;
        }
    }

    // A global session on loan from the compiler, handed back when the lease ends.
    class SlangCompiler::SessionLease {
    public:
        SessionLease(Impl &impl, Slang::ComPtr<slang::IGlobalSession> session) :
            impl_{&impl}, session_{std::move(session)} {}
        SessionLease(SessionLease const &) = delete;
        auto operator=(SessionLease const &) -> SessionLease & = delete;
        SessionLease(SessionLease &&other) noexcept :
            impl_{std::exchange(other.impl_, nullptr)}, session_{std::move(other.session_)} {}
        auto operator=(SessionLease &&) -> SessionLease & = delete;

        ~SessionLease() {
            if (impl_ == nullptr) {
                return;
            }

            {
                std::scoped_lock const lock{impl_->session_mutex};
                impl_->idle_sessions.push_back(std::move(session_));
            }

            impl_->session_available.notify_one();
        }

        [[nodiscard]] auto get() const noexcept -> slang::IGlobalSession * { return session_.get(); }

    private:
        Impl *impl_;
        Slang::ComPtr<slang::IGlobalSession> session_;
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
        auto const create_start = perf_clock::now();
        auto library_result = SlangLibrary::create_from_executable_directory();
        if (!library_result) {
            auto error = std::move(library_result.error());
            return std::unexpected{ShaderCompileError{
                    .type = ShaderCompileErrorType::slang_global_session_failed,
                    .result = error.result,
                    .diagnostics = std::move(error.diagnostics),
            }};
        }
        auto const library_ms = elapsed_ms(create_start);
        auto impl = std::make_unique<Impl>();
        impl->library = std::move(*library_result);
        auto const session_start = perf_clock::now();
        auto session = Slang::ComPtr<slang::IGlobalSession>{};
        auto const result = impl->library.create_global_session(session.writeRef());
        debug("[Perf] SlangCompiler::create: library load {:.2f} ms, global session {:.2f} ms", library_ms,
              elapsed_ms(session_start));
        if (SLANG_FAILED(result) || session == nullptr) {
            impl->library.destroy();
            return std::unexpected{ShaderCompileError{
                    .type = ShaderCompileErrorType::slang_global_session_failed,
                    .result = result,
                    .diagnostics = "Failed to create the Slang "
                                   "global session through "
                                   "slang.dll.",
            }};
        }
        impl->idle_sessions.push_back(std::move(session));
        impl->session_count = 1;
        impl->max_sessions = std::clamp<std::size_t>(std::thread::hardware_concurrency(), 2, 4);
        return SlangCompiler{
                std::move(impl),
        };
    }

    auto SlangCompiler::acquire_session() const -> std::expected<SessionLease, ShaderCompileError> {
        {
            auto lock = std::unique_lock{impl_->session_mutex};

            while (true) {
                if (!impl_->idle_sessions.empty()) {
                    auto session = std::move(impl_->idle_sessions.back());
                    impl_->idle_sessions.pop_back();
                    return SessionLease{*impl_, std::move(session)};
                }

                if (impl_->session_count < impl_->max_sessions) {
                    ++impl_->session_count;
                    break;
                }

                impl_->session_available.wait(lock);
            }
        }

        auto const start = perf_clock::now();
        auto session = Slang::ComPtr<slang::IGlobalSession>{};
        auto const result = impl_->library.create_global_session(session.writeRef());

        if (SLANG_FAILED(result) || session == nullptr) {
            {
                std::scoped_lock const lock{impl_->session_mutex};
                --impl_->session_count;
            }

            impl_->session_available.notify_one();

            return std::unexpected{make_error(ShaderCompileErrorType::slang_global_session_failed, result,
                                              "Failed to create an additional Slang global session.")};
        }

        debug("[Perf] Slang: created an extra global session in {:.2f} ms", elapsed_ms(start));

        return SessionLease{*impl_, std::move(session)};
    }

    auto SlangCompiler::compile(ShaderCompileRequest const &request) const
            -> std::expected<CompiledShader, ShaderCompileError> {
        if (auto const pack = installed_shader_pack()) {
            if (auto precompiled = pack->find(shader_request_key(request))) {
                return std::move(*precompiled);
            }
        }

        if (impl_ != nullptr) {
            auto pending = std::optional<std::shared_future<Impl::Result>>{};

            {
                std::scoped_lock const lock{impl_->prefetch_mutex};

                // A prefetched result is consumed once, so a hot reload recompiles the edited source.
                if (auto const it = impl_->prefetched.find(shader_request_key(request)); it != impl_->prefetched.end()) {
                    pending = std::move(it->second);
                    impl_->prefetched.erase(it);
                }
            }

            if (pending) {
                auto const wait_start = perf_clock::now();
                auto result = pending->get();
                debug("[Perf] Slang prefetch '{}' [{}]: waited {:.2f} ms", request.source_path.logical(),
                      request.entry_point, elapsed_ms(wait_start));
                return result;
            }
        }

        auto compiled = compile_with_slang(request);

        if (compiled && shader_recording()) {
            record_compiled_shader(request, *compiled);
        }

        return compiled;
    }

    auto SlangCompiler::compile_with_slang(ShaderCompileRequest const &request) const
            -> std::expected<CompiledShader, ShaderCompileError> {
        auto const requests = std::array{&request};
        auto results = compile_group(requests);
        return std::move(results.front());
    }

    auto SlangCompiler::prefetch(std::span<ShaderCompileRequest const> requests) const -> void {
        if (!valid()) {
            return;
        }

        auto const pack = installed_shader_pack();
        auto wanted = std::vector<ShaderCompileRequest>{};
        auto promises = std::vector<std::promise<Impl::Result>>{};

        {
            std::scoped_lock const lock{impl_->prefetch_mutex};

            for (auto const &request: requests) {
                auto key = shader_request_key(request);

                if ((pack != nullptr && pack->find(key).has_value()) || impl_->prefetched.contains(key)) {
                    continue;
                }

                auto &promise = promises.emplace_back();
                impl_->prefetched.emplace(std::move(key), promise.get_future().share());
                wanted.push_back(request);
            }
        }

        if (wanted.empty()) {
            return;
        }

        struct PrefetchJob {
            std::vector<ShaderCompileRequest> requests;
            std::vector<std::promise<Impl::Result>> promises;
            std::vector<std::vector<std::size_t>> groups;
            std::atomic<std::size_t> next_group{0};
        };

        auto job = std::make_shared<PrefetchJob>();
        job->groups = group_requests(std::span<ShaderCompileRequest const>{wanted});

        // Start the most expensive groups first so the biggest one is not left for last on an otherwise idle pool.
        // Cost scales with the source size and with every entry point compiled from the loaded module.
        {
            auto const cost = [&wanted](std::vector<std::size_t> const &group) {
                std::error_code error;
                auto const size = std::filesystem::file_size(wanted[group.front()].source_path.absolute(), error);
                return (error ? std::uintmax_t{0} : size) * group.size();
            };

            std::ranges::stable_sort(job->groups, [&cost](auto const &lhs, auto const &rhs) { return cost(lhs) > cost(rhs); });
        }
        job->requests = std::move(wanted);
        job->promises = std::move(promises);

        auto const worker_count = std::min(impl_->max_sessions, job->groups.size());

        std::scoped_lock const lock{impl_->prefetch_mutex};

        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            impl_->prefetch_workers.emplace_back([this, job] {
                while (true) {
                    auto const group_index = job->next_group.fetch_add(1, std::memory_order_relaxed);

                    if (group_index >= job->groups.size()) {
                        return;
                    }

                    auto const &group = job->groups[group_index];
                    auto entries = std::vector<ShaderCompileRequest const *>{};

                    for (auto const index: group) {
                        entries.push_back(&job->requests[index]);
                    }

                    auto results = compile_group(entries);

                    for (std::size_t k = 0; k < group.size(); ++k) {
                        if (results[k] && shader_recording()) {
                            record_compiled_shader(*entries[k], *results[k]);
                        }

                        job->promises[group[k]].set_value(std::move(results[k]));
                    }
                }
            });
        }
    }

    namespace {
        [[nodiscard]] auto from_slang_stage(SlangStage stage) noexcept -> std::optional<ShaderStage> {
            switch (stage) {
                case SLANG_STAGE_VERTEX:
                    return ShaderStage::vertex;
                case SLANG_STAGE_FRAGMENT:
                    return ShaderStage::fragment;
                case SLANG_STAGE_COMPUTE:
                    return ShaderStage::compute;
                case SLANG_STAGE_AMPLIFICATION:
                    return ShaderStage::task;
                case SLANG_STAGE_MESH:
                    return ShaderStage::mesh;
                default:
                    return std::nullopt;
            }
        }
    }

    auto SlangCompiler::discover_entry_points(DataPath const &source_path,
                                              std::span<std::filesystem::path const> include_directories) const
            -> std::expected<std::vector<DiscoveredEntryPoint>, ShaderCompileError> {
        if (!valid()) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_global_session_failed,
                                              SLANG_E_NOT_AVAILABLE, "SlangCompiler is not initialized.")};
        }

        auto source = read_source_file(source_path.absolute());

        if (!source) {
            return std::unexpected{std::move(source.error())};
        }

        auto lease = acquire_session();

        if (!lease) {
            return std::unexpected{std::move(lease.error())};
        }

        auto search_path_storage = std::vector<std::string>{source_path.absolute().parent_path().string()};

        for (auto const &include_directory: include_directories) {
            search_path_storage.push_back(include_directory.string());
        }

        auto search_paths = std::vector<char const *>{};

        for (auto const &search_path: search_path_storage) {
            search_paths.push_back(search_path.c_str());
        }

        auto options = std::array{make_integer_option(slang::CompilerOptionName::EmitSpirvDirectly, 1),
                                  make_integer_option(slang::CompilerOptionName::VulkanUseEntryPointName, 1)};

        auto target_description = slang::TargetDesc{
                .structureSize = sizeof(slang::TargetDesc),
                .format = SLANG_SPIRV,
                .profile = lease->get()->findProfile("spirv_1_6"),
        };

        auto session_description = slang::SessionDesc{
                .structureSize = sizeof(slang::SessionDesc),
                .targets = &target_description,
                .targetCount = 1,
                .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
                .searchPaths = search_paths.data(),
                .searchPathCount = static_cast<SlangInt>(search_paths.size()),
                .compilerOptionEntries = options.data(),
                .compilerOptionEntryCount = static_cast<std::uint32_t>(options.size()),
        };

        auto session = Slang::ComPtr<slang::ISession>{};

        if (SLANG_FAILED(lease->get()->createSession(session_description, session.writeRef())) || session == nullptr) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_session_failed, SLANG_FAIL,
                                              "IGlobalSession::createSession() failed.")};
        }

        static std::atomic<std::uint64_t> discovery_counter{0};

        auto const module_name = std::format("discover_{}", discovery_counter.fetch_add(1, std::memory_order_relaxed));
        auto const absolute = source_path.absolute().string();
        auto diagnostics = Slang::ComPtr<slang::IBlob>{};

        auto module = Slang::ComPtr<slang::IModule>{session->loadModuleFromSourceString(
                module_name.c_str(), absolute.c_str(), source->c_str(), diagnostics.writeRef())};

        if (module == nullptr) {
            auto text = std::string{};
            append_diagnostics(text, diagnostics);
            return std::unexpected{make_error(ShaderCompileErrorType::module_load_failed, SLANG_FAIL, std::move(text))};
        }

        auto const get_entry_point = std::bit_cast<decltype(&spReflection_getEntryPointByIndex)>(
                impl_->library.symbol("spReflection_getEntryPointByIndex"));
        auto const get_name = std::bit_cast<decltype(&spReflectionEntryPoint_getName)>(
                impl_->library.symbol("spReflectionEntryPoint_getName"));
        auto const get_stage = std::bit_cast<decltype(&spReflectionEntryPoint_getStage)>(
                impl_->library.symbol("spReflectionEntryPoint_getStage"));

        if (get_entry_point == nullptr || get_name == nullptr || get_stage == nullptr) {
            return std::unexpected{make_error(ShaderCompileErrorType::slang_global_session_failed,
                                              SLANG_E_NOT_AVAILABLE,
                                              "The Slang library does not export the reflection functions.")};
        }

        auto found = std::vector<DiscoveredEntryPoint>{};

        for (SlangInt32 index = 0; index < module->getDefinedEntryPointCount(); ++index) {
            auto entry_point = Slang::ComPtr<slang::IEntryPoint>{};

            if (SLANG_FAILED(module->getDefinedEntryPoint(index, entry_point.writeRef())) || entry_point == nullptr) {
                continue;
            }

            auto layout_diagnostics = Slang::ComPtr<slang::IBlob>{};
            auto *layout = reinterpret_cast<SlangReflection *>(entry_point->getLayout(0, layout_diagnostics.writeRef()));
            auto *entry_layout = layout != nullptr ? get_entry_point(layout, 0) : nullptr;

            if (entry_layout == nullptr) {
                auto text = std::string{};
                append_diagnostics(text, layout_diagnostics);
                return std::unexpected{make_error(ShaderCompileErrorType::entry_point_not_found, SLANG_FAIL,
                                                  "Could not reflect an entry point. " + text)};
            }

            auto const *name = get_name(entry_layout);
            auto const stage = from_slang_stage(get_stage(entry_layout));

            if (!stage) {
                warn("Shader bake: skipping entry point '{}' in '{}' (unsupported stage)", name,
                     source_path.logical());
                continue;
            }

            found.push_back(DiscoveredEntryPoint{.name = FlyString{name}, .stage = *stage});
        }

        return found;
    }

    auto SlangCompiler::compile_group(std::span<ShaderCompileRequest const *const> requests) const
            -> std::vector<std::expected<CompiledShader, ShaderCompileError>> {
        auto const group_start = perf_clock::now();
        auto results = std::vector<Impl::Result>{};
        results.reserve(requests.size());

        auto const fail_all = [&](ShaderCompileError const &failure) {
            results.clear();
            for (std::size_t i = 0; i < requests.size(); ++i) {
                results.emplace_back(std::unexpected{failure});
            }
            return results;
        };

        if (!valid()) {
            return fail_all(make_error(
                    ShaderCompileErrorType::slang_global_session_failed, SLANG_E_NOT_AVAILABLE,
                    std::format("Shader '{}' [{}] is not in the shader pack and Slang is not available to compile "
                                "it; re-bake the pack with --bake-shaders.",
                                requests.front()->source_path.logical(), requests.front()->entry_point)));
        }

        perf_events::record(PerfEvent::shader_compile, requests.size());

        auto const &request = *requests.front();

        for (auto const *entry: requests) {
            auto validation = validate_request(*entry);

            if (!validation) {
                return fail_all(validation.error());
            }
        }

        auto source_result = read_source_file(request.source_path.absolute());

        if (!source_result) {
            return fail_all(source_result.error());
        }

        auto source = std::move(*source_result);

        auto search_path_storage = std::vector<std::string>{};

        search_path_storage.reserve(request.include_directories.size() + 1);

        auto const parent_path = request.source_path.absolute().parent_path();

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

        auto macros = std::vector<slang::PreprocessorMacroDesc>{};

        macros.reserve(request.defines.size());

        for (auto const &define: request.defines) {
            macros.push_back(slang::PreprocessorMacroDesc{
                    .name = define.name.c_str(),
                    .value = define.value.c_str(),
            });
        }

        auto options = std::vector<slang::CompilerOptionEntry>{};

        options.reserve(8);
        options.push_back(make_integer_option(slang::CompilerOptionName::EmitSpirvDirectly, 1));
        options.push_back(make_integer_option(slang::CompilerOptionName::VulkanUseEntryPointName, 1));
        options.push_back(make_integer_option(slang::CompilerOptionName::Optimization,
                                              request.optimize ? SLANG_OPTIMIZATION_LEVEL_MAXIMAL
                                                               : SLANG_OPTIMIZATION_LEVEL_NONE));
        options.push_back(make_integer_option(slang::CompilerOptionName::DebugInformation,
                                              request.generate_debug_info ? SLANG_DEBUG_INFO_LEVEL_STANDARD
                                                                          : SLANG_DEBUG_INFO_LEVEL_NONE));
        options.push_back(make_integer_option(slang::CompilerOptionName::SkipSPIRVValidation, 0));
        options.push_back(make_integer_option(slang::CompilerOptionName::MatrixLayoutColumn, 1));
        options.push_back(make_integer_option(slang::CompilerOptionName::MatrixLayoutRow, 0));
        options.push_back(make_string_option(slang::CompilerOptionName::DisableWarning, "41012"));

        auto const lease_start = perf_clock::now();
        auto lease = acquire_session();

        if (!lease) {
            return fail_all(lease.error());
        }

        auto const lease_ms = elapsed_ms(lease_start);
        auto phase_start = perf_clock::now();

        auto target_description = slang::TargetDesc{
                .structureSize = sizeof(slang::TargetDesc),
                .format = SLANG_SPIRV,
                .profile = lease->get()->findProfile("spirv_1_6"),
                .flags = 0,
                .floatingPointMode = SLANG_FLOATING_POINT_MODE_DEFAULT,
                .lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_DEFAULT,
                .forceGLSLScalarBufferLayout = true,
                .compilerOptionEntries = nullptr,
                .compilerOptionEntryCount = 0,
        };

        if (target_description.profile == SLANG_PROFILE_UNKNOWN) {
            return fail_all(make_error(ShaderCompileErrorType::slang_session_failed, SLANG_E_NOT_AVAILABLE,
                                       "Slang does not recognize the "
                                       "\"spirv_1_6\" target profile."));
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

        auto const session_result = lease->get()->createSession(session_description, session.writeRef());

        if (SLANG_FAILED(session_result) || session == nullptr) {
            return fail_all(make_error(ShaderCompileErrorType::slang_session_failed, session_result,
                                       "IGlobalSession::createSession() "
                                       "failed."));
        }

        auto const session_ms = elapsed_ms(phase_start);
        phase_start = perf_clock::now();

        static std::atomic<std::uint64_t> module_name_counter{0};

        auto module_name = request.source_path.absolute().stem().string();

        if (module_name.empty()) {
            module_name = "shader";
        }

        module_name += "_" + std::to_string(module_name_counter.fetch_add(1, std::memory_order_relaxed));

        auto source_path = request.source_path.absolute().string();

        auto module_diagnostics = Slang::ComPtr<slang::IBlob>{};

        auto module = Slang::ComPtr<slang::IModule>{session->loadModuleFromSourceString(
                module_name.c_str(), source_path.c_str(), source.c_str(), module_diagnostics.writeRef())};

        auto module_diagnostics_text = std::string{};
        append_diagnostics(module_diagnostics_text, module_diagnostics);

        auto const module_ms = elapsed_ms(phase_start);

        if (module == nullptr) {
            return fail_all(make_error(ShaderCompileErrorType::module_load_failed, SLANG_FAIL,
                                       std::move(module_diagnostics_text)));
        }

        debug("[Perf] Slang module '{}' ({} entry points): lease {:.2f} ms, session {:.2f} ms, load {:.2f} ms",
              request.source_path.logical(), requests.size(), lease_ms, session_ms, module_ms);

        for (auto const *entry_request: requests) {
            results.push_back(compile_entry(*session, *module, *entry_request, module_diagnostics_text));
        }

        debug("[Perf] Slang group '{}': {} entry points in {:.2f} ms", request.source_path.logical(), requests.size(),
              elapsed_ms(group_start));

        return results;
    }

    auto SlangCompiler::compile_entry(slang::ISession &session, slang::IModule &module,
                                      ShaderCompileRequest const &request, std::string diagnostics) const
            -> std::expected<CompiledShader, ShaderCompileError> {
        auto const entry_start = perf_clock::now();

        auto entry_point = Slang::ComPtr<slang::IEntryPoint>{};

        auto entry_point_diagnostics = Slang::ComPtr<slang::IBlob>{};

        SlangResult result = module.findAndCheckEntryPoint(request.entry_point.c_str(), to_slang_stage(request.stage),
                                                           entry_point.writeRef(), entry_point_diagnostics.writeRef());

        append_diagnostics(diagnostics, entry_point_diagnostics);

        if (SLANG_FAILED(result) || entry_point == nullptr) {
            if (diagnostics.empty()) {
                diagnostics = "Entry point \"" + std::string{request.entry_point.view()} +
                              "\" was not found or does not "
                              "match the requested shader stage.";
            }

            return std::unexpected{
                    make_error(ShaderCompileErrorType::entry_point_not_found, result, std::move(diagnostics))};
        }

        auto components = std::array<slang::IComponentType *, 2>{
                &module,
                entry_point.get(),
        };

        auto composed_program = Slang::ComPtr<slang::IComponentType>{};

        auto composition_diagnostics = Slang::ComPtr<slang::IBlob>{};

        result = session.createCompositeComponentType(components.data(), static_cast<SlangInt>(components.size()),
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

        auto const link_ms = elapsed_ms(entry_start);
        auto phase_start = perf_clock::now();

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

        auto const codegen_ms = elapsed_ms(phase_start);
        phase_start = perf_clock::now();

        if (request.optimize) {
            auto opt_result = spirv_opt::run(std::move(spirv), false);

            if (!opt_result) {
                return std::unexpected{std::move(opt_result.error())};
            }

            spirv = std::move(*opt_result);
        }

        debug("[Perf] Slang compile '{}' [{}]: link {:.2f} ms, codegen {:.2f} ms, spirv-opt {:.2f} ms",
              request.source_path.logical(), request.entry_point, link_ms, codegen_ms, elapsed_ms(phase_start));

        if (!diagnostics.empty()) {
            warn("Slang diagnostics for '{}' [{}]:\n{}", request.source_path.logical(), request.entry_point,
                 diagnostics);
        }

        return CompiledShader{
                .stage = request.stage,
                .entry_point = request.entry_point,
                .spirv = std::move(spirv),
        };
    }

    auto SlangCompiler::valid() const noexcept -> bool { return impl_ != nullptr && impl_->library.valid(); }

    auto SlangCompiler::destroy() noexcept -> void {
        if (impl_ == nullptr) {
            return;
        }

        // Prefetch workers borrow this compiler, so let them finish before the sessions and library go away.
        auto workers = std::vector<std::thread>{};

        {
            std::scoped_lock const lock{impl_->prefetch_mutex};
            workers = std::move(impl_->prefetch_workers);
        }

        for (auto &worker: workers) {
            worker.join();
        }

        {
            std::scoped_lock const lock{impl_->prefetch_mutex};
            impl_->prefetched.clear();
        }

        {
            std::scoped_lock const lock{impl_->session_mutex};
            impl_->idle_sessions.clear();
        }

        impl_->library.destroy();

        impl_.reset();
    }
}
