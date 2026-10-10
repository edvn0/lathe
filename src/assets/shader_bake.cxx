#include "assets/shader_bake.hxx"

#include <algorithm>
#include <fstream>
#include <future>
#include <iterator>
#include <sstream>
#include <unordered_map>

#include "assets/shader_pack.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "core/thread_pool.hxx"

namespace renderer {

    namespace {
        struct FileEntryPoints {
            std::string logical;
            std::string relative;
            std::expected<std::vector<DiscoveredEntryPoint>, ShaderCompileError> entry_points;
        };

        [[nodiscard]] auto describe(ShaderCompileError const &error) -> std::string {
            return std::format("{} {}", error.type, error.diagnostics);
        }
    }

    auto parse_shader_variants(std::string_view text) -> std::expected<std::vector<ShaderVariant>, std::string> {
        auto variants = std::vector<ShaderVariant>{};
        auto stream = std::istringstream{std::string{text}};
        auto line = std::string{};
        auto number = std::size_t{0};

        while (std::getline(stream, line)) {
            ++number;

            if (auto const hash = line.find('#'); hash != std::string::npos) {
                line.resize(hash);
            }

            auto words = std::istringstream{line};
            auto variant = ShaderVariant{};

            if (!(words >> variant.file)) {
                continue;
            }

            if (!(words >> variant.entry)) {
                return std::unexpected{std::format("variants line {}: expected `<file> <entry> [NAME[=VALUE]]...`", number)};
            }

            auto define = std::string{};

            while (words >> define) {
                auto const equals = define.find('=');
                auto parsed = equals == std::string::npos
                                      ? ShaderDefine{.name = define, .value = "1"}
                                      : ShaderDefine{.name = define.substr(0, equals), .value = define.substr(equals + 1)};

                if (parsed.name.empty()) {
                    return std::unexpected{std::format("variants line {}: empty define name", number)};
                }

                variant.defines.push_back(std::move(parsed));
            }

            variants.push_back(std::move(variant));
        }

        return variants;
    }

    auto bake_shaders(SlangCompiler const &compiler, std::filesystem::path const &output)
            -> std::expected<ShaderBakeResult, std::string> {
        if (!compiler.valid()) {
            return std::unexpected{"Slang is not available, so shaders cannot be baked"};
        }

        auto const root = Paths::current().data_root();
        auto const shader_directory = root / "assets" / "shaders";

        std::error_code error;

        if (!std::filesystem::is_directory(shader_directory, error)) {
            return std::unexpected{std::format("'{}' is not a directory", shader_directory.string())};
        }

        auto files = std::vector<std::filesystem::path>{};

        for (auto it = std::filesystem::recursive_directory_iterator{shader_directory, error};
             !error && it != std::filesystem::recursive_directory_iterator{}; it.increment(error)) {
            if (it->is_regular_file(error) && it->path().extension() == ".slang") {
                files.push_back(it->path());
            }
        }

        std::ranges::sort(files);

        auto variants = std::vector<ShaderVariant>{};

        if (auto const variants_path = shader_directory / "variants.txt"; std::filesystem::exists(variants_path)) {
            std::ifstream in{variants_path};
            auto const text = std::string{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
            auto parsed = parse_shader_variants(text);

            if (!parsed) {
                return std::unexpected{std::format("{}: {}", variants_path.string(), parsed.error())};
            }

            variants = std::move(*parsed);
        }

        auto discoveries = std::vector<std::future<FileEntryPoints>>{};

        for (auto const &file: files) {
            auto relative = file.lexically_relative(shader_directory).generic_string();
            auto logical = (std::filesystem::path{"assets"} / "shaders" / relative).generic_string();

            discoveries.push_back(thread_pool().submit_task(
                    [&compiler, &shader_directory, logical = std::move(logical), relative = std::move(relative)]() mutable {
                        auto path = Paths::current().data(logical);

                        if (!path) {
                            return FileEntryPoints{
                                    .logical = logical,
                                    .relative = std::move(relative),
                                    .entry_points = std::unexpected{ShaderCompileError{
                                            .type = ShaderCompileErrorType::source_not_found,
                                            .diagnostics = "invalid shader path"}},
                            };
                        }

                        return FileEntryPoints{
                                .logical = logical,
                                .relative = std::move(relative),
                                .entry_points = compiler.discover_entry_points(*path, std::span{&shader_directory, 1}),
                        };
                    }));
        }

        auto requests = std::vector<ShaderCompileRequest>{};
        auto failures = std::vector<std::string>{};
        auto stage_of = std::unordered_map<std::string, ShaderStage>{};

        for (auto &future: discoveries) {
            auto const file = future.get();

            if (!file.entry_points) {
                failures.push_back(std::format("{}: {}", file.logical, describe(file.entry_points.error())));
                continue;
            }

            for (auto const &entry: *file.entry_points) {
                stage_of.emplace(std::format("{}|{}", file.relative, entry.name), entry.stage);

                requests.push_back(ShaderCompileRequest{
                        .source_path = *Paths::current().data(file.logical),
                        .entry_point = entry.name,
                        .stage = entry.stage,
                        .include_directories = {shader_directory},
                });
            }
        }

        auto const entry_point_count = requests.size();

        for (auto const &variant: variants) {
            auto const stage = stage_of.find(std::format("{}|{}", variant.file, variant.entry));

            if (stage == stage_of.end()) {
                failures.push_back(std::format("variants.txt: no entry point '{}' in '{}'", variant.entry, variant.file));
                continue;
            }

            requests.push_back(ShaderCompileRequest{
                    .source_path = *Paths::current().data(std::format("assets/shaders/{}", variant.file)),
                    .entry_point = FlyString{variant.entry},
                    .stage = stage->second,
                    .include_directories = {shader_directory},
                    .defines = variant.defines,
            });
        }

        start_shader_recording();
        compiler.prefetch(requests);

        for (auto const &request: requests) {
            if (auto compiled = compiler.compile(request); !compiled) {
                failures.push_back(std::format("{} [{}]: {}", request.source_path.logical(), request.entry_point,
                                               describe(compiled.error())));
            }
        }

        if (!failures.empty()) {
            std::ignore = finish_shader_recording(output);

            auto message = std::string{"shader bake failed:"};

            for (auto const &failure: failures) {
                message += "\n  " + failure;
            }

            std::error_code ignored;
            std::filesystem::remove(output, ignored);

            return std::unexpected{std::move(message)};
        }

        auto saved = finish_shader_recording(output);

        if (!saved) {
            return std::unexpected{std::move(saved.error())};
        }

        return ShaderBakeResult{.entry_points = entry_point_count, .variants = requests.size() - entry_point_count};
    }

}
