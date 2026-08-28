#include <BS_thread_pool.hpp>
#include <spirv_reflect.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    struct Job {
        std::filesystem::path spv_path;
        std::string struct_name;
    };

    struct Options {
        std::filesystem::path output_path;
        std::filesystem::path preamble_path;
        std::size_t thread_count = 0;
        std::vector<Job> jobs;
    };

    struct Field {
        std::string name;
        std::string_view cpp_type;
        std::uint32_t offset = 0;
        std::uint32_t size = 0;
    };

    struct ReflectedStruct {
        Job job;
        std::uint32_t size = 0;
        std::vector<Field> fields;
    };

    using Error = std::string;

    template<typename T>
    using Result = std::expected<T, Error>;

    class ReflectModule {
    public:
        ReflectModule() = default;

        ReflectModule(ReflectModule const &) = delete;
        auto operator=(ReflectModule const &) -> ReflectModule & = delete;

        ReflectModule(ReflectModule &&) noexcept = default;
        auto operator=(ReflectModule &&) noexcept -> ReflectModule & = default;

        ~ReflectModule() {
            if (module_) {
                spvReflectDestroyShaderModule(module_.get());
            }
        }

        [[nodiscard]]
        static auto create(std::span<std::byte const> spirv) -> Result<ReflectModule> {
            auto module = std::make_unique<SpvReflectShaderModule>();

            auto const result = spvReflectCreateShaderModule(spirv.size_bytes(), spirv.data(), module.get());

            if (result != SPV_REFLECT_RESULT_SUCCESS) {
                return std::unexpected("spvReflectCreateShaderModule failed with error " +
                                       std::to_string(static_cast<int>(result)));
            }

            return ReflectModule{std::move(module)};
        }

        [[nodiscard]]
        auto get() noexcept -> SpvReflectShaderModule * {
            return module_.get();
        }

    private:
        explicit ReflectModule(std::unique_ptr<SpvReflectShaderModule> module) : module_{std::move(module)} {}

        std::unique_ptr<SpvReflectShaderModule> module_;
    };

    [[nodiscard]]
    auto to_snake_case(std::string_view name) -> std::string {
        auto result = std::string{};
        result.reserve(name.size() + 4);

        for (std::size_t i = 0; i < name.size(); ++i) {
            auto const current = static_cast<unsigned char>(name[i]);

            auto const is_upper = std::isupper(current) != 0;

            auto const previous_is_lower_or_digit =
                    i > 0 && (std::islower(static_cast<unsigned char>(name[i - 1])) != 0 ||
                              std::isdigit(static_cast<unsigned char>(name[i - 1])) != 0);

            auto const next_is_lower =
                    i + 1 < name.size() && std::islower(static_cast<unsigned char>(name[i + 1])) != 0;

            if (is_upper && i != 0 && name[i - 1] != '_' && (previous_is_lower_or_digit || next_is_lower)) {
                result.push_back('_');
            }

            result.push_back(static_cast<char>(std::tolower(current)));
        }

        return result;
    }

    [[nodiscard]]
    auto is_cpp_keyword(std::string_view value) noexcept -> bool {
        static constexpr std::string_view keywords[] = {
                "alignas",     "alignof",   "and",        "and_eq",    "asm",      "auto",         "bitand",
                "bitor",       "bool",      "break",      "case",      "catch",    "char",         "char8_t",
                "char16_t",    "char32_t",  "class",      "compl",     "concept",  "const",        "consteval",
                "constexpr",   "constinit", "const_cast", "continue",  "co_await", "co_return",    "co_yield",
                "decltype",    "default",   "delete",     "do",        "double",   "dynamic_cast", "else",
                "enum",        "explicit",  "export",     "extern",    "false",    "float",        "for",
                "friend",      "goto",      "if",         "inline",    "int",      "long",         "mutable",
                "namespace",   "new",       "noexcept",   "not",       "not_eq",   "nullptr",      "operator",
                "or",          "or_eq",     "private",    "protected", "public",   "register",     "reinterpret_cast",
                "requires",    "return",    "short",      "signed",    "sizeof",   "static",       "static_assert",
                "static_cast", "struct",    "switch",     "template",  "this",     "thread_local", "throw",
                "true",        "try",       "typedef",    "typeid",    "typename", "union",        "unsigned",
                "using",       "virtual",   "void",       "volatile",  "wchar_t",  "while",        "xor",
                "xor_eq",
        };

        return std::find(std::begin(keywords), std::end(keywords), value) != std::end(keywords);
    }

    [[nodiscard]]
    auto make_identifier(char const *name) -> Result<std::string> {
        if (name == nullptr || *name == '\0') {
            return std::unexpected("encountered unnamed push-constant member");
        }

        auto result = to_snake_case(name);

        for (auto &character: result) {
            auto const value = static_cast<unsigned char>(character);

            if (std::isalnum(value) == 0 && character != '_') {
                character = '_';
            }
        }

        if (!result.empty() && std::isdigit(static_cast<unsigned char>(result.front())) != 0) {
            result.insert(result.begin(), '_');
        }

        if (is_cpp_keyword(result)) {
            result.push_back('_');
        }

        return result;
    }

    [[nodiscard]]
    auto scalar_cpp_type(SpvReflectTypeFlags flags, SpvReflectNumericTraits const &numeric)
            -> Result<std::string_view> {

        auto const width = numeric.scalar.width;

        if ((flags & SPV_REFLECT_TYPE_FLAG_FLOAT) != 0) {
            switch (width) {
                case 32:
                    return "float";

                case 64:
                    return "double";

                default:
                    return std::unexpected("unsupported floating-point width: " + std::to_string(width));
            }
        }

        if ((flags & SPV_REFLECT_TYPE_FLAG_INT) != 0) {
            auto const is_signed = numeric.scalar.signedness != 0;

            switch (width) {
                case 8:
                    return is_signed ? std::string_view{"std::int8_t"} : std::string_view{"std::uint8_t"};

                case 16:
                    return is_signed ? std::string_view{"std::int16_t"} : std::string_view{"std::uint16_t"};

                case 32:
                    return is_signed ? std::string_view{"std::int32_t"} : std::string_view{"std::uint32_t"};

                case 64:
                    return is_signed ? std::string_view{"std::int64_t"} : std::string_view{"std::uint64_t"};

                default:
                    return std::unexpected("unsupported integer width: " + std::to_string(width));
            }
        }

        return std::unexpected("push-constant member is not a supported integer or floating-point scalar");
    }

    auto flatten(SpvReflectBlockVariable const &member, std::vector<Field> &fields) -> Result<void> {

        auto const *type = member.type_description;

        if (type == nullptr) {
            return std::unexpected("push-constant member has no type description");
        }

        auto const flags = type->type_flags;

        auto const is_physical_storage_pointer =
                (flags & SPV_REFLECT_TYPE_FLAG_REF) != 0 && type->storage_class == SpvStorageClassPhysicalStorageBuffer;

        if (is_physical_storage_pointer) {
            auto name = make_identifier(member.name);

            if (!name) {
                return std::unexpected(std::move(name.error()));
            }

            name->append("_address");

            fields.push_back(Field{
                    .name = std::move(*name),
                    .cpp_type = "VkDeviceAddress",
                    .offset = member.absolute_offset,
                    .size = member.size,
            });

            return {};
        }

        if ((flags & SPV_REFLECT_TYPE_FLAG_STRUCT) != 0) {
            for (std::uint32_t i = 0; i < member.member_count; ++i) {
                auto result = flatten(member.members[i], fields);

                if (!result) {
                    return result;
                }
            }

            return {};
        }

        //
        // These require C++ representations that understand stride/alignment.
        // Silently treating one as a scalar, as the old generator did, can produce
        // a struct that compiles but doesn't actually describe the shader layout.
        //
        if ((flags & SPV_REFLECT_TYPE_FLAG_ARRAY) != 0) {
            return std::unexpected(std::string{"arrays are not currently supported for push-constant member "} +
                                   (member.name ? member.name : "<unnamed>"));
        }

        if ((flags & SPV_REFLECT_TYPE_FLAG_MATRIX) != 0) {
            return std::unexpected(std::string{"matrices are not currently supported for push-constant member "} +
                                   (member.name ? member.name : "<unnamed>"));
        }

        if ((flags & SPV_REFLECT_TYPE_FLAG_BOOL) != 0) {
            return std::unexpected(std::string{"bool is not supported for push-constant member "} +
                                   (member.name ? member.name : "<unnamed>") +
                                   "; use an explicitly sized integer instead");
        }

        auto name = make_identifier(member.name);

        if (!name) {
            return std::unexpected(std::move(name.error()));
        }

        auto component_type = scalar_cpp_type(flags, member.numeric);

        if (!component_type) {
            return std::unexpected(*name + ": " + component_type.error());
        }

        auto const component_size = member.numeric.scalar.width / 8;

        auto const component_count =
                (flags & SPV_REFLECT_TYPE_FLAG_VECTOR) != 0 ? member.numeric.vector.component_count : 1u;

        if (component_count == 0 || component_count > 4) {
            return std::unexpected(*name + ": unsupported vector component count " + std::to_string(component_count));
        }

        if (component_count == 1) {
            fields.push_back(Field{
                    .name = std::move(*name),
                    .cpp_type = *component_type,
                    .offset = member.absolute_offset,
                    .size = component_size,
            });

            return {};
        }

        static constexpr std::string_view suffixes[] = {
                "x",
                "y",
                "z",
                "w",
        };

        for (std::uint32_t i = 0; i < component_count; ++i) {
            fields.push_back(Field{
                    .name = *name + "_" + std::string{suffixes[i]},
                    .cpp_type = *component_type,
                    .offset = member.absolute_offset + i * component_size,
                    .size = component_size,
            });
        }

        return {};
    }

    [[nodiscard]]
    auto read_binary_file(std::filesystem::path const &path) -> Result<std::vector<std::byte>> {

        auto file = std::ifstream{
                path,
                std::ios::binary | std::ios::ate,
        };

        if (!file) {
            return std::unexpected("cannot open " + path.string());
        }

        auto const end = file.tellg();

        if (end < 0) {
            return std::unexpected("cannot determine size of " + path.string());
        }

        auto data = std::vector<std::byte>(static_cast<std::size_t>(end));

        file.seekg(0, std::ios::beg);

        if (!data.empty()) {
            file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));

            if (!file) {
                return std::unexpected("failed while reading " + path.string());
            }
        }

        return data;
    }

    [[nodiscard]]
    auto read_text_file(std::filesystem::path const &path) -> Result<std::string> {

        auto file = std::ifstream{
                path,
                std::ios::binary | std::ios::ate,
        };

        if (!file) {
            return std::unexpected("cannot open " + path.string());
        }

        auto const end = file.tellg();

        if (end < 0) {
            return std::unexpected("cannot determine size of " + path.string());
        }

        auto result = std::string(static_cast<std::size_t>(end), '\0');

        file.seekg(0, std::ios::beg);

        if (!result.empty()) {
            file.read(result.data(), static_cast<std::streamsize>(result.size()));

            if (!file) {
                return std::unexpected("failed while reading " + path.string());
            }
        }

        return result;
    }

    [[nodiscard]]
    auto reflect_job(Job const &job) -> Result<ReflectedStruct> {
        auto spirv = read_binary_file(job.spv_path);

        if (!spirv) {
            return std::unexpected(job.spv_path.string() + ": " + spirv.error());
        }

        auto module = ReflectModule::create(*spirv);

        if (!module) {
            return std::unexpected(job.spv_path.string() + ": " + module.error());
        }

        std::uint32_t block_count = 0;

        auto result = spvReflectEnumeratePushConstantBlocks(module->get(), &block_count, nullptr);

        if (result != SPV_REFLECT_RESULT_SUCCESS) {
            return std::unexpected(job.spv_path.string() + ": failed to enumerate push-constant blocks");
        }

        if (block_count != 1) {
            return std::unexpected(job.spv_path.string() + ": expected exactly one push-constant block, found " +
                                   std::to_string(block_count));
        }

        SpvReflectBlockVariable *block = nullptr;

        result = spvReflectEnumeratePushConstantBlocks(module->get(), &block_count, &block);

        if (result != SPV_REFLECT_RESULT_SUCCESS || block == nullptr) {
            return std::unexpected(job.spv_path.string() + ": failed to obtain push-constant block");
        }

        auto fields = std::vector<Field>{};
        fields.reserve(block->member_count * 2);

        for (std::uint32_t i = 0; i < block->member_count; ++i) {
            auto flatten_result = flatten(block->members[i], fields);

            if (!flatten_result) {
                return std::unexpected(job.spv_path.string() + ": " + flatten_result.error());
            }
        }

        std::ranges::stable_sort(fields, {}, &Field::offset);

        auto names = std::unordered_set<std::string>{};
        names.reserve(fields.size());

        for (auto const &field: fields) {
            if (!names.insert(field.name).second) {
                return std::unexpected(job.spv_path.string() + ": generated duplicate C++ field name '" + field.name +
                                       "'");
            }
        }

        std::uint32_t cursor = 0;

        for (auto const &field: fields) {
            if (field.offset < cursor) {
                return std::unexpected(job.spv_path.string() + ": overlapping reflected field '" + field.name + "'");
            }

            cursor = field.offset + field.size;
        }

        if (cursor > block->size) {
            return std::unexpected(job.spv_path.string() + ": reflected members extend beyond the push-constant block");
        }

        return ReflectedStruct{
                .job = job,
                .size = block->size,
                .fields = std::move(fields),
        };
    }

    auto append_struct(std::string &output, ReflectedStruct const &reflected) -> void {

        output += "// Generated from ";
        output += reflected.job.spv_path.filename().string();
        output += "\n";

        output += "struct ";
        output += reflected.job.struct_name;
        output += " {\n";

        std::uint32_t cursor = 0;
        std::uint32_t padding_index = 0;

        for (auto const &field: reflected.fields) {
            if (field.offset > cursor) {
                auto const padding_size = field.offset - cursor;

                output += "    std::uint8_t generated_padding_";
                output += std::to_string(padding_index++);
                output += "[";
                output += std::to_string(padding_size);
                output += "]{};\n";
            }

            output += "    ";
            output += field.cpp_type;
            output += " ";
            output += field.name;
            output += "{};\n";

            cursor = field.offset + field.size;
        }

        if (reflected.size > cursor) {
            output += "    std::uint8_t generated_padding_";
            output += std::to_string(padding_index);
            output += "[";
            output += std::to_string(reflected.size - cursor);
            output += "]{};\n";
        }

        output += "};\n\n";

        output += "static_assert(std::is_standard_layout_v<";
        output += reflected.job.struct_name;
        output += ">);\n";

        output += "static_assert(sizeof(";
        output += reflected.job.struct_name;
        output += ") == ";
        output += std::to_string(reflected.size);
        output += ");\n";

        for (auto const &field: reflected.fields) {
            output += "static_assert(offsetof(";
            output += reflected.job.struct_name;
            output += ", ";
            output += field.name;
            output += ") == ";
            output += std::to_string(field.offset);
            output += ");\n";
        }

        output += "\n";
    }

    [[nodiscard]]
    auto write_if_changed(std::filesystem::path const &path, std::string_view contents) -> Result<bool> {

        auto error = std::error_code{};

        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path(), error);

            if (error) {
                return std::unexpected("cannot create output directory " + path.parent_path().string() + ": " +
                                       error.message());
            }
        }

        if (std::filesystem::exists(path, error) && !error) {
            auto existing = read_text_file(path);

            if (!existing) {
                return std::unexpected(existing.error());
            }

            if (*existing == contents) {
                return false;
            }
        }

        auto file = std::ofstream{
                path,
                std::ios::binary | std::ios::trunc,
        };

        if (!file) {
            return std::unexpected("cannot write " + path.string());
        }

        file.write(contents.data(), static_cast<std::streamsize>(contents.size()));

        if (!file) {
            return std::unexpected("failed while writing " + path.string());
        }

        return true;
    }

    auto print_usage(char const *program) -> void {
        std::cerr << "usage:\n"
                  << "  " << program << " --output <header.hxx>"
                  << " --preamble <preamble.hxx>"
                  << " [--threads <count>]"
                  << " --shader <StructName> <input.spv>"
                  << " [--shader <StructName> <input.spv> ...]\n";
    }

    [[nodiscard]]
    auto parse_size(std::string_view value) -> Result<std::size_t> {
        std::size_t result = 0;

        auto const [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);

        if (error != std::errc{} || end != value.data() + value.size() || result == 0) {
            return std::unexpected("invalid positive integer: " + std::string{value});
        }

        return result;
    }

    [[nodiscard]]
    auto parse_options(int argc, char **argv) -> Result<Options> {

        auto options = Options{};

        for (int i = 1; i < argc;) {
            auto const argument = std::string_view{argv[i]};

            if (argument == "--output") {
                if (i + 1 >= argc) {
                    return std::unexpected("--output requires a path");
                }

                options.output_path = argv[i + 1];
                i += 2;
                continue;
            }

            if (argument == "--preamble") {
                if (i + 1 >= argc) {
                    return std::unexpected("--preamble requires a path");
                }

                options.preamble_path = argv[i + 1];
                i += 2;
                continue;
            }

            if (argument == "--threads") {
                if (i + 1 >= argc) {
                    return std::unexpected("--threads requires a count");
                }

                auto thread_count = parse_size(argv[i + 1]);

                if (!thread_count) {
                    return std::unexpected(thread_count.error());
                }

                options.thread_count = *thread_count;
                i += 2;
                continue;
            }

            if (argument == "--shader") {
                if (i + 2 >= argc) {
                    return std::unexpected("--shader requires <StructName> <input.spv>");
                }

                options.jobs.push_back(Job{
                        .spv_path = argv[i + 2],
                        .struct_name = argv[i + 1],
                });

                i += 3;
                continue;
            }

            if (argument == "--help" || argument == "-h") {
                return std::unexpected("");
            }

            return std::unexpected("unknown argument: " + std::string{argument});
        }

        if (options.output_path.empty()) {
            return std::unexpected("--output is required");
        }

        if (options.preamble_path.empty()) {
            return std::unexpected("--preamble is required");
        }

        if (options.jobs.empty()) {
            return std::unexpected("at least one --shader is required");
        }

        auto struct_names = std::unordered_set<std::string>{};

        for (auto const &job: options.jobs) {
            if (!struct_names.insert(job.struct_name).second) {
                return std::unexpected("duplicate generated struct name: " + job.struct_name);
            }
        }

        if (options.thread_count == 0) {
            auto const hardware_threads = std::max(1u, std::thread::hardware_concurrency());

            options.thread_count = std::min<std::size_t>(options.jobs.size(), hardware_threads);
        }

        return options;
    }

} // namespace

auto main(int argc, char **argv) -> int {
    auto options = parse_options(argc, argv);

    if (!options) {
        print_usage(argv[0]);

        if (!options.error().empty()) {
            std::cerr << "reflect_push_constants: " << options.error() << '\n';
        }

        return options.error().empty() ? 0 : 1;
    }

    auto preamble = read_text_file(options->preamble_path);

    if (!preamble) {
        std::cerr << "reflect_push_constants: " << preamble.error() << '\n';

        return 1;
    }

    auto pool = BS::thread_pool{
            options->thread_count,
    };

    auto futures = std::vector<std::future<Result<ReflectedStruct>>>{};

    futures.reserve(options->jobs.size());

    for (auto const &job: options->jobs) {
        futures.push_back(pool.submit_task([job] { return reflect_job(job); }));
    }

    auto reflected = std::vector<ReflectedStruct>{};

    reflected.reserve(options->jobs.size());

    //
    // Consume futures in declaration order, not completion order.
    // This keeps the generated header deterministic.
    //
    for (auto &future: futures) {
        auto result = future.get();

        if (!result) {
            std::cerr << "reflect_push_constants: " << result.error() << '\n';

            return 1;
        }

        reflected.push_back(std::move(*result));
    }

    auto output = std::string{};
    output.reserve(preamble->size() + reflected.size() * 1024);

    output += *preamble;

    if (!output.empty() && output.back() != '\n') {
        output.push_back('\n');
    }

    output += "\n"
              "// -----------------------------------------------------------------------------\n"
              "// Generated shader push constants. Do not edit.\n"
              "// -----------------------------------------------------------------------------\n"
              "\n";

    for (auto const &entry: reflected) {
        append_struct(output, entry);
    }

    auto written = write_if_changed(options->output_path, output);

    if (!written) {
        std::cerr << "reflect_push_constants: " << written.error() << '\n';

        return 1;
    }

    if (*written) {
        std::cout << "Generated " << options->output_path.string() << " from " << options->jobs.size() << " shader"
                  << (options->jobs.size() == 1 ? "" : "s") << '\n';
    }

    return 0;
}
