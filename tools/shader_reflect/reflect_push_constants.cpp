// Build-time tool: reads a compiled SPIR-V module, finds its single push
// constant block via SPIRV-Reflect, flattens it (descending through
// non-pointer struct members, stopping at PhysicalStorageBuffer pointers and
// at scalars/vectors), and writes a C++ struct definition that reproduces
// the shader's push-constant layout byte-for-byte.
//
// Usage: reflect_push_constants <input.spv> <StructName> <output.inc>
//
// Run once per shader entry point by the CMake shader-reflection step (see
// CMakeLists.txt); the per-shader .inc fragments this emits are concatenated
// into generated/include/shader_push_constants.hxx.

#include <spirv_reflect.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
    struct Field {
        std::string name;
        std::string cpp_type;
        std::uint32_t offset;
        std::uint32_t size;
        bool is_padding;
    };

    std::string to_snake_case(std::string const &name) {
        auto out = std::string{};
        out.reserve(name.size() + 4);

        for (std::size_t i = 0; i < name.size(); ++i) {
            char c = name[i];

            if (
                std::isupper(static_cast<unsigned char>(c))
                && i != 0
                && name[i - 1] != '_'
            ) {
                out += '_';
            }

            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        return out;
    }

    // Scalar (or per-component, for vectors) C++ type for a numeric leaf.
    std::string scalar_cpp_type(SpvReflectTypeFlags flags, SpvReflectNumericTraits const &numeric) {
        auto const width = numeric.scalar.width;

        if (flags & SPV_REFLECT_TYPE_FLAG_FLOAT) {
            if (width == 32) return "float";
            if (width == 64) return "double";
        } else if (flags & SPV_REFLECT_TYPE_FLAG_INT) {
            bool const is_signed = numeric.scalar.signedness != 0;

            if (width == 32) return is_signed ? "std::int32_t" : "std::uint32_t";
            if (width == 16) return is_signed ? "std::int16_t" : "std::uint16_t";
            if (width == 64) return is_signed ? "std::int64_t" : "std::uint64_t";
        }

        std::fprintf(stderr,
                     "reflect_push_constants: unsupported scalar (flags=0x%x, width=%u)\n",
                     flags, width);
        std::exit(1);
    }

    void flatten(SpvReflectBlockVariable const &member, std::vector<Field> &out) {
        auto const *type = member.type_description;
        auto const flags = type ? type->type_flags : 0u;

        bool const is_pointer =
            (flags & SPV_REFLECT_TYPE_FLAG_REF) != 0
            && type->storage_class == SpvStorageClassPhysicalStorageBuffer;

        if (is_pointer) {
            out.push_back(Field{
                    .name = to_snake_case(member.name ? member.name : "") + "_address",
                    .cpp_type = "VkDeviceAddress",
                    .offset = member.absolute_offset,
                    .size = member.size,
                    .is_padding = false,
            });
            return;
        }

        if (flags & SPV_REFLECT_TYPE_FLAG_STRUCT) {
            for (std::uint32_t i = 0; i < member.member_count; ++i) {
                flatten(member.members[i], out);
            }
            return;
        }

        auto const component_count =
            (flags & SPV_REFLECT_TYPE_FLAG_VECTOR) ? member.numeric.vector.component_count : 1;
        auto const component_type = scalar_cpp_type(flags, member.numeric);
        auto const component_size = member.numeric.scalar.width / 8;
        auto const base_name = to_snake_case(member.name ? member.name : "");

        if (component_count == 1) {
            out.push_back(Field{
                    .name = base_name,
                    .cpp_type = component_type,
                    .offset = member.absolute_offset,
                    .size = member.size,
                    .is_padding = false,
            });
            return;
        }

        static constexpr char const *suffixes[] = {"x", "y", "z", "w"};

        for (std::uint32_t i = 0; i < component_count; ++i) {
            out.push_back(Field{
                    .name = base_name + "_" + suffixes[i],
                    .cpp_type = component_type,
                    .offset = member.absolute_offset + i * component_size,
                    .size = component_size,
                    .is_padding = false,
            });
        }
    }

    std::vector<std::uint8_t> read_file(char const *path) {
        auto file = std::ifstream{path, std::ios::binary | std::ios::ate};

        if (!file) {
            std::fprintf(stderr, "reflect_push_constants: cannot open %s\n", path);
            std::exit(1);
        }

        auto const size = static_cast<std::size_t>(file.tellg());
        file.seekg(0);

        auto data = std::vector<std::uint8_t>(size);
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));

        return data;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <input.spv> <StructName> <output.inc>\n", argv[0]);
        return 1;
    }

    char const *spv_path = argv[1];
    char const *struct_name = argv[2];
    char const *output_path = argv[3];

    auto const spirv = read_file(spv_path);

    SpvReflectShaderModule module;
    auto result = spvReflectCreateShaderModule(spirv.size(), spirv.data(), &module);

    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        std::fprintf(stderr, "reflect_push_constants: spvReflectCreateShaderModule failed for %s (%d)\n",
                     spv_path, static_cast<int>(result));
        return 1;
    }

    std::uint32_t block_count = 0;
    spvReflectEnumeratePushConstantBlocks(&module, &block_count, nullptr);

    if (block_count != 1) {
        std::fprintf(stderr,
                     "reflect_push_constants: %s has %u push constant blocks, expected exactly 1\n",
                     spv_path, block_count);
        return 1;
    }

    SpvReflectBlockVariable *block = nullptr;
    spvReflectEnumeratePushConstantBlocks(&module, &block_count, &block);

    auto fields = std::vector<Field>{};
    flatten(*block, fields);

    // Insert explicit padding wherever the reflected layout has a gap, so
    // the generated struct's natural C++ layout reproduces the shader's
    // layout without relying on incidental alignment.
    auto laid_out = std::vector<Field>{};
    std::uint32_t cursor = 0;
    std::uint32_t pad_index = 0;

    for (auto const &field: fields) {
        if (field.offset > cursor) {
            laid_out.push_back(Field{
                    .name = "_pad" + std::to_string(pad_index++),
                    .cpp_type = "std::uint8_t",
                    .offset = cursor,
                    .size = field.offset - cursor,
                    .is_padding = true,
            });
        }

        laid_out.push_back(field);
        cursor = field.offset + field.size;
    }

    if (block->size > cursor) {
        laid_out.push_back(Field{
                .name = "_pad" + std::to_string(pad_index++),
                .cpp_type = "std::uint8_t",
                .offset = cursor,
                .size = block->size - cursor,
                .is_padding = true,
        });
    }

    auto out = std::ostringstream{};

    out << "// Generated from " << spv_path << " -- see tools/shader_reflect/reflect_push_constants.cpp\n";
    out << "struct " << struct_name << " {\n";

    for (auto const &field: laid_out) {
        if (field.is_padding) {
            out << "    " << field.cpp_type << " " << field.name << "[" << field.size << "];\n";
        } else {
            out << "    " << field.cpp_type << " " << field.name << ";\n";
        }
    }

    out << "};\n\n";
    out << "static_assert(sizeof(" << struct_name << ") == " << block->size << ");\n";

    for (auto const &field: laid_out) {
        if (field.is_padding) continue;

        out << "static_assert(offsetof(" << struct_name << ", " << field.name << ") == "
            << field.offset << ");\n";
    }

    out << "\n";

    auto output_file = std::ofstream{output_path, std::ios::binary};

    if (!output_file) {
        std::fprintf(stderr, "reflect_push_constants: cannot write %s\n", output_path);
        return 1;
    }

    output_file << out.str();

    spvReflectDestroyShaderModule(&module);

    return 0;
}
