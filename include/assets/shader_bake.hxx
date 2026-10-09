#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "assets/slang_compiler.hxx"

namespace renderer {

    // An extra compile of one entry point with defines, which entry point discovery cannot find on its own.
    struct ShaderVariant {
        std::string file; // relative to assets/shaders
        std::string entry;
        std::vector<ShaderDefine> defines;
    };

    // Parses assets/shaders/variants.txt: one variant per line as `<file> <entry> [NAME[=VALUE]]...`, with `#`
    // comments. A define without a value gets the value 1.
    [[nodiscard]] auto parse_shader_variants(std::string_view text) -> std::expected<std::vector<ShaderVariant>, std::string>;

    struct ShaderBakeResult {
        std::size_t entry_points = 0;
        std::size_t variants = 0;
    };

    // Compiles every entry point of every .slang file under <data root>/assets/shaders, plus the variants listed in
    // variants.txt there, into a shader pack at `output`. Needs no window, GPU or frames, so a game's own shaders are
    // covered by dropping them next to the engine's.
    [[nodiscard]] auto bake_shaders(SlangCompiler const &compiler, std::filesystem::path const &output)
            -> std::expected<ShaderBakeResult, std::string>;

}
