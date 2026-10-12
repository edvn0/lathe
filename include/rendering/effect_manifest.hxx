#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <volk.h>

// The declaration of a Lua-driven compute effect, read from a small JSON file next to its shader. It says what the
// shader reads and writes and how much work to dispatch; nothing else about the GPU is ever up to a script. Parsing
// and checking need no GPU (see EffectSystem for what turns this into graph passes).
//
//   {
//     "shader": "effects/tint.slang",            relative to assets/shaders, may not leave it
//     "entry": "main_cs",                        optional, default "main_cs"
//     "group_size": [8, 8],                      optional, default [64, 1]; must match [numthreads]
//     "bindings": [                              pushed to the shader in this order, then the params
//       {"name": "source", "image": "in", "source": "scene_colour"},       sampled; scene_colour or scene_depth
//       {"name": "result", "image": "out", "format": "rgba16f", "replaces": "scene_colour"},
//       {"name": "counts", "buffer": "read_write", "elements": 256}         "read", "write" or "read_write"
//     ],
//     "params": [
//       {"name": "strength", "type": "float", "default": 1.0, "min": 0.0, "max": 4.0},
//       {"name": "tint", "type": "float3", "default": [1.0, 0.5, 0.2]}
//     ],
//     "dispatch": {"per_pixel_of": "result"}     or {"threads": 4096} or {"elements_of": "counts"}
//   }
//
// Push constants are the bindings then the params, laid out as std430 lays out the shader's struct: an image is a
// uint (its bindless index), a buffer a 64-bit address, a float2 is aligned to 8 and a float3 or float4 to 16 (a float3
// takes 12 bytes). At most 128 bytes.
// "replaces": "scene_colour" makes composition sample the result instead (after_lighting and before_composite only).
// A buffer with "elements" is created and kept by the engine for each instance when the script does not give one.

inline constexpr std::size_t max_effect_push_bytes = 128;
inline constexpr std::uint32_t max_effect_buffer_elements = 1U << 24U;
inline constexpr std::uint32_t max_effect_threads = 1U << 26U;

enum class EffectParamType : std::uint8_t { float1, float2, float3, float4, int1, uint1 };

[[nodiscard]] constexpr auto component_count(EffectParamType type) noexcept -> std::size_t {
    switch (type) {
        case EffectParamType::float1:
        case EffectParamType::int1:
        case EffectParamType::uint1:
            return 1;
        case EffectParamType::float2:
            return 2;
        case EffectParamType::float3:
            return 3;
        case EffectParamType::float4:
            return 4;
    }
    return 1;
}

// std430 alignment of a param in the push constants.
[[nodiscard]] constexpr auto push_alignment(EffectParamType type) noexcept -> std::size_t {
    switch (type) {
        case EffectParamType::float2:
            return 8;
        case EffectParamType::float3:
        case EffectParamType::float4:
            return 16;
        default:
            return 4;
    }
}

[[nodiscard]] constexpr auto is_integer(EffectParamType type) noexcept -> bool {
    return type == EffectParamType::int1 || type == EffectParamType::uint1;
}

enum class EffectFormat : std::uint8_t { rgba16f, rgba8, r32f };

[[nodiscard]] constexpr auto to_vk_format(EffectFormat format) noexcept -> VkFormat {
    switch (format) {
        case EffectFormat::rgba16f:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case EffectFormat::rgba8:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case EffectFormat::r32f:
            return VK_FORMAT_R32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

// The engine images an effect may sample.
enum class EffectSource : std::uint8_t { scene_colour, scene_depth };

enum class EffectBufferAccess : std::uint8_t { read, write, read_write };

struct EffectParam {
    std::string name;
    EffectParamType type = EffectParamType::float1;
    std::array<double, 4> value{};
    double min = -1.0e9;
    double max = 1.0e9;
};

struct EffectImageIn {
    EffectSource source = EffectSource::scene_colour;
};

struct EffectImageOut {
    EffectFormat format = EffectFormat::rgba16f;
    // The composition image this output stands in for.
    bool replaces_scene_colour = false;
};

struct EffectBufferBinding {
    EffectBufferAccess access = EffectBufferAccess::read_write;
    // Zero: the script must supply a buffer. Otherwise the size of the engine's own buffer and the least it accepts.
    std::uint32_t elements = 0;
};

struct EffectBinding {
    std::string name;
    std::variant<EffectImageIn, EffectImageOut, EffectBufferBinding> what;
};

struct EffectDispatch {
    enum class Kind : std::uint8_t { per_pixel, threads, elements };

    Kind kind = Kind::per_pixel;
    // The image or buffer binding it follows, for per_pixel and elements.
    std::string of;
    std::uint32_t threads = 0;
};

struct EffectManifest {
    std::string shader;
    std::string entry = "main_cs";
    std::array<std::uint32_t, 2> group_size{64, 1};
    std::vector<EffectBinding> bindings;
    std::vector<EffectParam> params;
    EffectDispatch dispatch;

    [[nodiscard]] auto find_binding(std::string_view name) const noexcept -> EffectBinding const *;
    [[nodiscard]] auto find_param(std::string_view name) const noexcept -> EffectParam const *;
    [[nodiscard]] auto replaces_scene_colour() const noexcept -> bool;
    [[nodiscard]] auto push_bytes() const noexcept -> std::size_t;
};

// Reads and checks a manifest: the shader path stays inside the shader directory, names are identifiers and unique, the
// dispatch names a binding that exists, defaults are inside their ranges, the push constants fit, and so on. The text
// of the first problem is returned.
[[nodiscard]] auto parse_effect_manifest(std::string_view json_text) -> std::expected<EffectManifest, std::string>;

// A value a script gave for a param: numbers (one to four), checked against the param's type and range. The reason
// is returned when it does not fit.
[[nodiscard]] auto check_effect_value(EffectParam const &param, std::span<double const> numbers)
        -> std::expected<std::array<double, 4>, std::string>;
