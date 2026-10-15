#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

// glTF is read by fastgltf (the reference path: skins, animation, lights); the rest go through Assimp, see
// assets/assimp_import.hxx.
inline constexpr std::array<std::string_view, 2> gltf_model_extensions{".gltf", ".glb"};

inline constexpr std::array<std::string_view, 9> assimp_model_extensions{".fbx", ".obj", ".dae", ".ply", ".stl",
                                                                          ".usd", ".usda", ".usdc", ".usdz"};

inline constexpr std::array<std::string_view, 11> model_file_extensions{".gltf", ".glb", ".fbx", ".obj",
                                                                        ".dae",  ".ply", ".stl", ".usd",
                                                                        ".usda", ".usdc", ".usdz"};

[[nodiscard]]
inline auto model_extension_of(std::filesystem::path const &path) -> std::string {
    auto extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

[[nodiscard]]
inline auto is_gltf_model(std::filesystem::path const &path) -> bool {
    return std::ranges::contains(gltf_model_extensions, model_extension_of(path));
}

[[nodiscard]]
inline auto is_assimp_model(std::filesystem::path const &path) -> bool {
    return std::ranges::contains(assimp_model_extensions, model_extension_of(path));
}

[[nodiscard]]
inline auto is_supported_model(std::filesystem::path const &path) -> bool {
    return is_gltf_model(path) || is_assimp_model(path);
}
