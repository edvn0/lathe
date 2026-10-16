#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "serialisation/lbf_container.hxx"
#include "serialisation/lbf_error.hxx"

// A read-only look inside a scene file: the LBF container, the SCEN chunk's sections with their versions, the shared
// string table and whether the scene decodes. Nothing here instantiates the scene.

struct InspectedChunk {
    std::uint32_t type = 0;
    std::string type_name;
    std::uint64_t id = 0;
    std::uint16_t version = 0;
    bool compressed = false;
    std::uint64_t stored_size = 0;
    std::uint64_t raw_size = 0;
};

enum class InspectedSectionStatus : std::uint8_t {
    current,
    older, // an older version this build still reads
    too_old, // older than this build reads: decoding fails
    newer, // from a newer engine: skipped
    unknown, // not a section this build knows: skipped
};

struct InspectedSection {
    std::uint32_t type = 0;
    std::string name;
    std::uint16_t version = 0;
    std::uint16_t current_version = 0;
    std::uint64_t size = 0;
    InspectedSectionStatus status = InspectedSectionStatus::current;
    bool inline_strings = false; // this version still stores its strings inline
};

struct InspectedScene {
    std::uint16_t chunk_version = 0;
    std::uint64_t payload_size = 0;
    std::vector<InspectedSection> sections;

    std::optional<std::vector<std::string>> strings; // the strings section's contents, if present and well formed
    std::uint64_t strings_size = 0;

    std::string layout_problem; // the section framing was cut short or the table is malformed
    std::optional<std::string> decode_error; // set when decode_scene refuses the payload
    std::vector<std::pair<std::string, std::size_t>> counts; // what decoded, e.g. {"entities", 12}
};

struct SceneFileInfo {
    std::filesystem::path path;
    LbfFileHeader header;
    std::vector<InspectedChunk> chunks;
    std::optional<InspectedScene> scene; // empty when the file has no SCEN chunk
    std::string scene_problem; // why a present SCEN chunk could not be read
};

[[nodiscard]] auto inspect_scene_payload(std::span<std::byte const> payload, std::uint16_t chunk_version)
        -> InspectedScene;

[[nodiscard]] auto inspect_scene_file(std::filesystem::path const &path) -> std::expected<SceneFileInfo, LbfError>;
