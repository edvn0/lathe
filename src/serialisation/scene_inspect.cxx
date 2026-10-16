#include "serialisation/scene_inspect.hxx"

#include <algorithm>
#include <string_view>

#include "core/error_describe.hxx"
#include "serialisation/byte_stream.hxx"
#include "serialisation/scene_codec.hxx"
#include "serialisation/string_table.hxx"

namespace {

    constexpr std::size_t max_listed_strings = 10'000;

    [[nodiscard]] auto status_of(SceneSectionInfo const &info, std::uint16_t version) -> InspectedSectionStatus {
        if (version > info.current_version) {
            return InspectedSectionStatus::newer;
        }

        if (version < info.oldest_readable) {
            return InspectedSectionStatus::too_old;
        }

        return version < info.current_version ? InspectedSectionStatus::older : InspectedSectionStatus::current;
    }

    [[nodiscard]] auto split_strings(std::span<std::byte const> blob) -> std::vector<std::string> {
        std::vector<std::string> result;
        auto const *const chars = reinterpret_cast<char const *>(blob.data());
        std::size_t start = 1;

        for (std::size_t at = 1; at < blob.size() && result.size() < max_listed_strings; ++at) {
            if (chars[at] == '\0') {
                result.emplace_back(chars + start, at - start);
                start = at + 1;
            }
        }

        return result;
    }

} // namespace

auto inspect_scene_payload(std::span<std::byte const> payload, std::uint16_t chunk_version) -> InspectedScene {
    InspectedScene scene{.chunk_version = chunk_version, .payload_size = payload.size()};

    ByteReader reader{payload};

    while (reader.remaining() > 0) {
        InspectedSection section;
        section.type = reader.read<std::uint32_t>();
        section.version = reader.read<std::uint16_t>();
        static_cast<void>(reader.read<std::uint16_t>());
        section.size = reader.read<std::uint64_t>();
        auto const bytes = reader.read_bytes(static_cast<std::size_t>(section.size));

        if (reader.failed()) {
            scene.layout_problem = "a section header or body runs past the end of the payload";
            break;
        }

        if (auto const info = scene_section_info(section.type)) {
            section.name = std::string{info->name};
            section.current_version = info->current_version;
            section.status = status_of(*info, section.version);
            section.inline_strings = info->string_offsets_from != 0 && section.version < info->string_offsets_from;
        } else {
            section.name = "unknown";
            section.status = InspectedSectionStatus::unknown;
        }

        if (section.type == scene_section::strings && section.status != InspectedSectionStatus::newer) {
            scene.strings_size = bytes.size();

            if (scene.strings.has_value() || !StringBlob::parse(bytes)) {
                scene.layout_problem = "the strings section is duplicated or not NUL-terminated";
            } else {
                scene.strings = split_strings(bytes);
            }
        }

        scene.sections.push_back(std::move(section));
    }

    auto const decoded = decode_scene(payload, chunk_version);

    if (!decoded) {
        scene.decode_error = describe(decoded.error().cause.value_or(ErrorCause{}));
        return scene;
    }

    scene.counts = {
            {"models", decoded->models.size()},
            {"textures", decoded->textures.size()},
            {"materials", decoded->materials.size()},
            {"entities", decoded->entities.size()},
            {"model components", decoded->model_components.size()},
            {"instanced models", decoded->instanced_models.size()},
            {"point lights", decoded->point_lights.size()},
            {"spot lights", decoded->spot_lights.size()},
            {"particle emitters", decoded->particle_emitters.size()},
            {"texts", decoded->texts.size()},
            {"rigid bodies", decoded->rigid_bodies.size()},
            {"scripts", decoded->scripts.size()},
            {"behaviours", decoded->behaviours.size()},
            {"lifetimes", decoded->lifetimes.size()},
    };

    return scene;
}

auto inspect_scene_file(std::filesystem::path const &path) -> std::expected<SceneFileInfo, LbfError> {
    auto reader = LbfReader::open(path);

    if (!reader) {
        return std::unexpected(reader.error());
    }

    SceneFileInfo info{.path = path, .header = reader->header()};

    for (auto const &entry: reader->chunks()) {
        info.chunks.push_back(InspectedChunk{
                .type = entry.type,
                .type_name = lbf_chunk_type_name(entry.type),
                .id = entry.id,
                .version = entry.version,
                .compressed = entry.compression != LbfCompression::none,
                .stored_size = entry.stored_size,
                .raw_size = entry.raw_size,
        });
    }

    if (auto const *const entry = reader->find(lbf_chunk::scene)) {
        if (auto payload = reader->read_chunk(*entry)) {
            info.scene = inspect_scene_payload(*payload, entry->version);
        } else {
            info.scene_problem = describe(payload.error().cause.value_or(ErrorCause{}));
        }
    }

    return info;
}
