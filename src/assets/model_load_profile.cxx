#include "assets/model_load_profile.hxx"

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace {

    auto to_ms(ProfileNanos const &value) -> double {
        return static_cast<double>(value.load(std::memory_order_relaxed)) / 1'000'000.0;
    }

    // "name 12 ms, name 3 ms" for the stages worth showing (>= 0.5 ms), with their sum up front.
    auto stage_list(std::initializer_list<std::pair<std::string_view, double>> stages) -> std::string {
        auto total = 0.0;
        auto detail = std::string{};
        for (auto const &[name, ms] : stages) {
            total += ms;
            if (ms >= 0.5) {
                detail += std::format("{}{} {:.0f}", detail.empty() ? "" : ", ", name, ms);
            }
        }
        return std::format("{:.0f} ms{}", total, detail.empty() ? "" : " (" + detail + ")");
    }

}

auto format_model_load_profile(ModelLoadProfile const &profile) -> std::string {
    auto const cpu = stage_list({{"gltf", to_ms(profile.gltf_parse_ns)},
                                 {"materials", to_ms(profile.material_resolve_ns)},
                                 {"extract", to_ms(profile.primitive_extract_ns)},
                                 {"tangents", to_ms(profile.tangent_generation_ns)},
                                 {"lods", to_ms(profile.lod_generation_ns)},
                                 {"compress", to_ms(profile.vertex_compression_ns)},
                                 {"meshlets", to_ms(profile.meshlet_build_ns)}});

    auto const textures = stage_list({{"lookup", to_ms(profile.texture_cache_lookup_ns)},
                                      {"decode", to_ms(profile.texture_decode_ns)},
                                      {"mips", to_ms(profile.texture_mip_generation_ns)},
                                      {"encode", to_ms(profile.texture_encode_ns)},
                                      {"transcode", to_ms(profile.texture_transcode_ns)},
                                      {"write", to_ms(profile.texture_cache_write_ns)}});

    auto const gpu_ms = to_ms(profile.material_creation_ns) + to_ms(profile.geometry_upload_ns);

    return std::format("  wall {:.0f} ms | gpu upload {:.0f} ms over {} frames\n"
                       "  cpu (summed over workers): {}\n"
                       "  {} textures ({} cached, {} built): {}",
                       to_ms(profile.total_wall_ns), gpu_ms, profile.gpu_upload_frames.load(std::memory_order_relaxed),
                       cpu, profile.texture_count.load(std::memory_order_relaxed),
                       profile.texture_cache_hits.load(std::memory_order_relaxed),
                       profile.texture_cache_misses.load(std::memory_order_relaxed), textures);
}
