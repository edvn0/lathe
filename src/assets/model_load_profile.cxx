#include "assets/model_load_profile.hxx"

#include <format>

namespace {

    auto to_ms(ProfileNanos const &value) -> double {
        return static_cast<double>(value.load(std::memory_order_relaxed)) / 1'000'000.0;
    }

} // namespace

auto format_model_load_profile(ModelLoadProfile const &profile) -> std::string {
    auto const cpu_parse_total_ms = to_ms(profile.gltf_parse_ns) + to_ms(profile.material_resolve_ns) +
                                    to_ms(profile.primitive_extract_ns) + to_ms(profile.tangent_generation_ns) +
                                    to_ms(profile.lod_generation_ns);

    auto const gpu_upload_total_ms = to_ms(profile.material_creation_ns) + to_ms(profile.vertex_compression_ns) +
                                     to_ms(profile.geometry_upload_ns);

    // Summed across every texture job, which run concurrently on separate
    // thread_pool() workers -- this is aggregate CPU-seconds spent, not
    // wall-clock time, and can (and usually will) exceed total_wall_ns.
    auto const texture_cpu_total_ms = to_ms(profile.texture_cache_lookup_ns) + to_ms(profile.texture_decode_ns) +
                                      to_ms(profile.texture_mip_generation_ns) + to_ms(profile.texture_encode_ns) +
                                      to_ms(profile.texture_transcode_ns) + to_ms(profile.texture_cache_write_ns);

    return std::format(
            "  total wall time:  {:.2f} ms\n"
            "  cpu parse:        {:.2f} ms  (gltf parse {:.2f} / material resolve {:.2f} / primitive extract {:.2f} "
            "/ tangents {:.2f} / lods {:.2f})\n"
            "  gpu upload:       {:.2f} ms across {} frame(s)  (material create {:.2f} / vertex compress {:.2f} / "
            "geometry upload {:.2f})\n"
            "  textures:         {:.2f} ms of worker CPU time across {} texture(s) ({} cache hit(s), {} cache "
            "miss(es))\n"
            "                    cache lookup {:.2f} / decode {:.2f} / mip gen {:.2f} / encode {:.2f} / transcode "
            "{:.2f} / cache write {:.2f}",
            to_ms(profile.total_wall_ns), cpu_parse_total_ms, to_ms(profile.gltf_parse_ns),
            to_ms(profile.material_resolve_ns), to_ms(profile.primitive_extract_ns),
            to_ms(profile.tangent_generation_ns), to_ms(profile.lod_generation_ns), gpu_upload_total_ms,
            profile.gpu_upload_frames.load(std::memory_order_relaxed), to_ms(profile.material_creation_ns),
            to_ms(profile.vertex_compression_ns), to_ms(profile.geometry_upload_ns), texture_cpu_total_ms,
            profile.texture_count.load(std::memory_order_relaxed), profile.texture_cache_hits.load(std::memory_order_relaxed),
            profile.texture_cache_misses.load(std::memory_order_relaxed), to_ms(profile.texture_cache_lookup_ns),
            to_ms(profile.texture_decode_ns), to_ms(profile.texture_mip_generation_ns), to_ms(profile.texture_encode_ns),
            to_ms(profile.texture_transcode_ns), to_ms(profile.texture_cache_write_ns));
}
