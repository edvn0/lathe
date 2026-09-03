#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

// Nanosecond accumulator. Fields of this type on ModelLoadProfile below get
// added to from more than one thread concurrently -- the texture pipeline
// runs many jobs in parallel across thread_pool() workers for a single
// model load (see TextureStreamer::request) -- so every accumulator has to
// tolerate concurrent writes rather than just being a plain std::int64_t.
using ProfileNanos = std::atomic<std::int64_t>;

// Wall-clock breakdown of one streamed model's load, from ModelStreamer::
// request() to the frame its Model gets installed. Threaded through as an
// optional std::shared_ptr<ModelLoadProfile> (see ModelCpuData::profile) --
// a null profile makes every ScopedProfileSample a no-op, so leaving it
// unset costs nothing. Grouped by which thread the work happens on:
//
//   - CPU parse fields are filled in by load_model_cpu() on a single
//     thread_pool() background thread (one model load, one thread -- no
//     concurrent writers, but ProfileNanos is used uniformly anyway).
//   - GPU upload fields are filled in by step_model_gpu_upload() on the
//     render thread, one call per frame until the model finishes.
//   - Texture fields are filled in by the texture pipeline
//     (texture_pipeline.cxx), with one job per texture running concurrently
//     on its own thread_pool() worker -- these genuinely need the atomics.
//
// See format_model_load_profile() to turn this into something loggable.
struct ModelLoadProfile {
    // ---- CPU parse (load_model_cpu, one background thread) ----
    ProfileNanos gltf_parse_ns{0}; // fastgltf parser: reading + validating the file
    ProfileNanos material_resolve_ns{0}; // load_material_cpu: image source resolution, sampler selection
    ProfileNanos primitive_extract_ns{0}; // load_primitive_cpu: reading vertex/index accessors
    ProfileNanos tangent_generation_ns{0}; // generate_tangents: MikkTSpace + weld/optimize
    ProfileNanos lod_generation_ns{0}; // generate_mesh_lods: meshopt_simplify per level

    // ---- GPU upload (step_model_gpu_upload, render thread, spread across frames) ----
    ProfileNanos material_creation_ns{0}; // to_gpu_material + MaterialStorage::create_material
    ProfileNanos vertex_compression_ns{0}; // compress_vertices
    ProfileNanos geometry_upload_ns{0}; // GeometryArena::allocate_vertices/allocate_indices
    std::atomic<std::uint32_t> gpu_upload_frames{0}; // how many process_ready() calls it took

    // ---- Texture pipeline (thread_pool workers, one job per texture, concurrent) ----
    ProfileNanos texture_cache_lookup_ns{0}; // try_load_cached: stat + ktxTexture2_CreateFromNamedFile
    ProfileNanos texture_decode_ns{0}; // DecodedImage::load_from_file/memory (cache misses only)
    ProfileNanos texture_mip_generation_ns{0}; // generate_mip_chain (cache misses only)
    ProfileNanos texture_encode_ns{0}; // ktxTexture2_CompressBasisEx (cache misses only)
    ProfileNanos texture_transcode_ns{0}; // ktxTexture2_TranscodeBasis (hits and misses)
    ProfileNanos texture_cache_write_ns{0}; // ktxTexture2_WriteToNamedFile (cache misses only)
    std::atomic<std::uint32_t> texture_count{0};
    std::atomic<std::uint32_t> texture_cache_hits{0};
    std::atomic<std::uint32_t> texture_cache_misses{0};

    // How many texture jobs TextureStreamer actually queued for this model
    // (set as each TextureStreamer::request()/request_from_memory() call
    // succeeds -- see texture_streamer.cxx). ModelStreamer compares this
    // against texture_count to know when every one of them has finished --
    // textures stream in independently of the model's own GPU upload and
    // can finish well after it installs, so this is what lets the final
    // logged profile cover every texture instead of whichever few happened
    // to finish first.
    std::atomic<std::uint32_t> expected_texture_count{0};

    // Stamped once by ModelStreamer across the whole request()-to-installed
    // span -- the only field that isn't a sum of some thread's busy time, so
    // it's what actually tells you how long the caller waited. Necessarily
    // >= every other field's wall-clock contribution, since those all
    // happen inside this span, often overlapping across threads.
    ProfileNanos total_wall_ns{0};
};

// RAII sample: adds the elapsed time since construction to `*target` on
// scope exit. `target` may be null (profiling off for this load) -- every
// operation becomes a no-op in that case, including skipping the now() call,
// so an unprofiled model load pays nothing for this.
class ScopedProfileSample {
public:
    explicit ScopedProfileSample(ProfileNanos *target) noexcept
        : target_(target), start_(target != nullptr ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{}) {}

    ScopedProfileSample(ScopedProfileSample const &) = delete;
    auto operator=(ScopedProfileSample const &) -> ScopedProfileSample & = delete;

    // Adds the elapsed time so far and disarms the destructor -- lets a
    // sample cover a prefix of its enclosing scope (e.g. "everything up to
    // here is section A, the rest is section B") without a nested block.
    // Safe to call at most once; a second call or the destructor afterward
    // is a no-op.
    auto stop() noexcept -> void {
        if (target_ == nullptr) {
            return;
        }

        auto const elapsed = std::chrono::steady_clock::now() - start_;

        target_->fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                           std::memory_order_relaxed);

        target_ = nullptr;
    }

    ~ScopedProfileSample() { stop(); }

private:
    ProfileNanos *target_;
    std::chrono::steady_clock::time_point start_;
};

// Multi-line, milliseconds-and-percentages breakdown of `profile`, grouped
// into its CPU parse / GPU upload / texture sections -- see ModelStreamer::
// process_ready for where this gets logged once a streamed model finishes.
[[nodiscard]]
auto format_model_load_profile(ModelLoadProfile const &profile) -> std::string;
