#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

// Texture jobs for one model run concurrently, so the accumulators are atomic.
using ProfileNanos = std::atomic<std::int64_t>;

// Timing breakdown of one streamed model load, from ModelStreamer::request() to install. A null profile makes
// every ScopedProfileSample a no-op. CPU parse runs on one background thread, GPU upload on the render thread
// across frames, and texture jobs concurrently on thread_pool().
struct ModelLoadProfile {
    // CPU parse
    ProfileNanos gltf_parse_ns{0}; // fastgltf: reading and validating the file
    ProfileNanos material_resolve_ns{0}; // load_material_cpu: image source resolution, sampler selection
    ProfileNanos primitive_extract_ns{0}; // load_primitive_cpu: reading vertex/index accessors
    ProfileNanos tangent_generation_ns{0}; // generate_tangents: MikkTSpace + weld/optimize
    ProfileNanos lod_generation_ns{0}; // generate_mesh_lods: meshopt_simplify per level
    ProfileNanos vertex_compression_ns{0}; // prepare_primitive_gpu_data: compress_vertices
    ProfileNanos meshlet_build_ns{0}; // prepare_primitive_gpu_data: build_meshlets per LOD

    // GPU upload
    ProfileNanos material_creation_ns{0}; // to_gpu_material + MaterialStorage::create_material
    ProfileNanos geometry_upload_ns{0}; // GeometryArena vertex/index/meshlet uploads
    std::atomic<std::uint32_t> gpu_upload_frames{0}; // number of process_ready() calls

    // Texture pipeline
    ProfileNanos texture_cache_lookup_ns{0}; // try_load_cached: stat + ktxTexture2_CreateFromNamedFile
    ProfileNanos texture_decode_ns{0}; // DecodedImage::load_from_file/memory (cache misses only)
    ProfileNanos texture_mip_generation_ns{0}; // generate_mip_chain (cache misses only)
    ProfileNanos texture_encode_ns{0}; // ktxTexture2_CompressBasisEx (cache misses only)
    ProfileNanos texture_transcode_ns{0}; // ktxTexture2_TranscodeBasis (hits and misses)
    ProfileNanos texture_cache_write_ns{0}; // ktxTexture2_WriteToNamedFile (cache misses only)
    std::atomic<std::uint32_t> texture_count{0};
    std::atomic<std::uint32_t> texture_cache_hits{0};
    std::atomic<std::uint32_t> texture_cache_misses{0};

    // Texture jobs queued for this model. ModelStreamer waits until texture_count reaches this before logging, so
    // textures finishing after install are included.
    std::atomic<std::uint32_t> expected_texture_count{0};

    // request() to installed. The only field that isn't a sum of busy time.
    ProfileNanos total_wall_ns{0};
};

// Adds the elapsed time to `*target` on scope exit. A null `target` makes it a no-op.
class ScopedProfileSample {
public:
    explicit ScopedProfileSample(ProfileNanos *target) noexcept
        : target_(target), start_(target != nullptr ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{}) {}

    ScopedProfileSample(ScopedProfileSample const &) = delete;
    auto operator=(ScopedProfileSample const &) -> ScopedProfileSample & = delete;

    // Adds the elapsed time now and disarms the destructor. Later calls are no-ops.
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

// Multi-line breakdown in milliseconds and percentages.
[[nodiscard]]
auto format_model_load_profile(ModelLoadProfile const &profile) -> std::string;
