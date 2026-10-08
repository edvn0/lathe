#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

using ProfileNanos = std::atomic<std::int64_t>;

struct ModelLoadProfile {
    ProfileNanos gltf_parse_ns{0};
    ProfileNanos material_resolve_ns{0};
    ProfileNanos primitive_extract_ns{0};
    ProfileNanos tangent_generation_ns{0};
    ProfileNanos lod_generation_ns{0};
    ProfileNanos vertex_compression_ns{0};
    ProfileNanos meshlet_build_ns{0};

    ProfileNanos material_creation_ns{0};
    ProfileNanos geometry_upload_ns{0};
    std::atomic<std::uint32_t> gpu_upload_frames{0};

    ProfileNanos texture_cache_lookup_ns{0};
    ProfileNanos texture_decode_ns{0};
    ProfileNanos texture_mip_generation_ns{0};
    ProfileNanos texture_encode_ns{0};
    ProfileNanos texture_transcode_ns{0};
    ProfileNanos texture_cache_write_ns{0};
    std::atomic<std::uint32_t> texture_count{0};
    std::atomic<std::uint32_t> texture_cache_hits{0};
    std::atomic<std::uint32_t> texture_cache_misses{0};

    std::atomic<std::uint32_t> expected_texture_count{0};

    ProfileNanos total_wall_ns{0};
};

class ScopedProfileSample {
public:
    explicit ScopedProfileSample(ProfileNanos *target) noexcept :
        target_(target),
        start_(target != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}

    ScopedProfileSample(ScopedProfileSample const &) = delete;
    auto operator=(ScopedProfileSample const &) -> ScopedProfileSample & = delete;

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

[[nodiscard]]
auto format_model_load_profile(ModelLoadProfile const &profile) -> std::string;
