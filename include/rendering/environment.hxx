#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "assets/hdr_image.hxx"
#include "core/renderer_error.hxx"
#include "gpu/buffer.hxx"
#include "gpu/gpu_resource_table.hxx"
#include "gpu/image_storage.hxx"
#include "gpu/sampler_storage.hxx"
#include "rendering/pipeline_graph_repository.hxx"
#include "scene/environment.hxx"

struct GpuEnvironmentSh {
    glm::vec4 coefficients[9]{};
};

static_assert(sizeof(GpuEnvironmentSh) == 144);

struct EnvironmentUboBlock {
    std::uint32_t flags = 0;
    float exposure = 1.0F;
    glm::vec4 rotation{1.0F, 0.0F, 0.0F, 1.0F};
    glm::vec4 intensity{1.0F, 1.0F, 0.0F, 0.0F};

    std::uint32_t radiance_cube_texture = 0;
    std::uint32_t prefilter_cube_texture = 0;
    std::uint32_t brdf_lut_texture = 0;
    std::uint32_t sampler = 0;

    std::array<glm::vec4, 4> sky_perez{};
    glm::vec4 sky_zenith{};
    glm::vec4 sky_ground{};
    glm::vec4 sun_direction_cos_radius{0.0F, 1.0F, 0.0F, 1.0F};
    glm::vec4 sun_disc_radiance{};

    VkDeviceAddress sh_address = 0;
};

static_assert(sizeof(EnvironmentUboBlock) == 192);
static_assert(offsetof(EnvironmentUboBlock, rotation) == 8);
static_assert(offsetof(EnvironmentUboBlock, radiance_cube_texture) == 40);
static_assert(offsetof(EnvironmentUboBlock, sky_perez) == 56);
static_assert(offsetof(EnvironmentUboBlock, sky_zenith) == 120);
static_assert(offsetof(EnvironmentUboBlock, sun_direction_cos_radius) == 152);
static_assert(offsetof(EnvironmentUboBlock, sh_address) == 184);

namespace environment_flag {
    inline constexpr std::uint32_t ibl_valid = 1U << 0;
    inline constexpr std::uint32_t skybox = 1U << 1;
    inline constexpr std::uint32_t sky_procedural = 1U << 2;
    inline constexpr std::uint32_t specular_occlusion = 1U << 3;
    inline constexpr std::uint32_t fog_sky = 1U << 4;
    inline constexpr std::uint32_t multi_scatter = 1U << 5;
    inline constexpr std::uint32_t fog_from_environment = 1U << 6;
    inline constexpr std::uint32_t debug_shift = 8;
}

enum class EnvironmentDebugView : std::uint8_t {
    none = 0,
    diffuse_only,
    specular_only,
    specular_occlusion,
    sky_prefilter,
    sky_irradiance,
    white_furnace,
    count,
};

[[nodiscard]]
constexpr auto to_string(EnvironmentDebugView view) noexcept -> std::string_view {
    switch (view) {
        case EnvironmentDebugView::none:
            return "None";
        case EnvironmentDebugView::diffuse_only:
            return "Diffuse IBL only";
        case EnvironmentDebugView::specular_only:
            return "Specular IBL only";
        case EnvironmentDebugView::specular_occlusion:
            return "Specular occlusion";
        case EnvironmentDebugView::sky_prefilter:
            return "Sky = prefilter LOD";
        case EnvironmentDebugView::sky_irradiance:
            return "Sky = SH irradiance";
        case EnvironmentDebugView::white_furnace:
            return "White furnace";
        case EnvironmentDebugView::count:
            break;
    }

    return "Unknown";
}

struct EnvironmentPipelines {
    PipelineNodeHandle brdf_lut{};
    PipelineNodeHandle equirect_to_cube{};
    PipelineNodeHandle sky_to_cube{};
    PipelineNodeHandle downsample{};
    PipelineNodeHandle sh_project{};
    PipelineNodeHandle prefilter{};
};

enum class EnvironmentPhase : std::uint8_t {
    idle,
    decoding,
    building,
    ready,
    failed,
};

struct EnvironmentStatus {
    EnvironmentPhase phase = EnvironmentPhase::idle;

    std::uint32_t step = 0;
    std::uint32_t step_count = 0;

    std::string message;

    bool ibl_live = false;
};

struct EnvironmentDebugSettings {
    EnvironmentDebugView view = EnvironmentDebugView::none;

    float prefilter_lod = 0.0F;

    bool amortize_rebuilds = true;
};

struct EnvironmentValidation {
    bool ran = false;
    bool passed = false;

    float lut_max_error = 0.0F;

    float sh_max_relative_error = 0.0F;

    std::string summary;
};

class EnvironmentSystem {
public:
    struct CreateInfo {
        VulkanContext *context = nullptr;
        ImageStorage *images = nullptr;
        SamplerStorage *samplers = nullptr;
        PipelineGraphRepository *pipelines = nullptr;

        EnvironmentPipelines handles{};
        std::uint32_t frames_in_flight = 2;
    };

    EnvironmentSystem() = default;
    ~EnvironmentSystem();

    EnvironmentSystem(EnvironmentSystem const &) = delete;
    auto operator=(EnvironmentSystem const &) -> EnvironmentSystem & = delete;
    EnvironmentSystem(EnvironmentSystem &&) noexcept = delete;
    auto operator=(EnvironmentSystem &&) noexcept -> EnvironmentSystem & = delete;

    [[nodiscard]]
    auto initialize(CreateInfo const &create_info) -> std::expected<void, RendererError>;

    auto set_environment(SceneEnvironment const &environment) -> void { desired_ = environment; }
    [[nodiscard]] auto environment() const noexcept -> SceneEnvironment const & { return desired_; }

    [[nodiscard]]
    auto debug_settings() noexcept -> EnvironmentDebugSettings & {
        return debug_;
    }
    [[nodiscard]]
    auto debug_settings() const noexcept -> EnvironmentDebugSettings const & {
        return debug_;
    }

    auto provide_hdr(std::string source, HdrImage image) -> void {
        provided_.insert_or_assign(std::move(source), std::make_shared<HdrImage const>(std::move(image)));
    }

    auto rebuild() noexcept -> void { ++generation_; }

    [[nodiscard]]
    auto prepare(VkCommandBuffer command_buffer, std::uint64_t frame_number) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto ubo_block() const -> EnvironmentUboBlock;

    [[nodiscard]]
    auto has_pending_record() const -> bool;

    auto record(VkCommandBuffer command_buffer, GpuResourceTable &resource_table, std::uint32_t frame_index,
                VkDeviceAddress ubo_address) -> void;

    [[nodiscard]]
    auto status() const -> EnvironmentStatus;

    [[nodiscard]]
    auto validate_against_cpu() -> EnvironmentValidation;

    [[nodiscard]] auto brdf_lut_texture_index() const noexcept -> std::uint32_t;
    [[nodiscard]] auto radiance_face_texture_index(std::uint32_t mip, std::uint32_t face) const noexcept
            -> std::uint32_t;
    [[nodiscard]] auto radiance_mip_count() const noexcept -> std::uint32_t { return radiance_.mip_count; }
    [[nodiscard]] auto prefilter_face_texture_index(std::uint32_t mip, std::uint32_t face) const noexcept
            -> std::uint32_t;
    [[nodiscard]] static constexpr auto prefilter_mip_count() noexcept -> std::uint32_t { return prefilter_mips; }

    [[nodiscard]] auto live_sh_address() const noexcept -> VkDeviceAddress;

    auto destroy() noexcept -> void;

    static constexpr std::uint32_t prefilter_size = 256;
    static constexpr std::uint32_t prefilter_mips = 6;
    static constexpr std::uint32_t brdf_lut_size = 128;
    static constexpr std::uint32_t procedural_cube_size = 256;
    static constexpr std::uint32_t sh_level_size = 32;

private:
    struct CubeResource {
        ImageHolder image;
        std::vector<ImageHolder> face_slots;
        std::uint32_t size = 0;
        std::uint32_t mip_count = 0;

        [[nodiscard]]
        auto face_slot(std::uint32_t mip, std::uint32_t face) const noexcept -> ImageHandle {
            return face_slots[(static_cast<std::size_t>(mip) * 6U) + face].handle();
        }

        CubeResource() = default;
        ~CubeResource() = default;
        CubeResource(CubeResource const &) = delete;
        auto operator=(CubeResource const &) -> CubeResource & = delete;
        CubeResource(CubeResource &&) noexcept = default;
        auto operator=(CubeResource &&other) noexcept -> CubeResource & {
            face_slots = std::move(other.face_slots);
            image = std::move(other.image);
            size = other.size;
            mip_count = other.mip_count;

            return *this;
        }
    };

    struct BuildKey {
        EnvironmentSource source = EnvironmentSource::flat_ambient;
        std::string hdr_source;
        std::uint32_t cube_size = 0;

        std::int32_t azimuth_centidegrees = 0;
        std::int32_t elevation_centidegrees = 0;
        std::int32_t turbidity_milli = 0;
        std::int32_t sky_intensity_milli = 0;
        std::array<std::int32_t, 3> ground_milli{};

        std::uint64_t generation = 0;

        auto operator==(BuildKey const &) const -> bool = default;
    };

    struct DecodeJob {
        std::string path;
        std::future<std::expected<HdrImage, HdrImageError>> future;
    };

    struct Build {
        BuildKey key;
        std::uint32_t target_set = 0;
        bool amortized = false;
        bool captured = false;
        std::uint32_t faces_done = 0;
    };

    struct FramePlan {
        bool brdf_lut = false;
        bool capture = false;
        std::uint32_t first_face = 0;
        std::uint32_t face_count = 0;
        bool flip = false;
        std::uint32_t target_set = 0;
    };

    struct Retired {
        std::uint64_t frame = 0;
        CubeResource cube;
        ImageHolder image;
        Buffer buffer;
    };

    [[nodiscard]] auto desired_key() const -> BuildKey;
    [[nodiscard]] auto create_cube(std::uint32_t size, std::uint32_t mip_count, std::string_view name,
                                   bool transfer_dst) -> std::expected<CubeResource, RendererError>;
    [[nodiscard]] auto finish_decode(VkCommandBuffer command_buffer, HdrImage const &image)
            -> std::expected<void, RendererError>;
    [[nodiscard]] auto upload_equirect(VkCommandBuffer command_buffer, HdrImage const &image)
            -> std::expected<void, RendererError>;
    [[nodiscard]] auto upload_cube(VkCommandBuffer command_buffer, HdrImage const &image)
            -> std::expected<void, RendererError>;
    [[nodiscard]] auto ensure_procedural_radiance() -> std::expected<void, RendererError>;
    auto retire(CubeResource &&cube) -> void;
    auto retire(ImageHolder &&image) -> void;
    auto retire(Buffer &&buffer) -> void;
    auto plan_frame(bool amortize) -> void;
    [[nodiscard]] auto pipelines_ready() const -> bool;
    auto detect_shader_changes() -> void;

    [[nodiscard]]
    auto prefilter_set_valid(std::uint32_t set) const noexcept -> bool {
        return prefilter_[set].image.handle().valid();
    }

    VulkanContext *context_ = nullptr;
    ImageStorage *images_ = nullptr;
    SamplerStorage *samplers_ = nullptr;
    PipelineGraphRepository *pipelines_ = nullptr;
    EnvironmentPipelines handles_{};
    std::uint32_t frames_in_flight_ = 2;
    std::uint64_t frame_number_ = 0;

    SceneEnvironment desired_{};
    EnvironmentDebugSettings debug_{};
    std::uint64_t generation_ = 0;

    ImageHolder brdf_lut_;
    CubeResource radiance_;
    EnvironmentSource radiance_source_ = EnvironmentSource::flat_ambient;
    std::string radiance_hdr_;
    bool radiance_is_cube_source_ = false;
    std::array<CubeResource, 2> prefilter_;
    Buffer sh_buffer_;
    ImageHolder equirect_;
    SamplerHandle equirect_sampler_{};
    std::vector<Retired> retired_;

    std::unordered_map<std::string, std::shared_ptr<HdrImage const>> provided_;
    std::optional<DecodeJob> decode_;
    std::string decode_error_;
    std::string error_message_;
    std::string loaded_hdr_;
    bool equirect_pending_projection_ = false;

    bool lut_ready_ = false;
    bool radiance_captured_ = false;
    std::optional<Build> building_;
    std::optional<BuildKey> live_key_;
    std::uint32_t live_set_ = 0;
    FramePlan plan_{};

    std::array<ShaderObjectHandle, 6> shader_handles_{};
    bool initialised_ = false;
};
