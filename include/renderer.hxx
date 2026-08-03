#pragma once

#include <atomic>
#include <queue>
#include <volk.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "buffer.hxx"
#include "config.hxx"
#include "error_context.hxx"
#include "forward.hxx"
#include "forward_target.hxx"
#include "geometry_arena.hxx"
#include "gpu_resource_table.hxx"
#include "image_storage.hxx"
#include "load_model.hxx"
#include "material_storage.hxx" // TextureHandle, MaterialCreateInfo now live here
#include "pipeline_graph_repository.hxx"
#include "pipeline_storage.hxx"
#include "sampler_storage.hxx"
#include "shader_change_queue.hxx"
#include "slang_compiler.hxx"

enum class RenderStage : std::uint32_t { FullFrame = 0, DepthPrepass, ForwardPass, Composition, Count };
constexpr auto to_string(RenderStage stage) -> std::string_view {
    switch (stage) {
        case RenderStage::FullFrame:
            return "Full Frame";
        case RenderStage::DepthPrepass:
            return "Depth prepass";
        case RenderStage::ForwardPass:
            return "Forward Pass";
        case RenderStage::Composition:
            return "Composition Pass";
        default:
            return "Unknown";
    }
}
inline constexpr std::uint32_t stage_count = static_cast<std::uint32_t>(RenderStage::Count);
inline constexpr std::uint32_t query_count = stage_count * 2;

struct StageTimings {
    std::array<float, stage_count> milliseconds{};
    bool valid = false;
};

struct ModelHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return index != 0;
    }

    auto operator==(ModelHandle const &) const -> bool = default;
};

struct MeshHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return index != 0;
    }

    auto operator==(MeshHandle const &) const -> bool = default;
};

struct SubmeshCreateInfo {
    MeshGeometry geometry{};
    MaterialHandle material{};
};

struct MeshCreateInfo {
    std::span<const SubmeshCreateInfo> submeshes;
};

struct SwapchainImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
};

enum class RendererErrorType : std::uint8_t {
    invalid_argument,
    invalid_mesh,
    invalid_model,
    invalid_material,
    unsupported_index_type,
    capacity_exceeded,
    size_overflow,
    model_load_error,
    geometry_error,
    material_error,
    image_error,
    forward_target_error,
    device_error,
    pipeline_error,
    compiler_error,
    pipeline_storage_error,
    gpu_resource_table_error,
    pipeline_graph_error,
    invalid_pipeline,
};

struct UBO {
    glm::mat4 view_projection;
    glm::mat4 view;
    glm::mat4 projection;

    glm::vec3 camera_position;
    glm::vec3 fog_colour;
    float fog_extinction = 0.003F;
    float fog_inscattering = 1.0F;
};

struct RendererError {
    RendererErrorType type = RendererErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

static_assert(std::is_copy_constructible_v<RendererError>,
             "RendererError must stay copyable -- std::expected<T, RendererError> copies it throughout this codebase");

struct RendererCreateInfo {
    VkExtent2D extent{};

    std::uint32_t frames_in_flight = 0;

    VkDeviceSize geometry_capacity = 256UZ * 1024UZ * 1024UZ;

    std::uint32_t material_capacity = 4096;
    std::uint32_t mesh_capacity = 4096;
    std::uint32_t model_capacity = 1024;
    std::uint32_t image_capacity = 4096;
    std::uint32_t sampler_capacity = 128;
    std::uint32_t pipeline_capacity = 128;

    VkFormat hdr_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;

    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    std::uint32_t maximum_draw_count = 65536;
    std::uint32_t maximum_submission_count = 65536;
};

struct Renderer {
    explicit Renderer(VulkanContext &context) noexcept;

    Renderer(Renderer const &) = delete;
    auto operator=(Renderer const &) -> Renderer & = delete;

    Renderer(Renderer &&) = delete;
    auto operator=(Renderer &&) -> Renderer & = delete;

    [[nodiscard]]
    auto initialize(RendererCreateInfo const &create_info) -> std::expected<void, RendererError>;

    auto destroy() noexcept -> void;

    [[nodiscard]]
    auto load_model(std::filesystem::path const &path) -> std::expected<ModelHandle, RendererError>;

    // Uploads already-CPU-side geometry (e.g. procedurally generated engine
    // primitives — see primitive_meshes.hxx / engine_models.hxx) through the
    // same GPU upload path as load_model, without touching disk or a glTF
    // parser. Skips the path-based model cache load_model uses.
    [[nodiscard]]
    auto create_model_from_cpu_data(ModelCpuData const &cpu_data) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model, MaterialHandle fallback_material)
            -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto create_model(Model const &model) -> std::expected<ModelHandle, RendererError>;

    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 const &transform) -> std::expected<void, RendererError>;
    [[nodiscard]]
    auto submit_model(ModelHandle model, glm::mat4 &&) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto create_material(MaterialCreateInfo const &create_info) -> std::expected<MaterialHandle, RendererError>;

    [[nodiscard]]
    auto update_material(MaterialHandle handle, MaterialCreateInfo const &create_info)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto destroy_material(MaterialHandle handle) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto create_mesh(MeshCreateInfo const &create_info) -> std::expected<MeshHandle, RendererError>;

    [[nodiscard]]
    auto destroy_mesh(MeshHandle handle) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto submit_mesh(MeshHandle mesh, glm::mat4 const &transform) -> std::expected<void, RendererError>;

    struct CameraMatrices {
        glm::mat4 view;
        glm::mat4 projection;
    };
    [[nodiscard]]
    auto prepare_frame(VkCommandBuffer command_buffer, const CameraMatrices &, std::uint32_t frame_index)
            -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto record_frame(VkCommandBuffer command_buffer, SwapchainImage const &swapchain_image, std::uint32_t frame_index,
                      std::function<void(VkCommandBuffer)> const &overlay = {}) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto resize(VkExtent2D extent) -> std::expected<void, RendererError>;

    [[nodiscard]]
    auto default_material() const noexcept -> MaterialHandle {
        return default_material_handle_;
    }

    [[nodiscard]]
    auto geometry_arena() noexcept -> GeometryArena & {
        return geometry_arena_;
    }

    auto notify_shader_file_changed(std::filesystem::path path) -> void { shader_change_queue_.push(std::move(path)); }


    [[nodiscard]] auto aspect(std::uint32_t index) const -> float {
        return static_cast<float>(frames_[index].forward_target.extent().width) /
               static_cast<float>(frames_[index].forward_target.extent().height);
    }

    void queue_render_thread_event(std::function<void()> &&);
    void drain_event_queue() {
        if (queued_events_.load(std::memory_order_relaxed) == 0) [[likely]] {
            return;
        }

        std::queue<std::function<void()>> local_queue;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            std::swap(event_queue_, local_queue);
        }

        while (!local_queue.empty()) {
            local_queue.front()();
            local_queue.pop();
        }
    }

    [[nodiscard]] auto context() noexcept -> VulkanContext & { return context_; }

    [[nodiscard]] auto image_storage() noexcept -> ImageStorage & { return image_storage_; }
    [[nodiscard]] auto sampler_storage() noexcept -> SamplerStorage & { return sampler_storage_; }
    [[nodiscard]] auto resource_table() noexcept -> GpuResourceTable & { return gpu_resource_table_; }

    [[nodiscard]] auto resolve_pipeline(PipelineNodeHandle handle) const noexcept -> Pipeline const * {
        return pipeline_graph_.resolve(handle);
    }

    [[nodiscard]] auto register_pipeline(PipelineRegisterInfo info)
            -> std::expected<PipelineNodeHandle, RendererError> {
        auto registered = pipeline_graph_.register_pipeline(compiler, std::move(info));

        if (!registered) {
            return std::unexpected(RendererError{
                    .type = RendererErrorType::pipeline_graph_error,
                    .cause = ErrorCause{Boxed<PipelineGraphError>{registered.error()}},
            });
        }

        return *registered;
    }

    [[nodiscard]] auto last_frame_timings() const noexcept -> StageTimings const & { return last_frame_timings_; }


    auto wait_idle() -> std::expected<void, RendererError>;

private:
    struct Submesh {
        MeshGeometry geometry{};
        MaterialHandle material{};
    };

    struct MeshSlot {
        std::vector<Submesh> submeshes;

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
    };

    struct Submission {
        MeshHandle mesh{};
        glm::mat4 transform{1.0F};
    };

    struct alignas(16) GpuDraw {
        VkDeviceAddress vertex_address = 0;

        std::uint32_t material_index = 0;
        std::uint32_t transform_index = 0;
    };

    static_assert(std::is_trivially_copyable_v<GpuDraw>);

    static_assert(sizeof(GpuDraw) == 16);

    struct RendererFrame {
        Buffer upload_buffer{};
        Buffer draw_buffer{};
        Buffer transform_buffer{};
        Buffer indirect_buffer{};

        ForwardTarget forward_target{};

        VkDeviceSize draw_upload_offset = 0;
        VkDeviceSize transform_upload_offset = 0;
        VkDeviceSize indirect_upload_offset = 0;

        std::vector<GpuDraw> draws;
        std::vector<glm::mat4> transforms;

        std::vector<VkDrawIndexedIndirectCommand> indirect_commands;

        // Number of VkDrawIndexedIndirectCommand entries in indirect_commands
        // (one per unique (mesh, submesh) batch this frame) — NOT the number
        // of GpuDraw / instance entries in `draws`. This is the value that
        // must be passed as drawCount to vkCmdDrawIndexedIndirect.
        std::uint32_t indirect_command_count = 0;
    };

    struct ModelDraw {
        MeshHandle mesh{};
        glm::mat4 local_transform{1.0F};
    };

    struct ModelSlot {
        std::vector<ModelDraw> draws;

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
    };

    struct ModelSubmission {
        ModelHandle model{};
        glm::mat4 transform{1.0F};
    };

    [[nodiscard]]
    auto mesh_slot(MeshHandle handle) noexcept -> MeshSlot *;

    [[nodiscard]]
    auto mesh_slot(MeshHandle handle) const noexcept -> MeshSlot const *;

    [[nodiscard]]
    auto model_slot(ModelHandle handle) noexcept -> ModelSlot *;

    [[nodiscard]]
    auto model_slot(ModelHandle handle) const noexcept -> ModelSlot const *;

    [[nodiscard]]
    auto upload_frame_data(VkCommandBuffer command_buffer, RendererFrame &frame) -> std::expected<void, RendererError>;

    auto clear_submissions() noexcept -> void;

    VulkanContext &context_;

    VkFormat hdr_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkExtent2D extent_{};

    GeometryArena geometry_arena_{};
    MaterialStorage material_storage_{};
    ImageStorage image_storage_{};
    SamplerStorage sampler_storage_{};
    GpuResourceTable gpu_resource_table_{};

    std::vector<Buffer> ubos_;

    PipelineGraphRepository pipeline_graph_;
    PipelineNodeHandle depth_prepass_pipeline_;
    PipelineNodeHandle forward_pipeline_;
    PipelineNodeHandle composite_pipeline_;
    ShaderChangeQueue shader_change_queue_;

    std::vector<MeshSlot> meshes_;
    std::uint32_t mesh_free_head_ = 0;

    std::vector<ModelSlot> models_;
    std::uint32_t model_free_head_ = 0;

    std::unordered_map<std::size_t, ModelHandle> model_cache_;

    std::vector<Submission> submissions_;
    std::vector<ModelSubmission> model_submissions_;

    std::vector<RendererFrame> frames_;

    MaterialHandle default_material_handle_{};

    std::uint32_t maximum_draw_count_ = 0;
    std::uint32_t maximum_submission_count_ = 0;

    StageTimings last_frame_timings_{};

    std::queue<std::function<void()>> event_queue_;
    std::atomic_uint32_t queued_events_;
    std::mutex queue_mutex_;

    struct FrameTimestamps {
        VkQueryPool query_pool{VK_NULL_HANDLE};
        bool has_results{false};
    };
    std::vector<FrameTimestamps> timestamp_queries_;
    float timestamp_period_{1.0F};

    bool initialized_ = false;
    renderer::SlangCompiler compiler;
};

template<>
struct std::formatter<RendererErrorType> : std::formatter<std::string_view> {
    constexpr auto format(RendererErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case RendererErrorType::invalid_argument:
                    return "invalid_argument";
                case RendererErrorType::invalid_mesh:
                    return "invalid_mesh";
                case RendererErrorType::invalid_model:
                    return "invalid_model";
                case RendererErrorType::invalid_material:
                    return "invalid_material";
                case RendererErrorType::unsupported_index_type:
                    return "unsupported_index_type";
                case RendererErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case RendererErrorType::size_overflow:
                    return "size_overflow";
                case RendererErrorType::model_load_error:
                    return "model_load_error";
                case RendererErrorType::geometry_error:
                    return "geometry_error";
                case RendererErrorType::material_error:
                    return "material_error";
                case RendererErrorType::image_error:
                    return "image_error";
                case RendererErrorType::forward_target_error:
                    return "forward_target_error";
                case RendererErrorType::device_error:
                    return "device_error";
                case RendererErrorType::pipeline_error:
                    return "pipeline_error";
                case RendererErrorType::compiler_error:
                    return "compiler_error";
                case RendererErrorType::invalid_pipeline:
                    return "invalid_pipeline";
                case RendererErrorType::pipeline_storage_error:
                    return "pipeline_storage_error";
                case RendererErrorType::gpu_resource_table_error:
                    return "gpu_resource_table_error";
                case RendererErrorType::pipeline_graph_error:
                    return "pipeline_graph_error";
            }

            return "unknown_renderer_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};
