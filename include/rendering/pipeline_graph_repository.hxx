#pragma once

#include <BS_thread_pool.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <future>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/error_context.hxx"

#include "assets/slang_compiler.hxx"
#include "core/forward.hxx"
#include "gpu/shader_object_storage.hxx"
#include "gpu/shader_stage.hxx"

struct PipelineNodeHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return generation != 0;
    }

    auto operator==(PipelineNodeHandle const &) const -> bool = default;
};

enum class PipelineGraphErrorType : std::uint8_t {
    invalid_argument,
    invalid_handle,
    capacity_exceeded,
    compiler_error,
    pipeline_storage_error,
};

struct PipelineGraphError {
    PipelineGraphErrorType type = PipelineGraphErrorType::invalid_argument;

    std::optional<ErrorCause> cause;
};

template<>
struct std::formatter<PipelineGraphErrorType> : std::formatter<std::string_view> {
    constexpr auto format(PipelineGraphErrorType error, std::format_context &context) const {
        auto const name = [&]() constexpr -> std::string_view {
            switch (error) {
                case PipelineGraphErrorType::invalid_argument:
                    return "invalid_argument";
                case PipelineGraphErrorType::invalid_handle:
                    return "invalid_handle";
                case PipelineGraphErrorType::capacity_exceeded:
                    return "capacity_exceeded";
                case PipelineGraphErrorType::compiler_error:
                    return "compiler_error";
                case PipelineGraphErrorType::pipeline_storage_error:
                    return "pipeline_storage_error";
            }

            return "unknown_pipeline_graph_error";
        }();

        return std::formatter<std::string_view>::format(name, context);
    }
};

struct PrecompiledStage {
    renderer::ShaderCompileRequest request;
    renderer::CompiledShader compiled;
};

struct PrecompiledPipelineRegisterInfo {
    std::vector<PrecompiledStage> stages;
    std::vector<VkDescriptorSetLayout> additional_descriptor_set_layouts{};
    std::vector<VkPushConstantRange> push_constant_ranges;
    std::vector<VkFormat> colour_formats;
    std::vector<VkDynamicState> dynamic_states{};

    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    bool blending = false;

    std::string debug_name;
};

struct PipelineGraphCreateInfo {
    std::uint32_t pipeline_capacity = 0;
    std::uint32_t frames_in_flight = 1;

    VkDescriptorSetLayout global_descriptor_set_layout = VK_NULL_HANDLE;

    std::filesystem::path cache_file_path;
    std::filesystem::path shader_binary_cache_directory;

    std::string_view debug_name = "pipeline_graph";
};

struct PipelineRegisterInfo {
    std::vector<renderer::ShaderCompileRequest> stages;
    std::vector<VkDescriptorSetLayout> additional_descriptor_set_layouts{};
    std::vector<VkPushConstantRange> push_constant_ranges;
    std::vector<VkFormat> colour_formats;
    std::vector<VkDynamicState> dynamic_states{};

    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    bool blending = false;

    std::string debug_name;
};

// Owns pipeline storage and a small DAG:
//
//   source_file -> shader_stage -> pipeline
//
// A changed file dirties its stages, which marks their pipelines for rebuild. process_dirty() recompiles each
// dirty stage once and rebuilds a pipeline once all its stages are clean. Replaced pipelines are destroyed
// frames_in_flight frames later.
class PipelineGraphRepository {
public:
    PipelineGraphRepository() = default;
    ~PipelineGraphRepository();

    PipelineGraphRepository(PipelineGraphRepository const &) = delete;
    auto operator=(PipelineGraphRepository const &) -> PipelineGraphRepository & = delete;

    PipelineGraphRepository(PipelineGraphRepository &&other) noexcept;
    auto operator=(PipelineGraphRepository &&other) noexcept -> PipelineGraphRepository &;

    [[nodiscard]]
    static auto create(VulkanContext &context, PipelineGraphCreateInfo const &create_info)
            -> std::expected<PipelineGraphRepository, PipelineGraphError>;

    [[nodiscard]]
    auto register_pipeline(PipelineRegisterInfo register_info) -> std::expected<PipelineNodeHandle, PipelineGraphError>;

    [[nodiscard]]
    auto register_pipelines_parallel(std::span<PipelineRegisterInfo> register_infos)
            -> std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>>;

    [[nodiscard]]
    auto register_pipeline_precompiled(PrecompiledPipelineRegisterInfo register_info)
            -> std::expected<PipelineNodeHandle, PipelineGraphError>;

    [[nodiscard]]
    auto register_pipelines_precompiled_parallel(std::span<PrecompiledPipelineRegisterInfo> register_infos)
            -> std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>>;

    auto save_pipeline_cache() const -> void;

    [[nodiscard]]
    auto resolve_shader_objects(PipelineNodeHandle handle) const noexcept -> ShaderObjectSet const *;

    [[nodiscard]]
    auto shader_object_handle(PipelineNodeHandle handle) const noexcept -> ShaderObjectHandle;

    // Call once per frame with the paths changed since the last call.
    auto on_files_changed(std::span<std::filesystem::path const> changed_files) -> void;

    // Call once per frame. Failed compiles log and keep the live pipeline.
    auto process_dirty() -> void;

    // Call once per frame before recording resolve() results.
    auto tick_retirement() -> void;

    auto destroy() noexcept -> void;

private:
    struct SourceFileNode {
        std::filesystem::path path;
        std::vector<std::uint32_t> dependent_stages;
    };

    struct ShaderStageNode {
        renderer::ShaderCompileRequest request;
        std::vector<std::uint32_t> spirv;
        std::string entry_point;
        std::vector<std::uint32_t> source_file_indices;
        std::vector<std::uint32_t> dependent_pipelines;

        bool dirty = true;
        bool has_compiled_once = false;

        // Retry a broken shader only after a newer file change.
        std::uint64_t last_change_generation = 0;
        std::uint64_t last_attempt_generation = 0;
    };

    struct PipelineNode {
        std::vector<std::uint32_t> stage_indices;
        PipelineRegisterInfo register_info;
        ShaderObjectHandle live_shader_object_handle{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
        bool pending_rebuild = false;
    };

    struct RetiringPipeline {
        ShaderObjectHandle shader_object_handle;
        std::uint32_t frames_remaining = 0;
    };

    struct BuiltNode {
        ShaderObjectHandle shader_object_handle;
    };

    [[nodiscard]]
    auto find_or_create_source_file(std::filesystem::path const &path) -> std::uint32_t;

    [[nodiscard]]
    auto find_or_create_stage(renderer::ShaderCompileRequest const &request, std::uint32_t owning_pipeline)
            -> std::uint32_t;

    auto link_stage_source_files(std::uint32_t stage_index) -> void;

    [[nodiscard]]
    auto build_node(PipelineNode const &node) -> std::expected<BuiltNode, PipelineGraphError>;
    [[nodiscard]]
    auto build_shader_object_node(PipelineNode const &node) -> std::expected<ShaderObjectHandle, PipelineGraphError>;

    auto retire(ShaderObjectHandle handle) -> void;

    [[nodiscard]]
    static auto to_vk_stage(renderer::ShaderStage stage) noexcept -> VkShaderStageFlagBits;

    ShaderObjectStorage shader_object_storage_;

    std::vector<SourceFileNode> source_files_;
    std::unordered_map<std::string, std::uint32_t> source_file_lookup_;

    std::vector<ShaderStageNode> stage_nodes_;
    std::unordered_map<std::string, std::uint32_t> stage_lookup_;

    std::vector<PipelineNode> pipeline_nodes_;
    std::uint32_t pipeline_free_head_ = 0;

    std::vector<RetiringPipeline> retiring_;
    std::uint32_t frames_in_flight_ = 1;

    std::uint64_t change_generation_ = 0;
};
