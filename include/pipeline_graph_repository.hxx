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

#include "error_context.hxx"

#include "forward.hxx"
#include "pipeline.hxx"
#include "pipeline_storage.hxx"
#include "shader_object.hxx"
#include "shader_object_storage.hxx"
#include "slang_compiler.hxx"

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

struct PipelineGraphCreateInfo {
    std::uint32_t pipeline_capacity = 0;
    std::uint32_t frames_in_flight = 1;

    VkDescriptorSetLayout global_descriptor_set_layout = VK_NULL_HANDLE;

    // Forwarded to PipelineStorageCreateInfo::cache_file_path. Empty means
    // no disk persistence for the VkPipelineCache.
    std::filesystem::path cache_file_path;

    std::string_view debug_name = "pipeline_graph";
};

struct PipelineRegisterInfo {
    std::vector<renderer::ShaderCompileRequest> stages;
    std::vector<VkDescriptorSetLayout> additional_descriptor_set_layouts{};
    std::vector<VkPushConstantRange> push_constant_ranges;
    std::vector<VkFormat> colour_formats;

    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    bool blending = false;

    // When true, register_pipeline() builds a ShaderObjectSet (via
    // ShaderObjectStorage) instead of a Pipeline for this node. See
    // docs/pipeline_to_shader_objects.md Phase 4 -- allows migrating one
    // registered pipeline at a time while the rest keep using VkPipeline.
    bool use_shader_objects = false;

    std::string debug_name;
};

/*
 * Owns pipeline storage plus a small DAG:
 *
 *   source_file -> shader_stage -> pipeline
 *
 * A changed source file marks every shader_stage that reads it dirty,
 * which marks every pipeline that owns one of those stages pending for
 * rebuild. process_dirty() recompiles only the dirty stages (shared
 * stages across pipelines recompile once) and rebuilds a VkPipeline
 * only once every stage it needs is clean. Old VkPipeline handles are
 * destroyed frames_in_flight frames after being replaced, never
 * immediately, since in-flight command buffers may still reference
 * them.
 */
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

    // Compiles every stage synchronously and builds the pipeline before
    // returning. Shared stages (same source_path + entry_point + stage,
    // already registered by another pipeline) reuse their existing
    // compiled SPIR-V instead of recompiling.
    [[nodiscard]]
    auto register_pipeline(renderer::SlangCompiler const &compiler, PipelineRegisterInfo register_info)
            -> std::expected<PipelineNodeHandle, PipelineGraphError>;

    // Batched equivalent of calling register_pipeline() once per entry, but
    // shared dirty stages compile exactly once across the whole batch, all
    // dirty stages compile concurrently on thread_pool, and all pipelines
    // build concurrently once compiling finishes. See
    // docs/parallel-pipeline.md Task 3. A compile failure anywhere in the
    // batch fails every reserved entry (rare -- usually a broken shared
    // shader file); a build failure only fails that one entry.
    [[nodiscard]]
    auto register_pipelines_parallel(renderer::SlangCompiler const &compiler,
                                     std::span<PipelineRegisterInfo> register_infos)
            -> std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>>;

    // Persists the VkPipelineCache to disk (see
    // PipelineGraphCreateInfo::cache_file_path). Logs and does nothing on
    // failure -- never treat this as fatal on a shutdown path.
    auto save_pipeline_cache() const -> void;

    [[nodiscard]]
    auto resolve(PipelineNodeHandle handle) const noexcept -> Pipeline const *;

    [[nodiscard]]
    auto pipeline_handle(PipelineNodeHandle handle) const noexcept -> PipelineHandle;

    // Valid only for nodes registered with use_shader_objects = true; returns
    // nullptr for a VkPipeline-backed node (or an unknown/stale handle).
    [[nodiscard]]
    auto resolve_shader_objects(PipelineNodeHandle handle) const noexcept -> ShaderObjectSet const *;

    [[nodiscard]]
    auto shader_object_handle(PipelineNodeHandle handle) const noexcept -> ShaderObjectHandle;

    // Call once per frame with whatever changed-file paths a watcher
    // collected since the last call. Cheap no-op if empty.
    auto on_files_changed(std::span<std::filesystem::path const> changed_files) -> void;

    // Call once per frame. Recompiles dirty stages and rebuilds any
    // pipeline whose stages are now all clean. Failed compiles log and
    // leave the live pipeline untouched.
    auto process_dirty(renderer::SlangCompiler const &compiler) -> void;

    // Call once per frame, before resolve() calls that will be recorded
    // into a new command buffer. Destroys pipelines that have been
    // retired for frames_in_flight frames.
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

        // Used to avoid retrying a known-broken shader every single
        // frame: only re-attempt once a *new* file change bumped
        // last_change_generation past our last failed attempt.
        std::uint64_t last_change_generation = 0;
        std::uint64_t last_attempt_generation = 0;
    };

    struct PipelineNode {
        std::vector<std::uint32_t> stage_indices;
        PipelineRegisterInfo register_info;
        PipelineHandle live_handle{};
        ShaderObjectHandle live_shader_object_handle{};

        std::uint32_t generation = 1;
        std::uint32_t next_free = 0;

        bool occupied = false;
        bool pending_rebuild = false;
    };

    struct RetiringPipeline {
        PipelineHandle handle;
        ShaderObjectHandle shader_object_handle;
        bool is_shader_object = false;
        std::uint32_t frames_remaining = 0;
    };

    struct BuiltNode {
        PipelineHandle handle;
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

    auto retire(PipelineHandle handle) -> void;
    auto retire(ShaderObjectHandle handle) -> void;

    [[nodiscard]]
    static auto to_vk_stage(renderer::ShaderStage stage) noexcept -> VkShaderStageFlagBits;

    PipelineStorage storage_;
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
