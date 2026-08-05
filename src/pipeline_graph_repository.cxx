#include "pipeline_graph_repository.hxx"

#include <algorithm>
#include <utility>

#include "error_describe.hxx"
#include "logger.hxx"

namespace {
    auto to_lookup_key(std::filesystem::path const &path) -> std::string {
        std::error_code error_code;
        auto canonical = std::filesystem::weakly_canonical(path, error_code);

        return (error_code ? path : canonical).generic_string();
    }

    auto to_stage_key(renderer::ShaderCompileRequest const &request) -> std::string {
        return to_lookup_key(request.source_path) + "|" + request.entry_point + "|" +
               std::to_string(static_cast<int>(request.stage));
    }
} // namespace

PipelineGraphRepository::~PipelineGraphRepository() { destroy(); }

PipelineGraphRepository::PipelineGraphRepository(PipelineGraphRepository &&other) noexcept :
    storage_(std::move(other.storage_)), source_files_(std::move(other.source_files_)),
    source_file_lookup_(std::move(other.source_file_lookup_)), stage_nodes_(std::move(other.stage_nodes_)),
    stage_lookup_(std::move(other.stage_lookup_)), pipeline_nodes_(std::move(other.pipeline_nodes_)),
    pipeline_free_head_(other.pipeline_free_head_), retiring_(std::move(other.retiring_)),
    frames_in_flight_(other.frames_in_flight_), change_generation_(other.change_generation_) {
    other.pipeline_free_head_ = 0;
    other.frames_in_flight_ = 1;
    other.change_generation_ = 0;
}

auto PipelineGraphRepository::operator=(PipelineGraphRepository &&other) noexcept -> PipelineGraphRepository & {
    if (this == &other) {
        return *this;
    }

    destroy();

    storage_ = std::move(other.storage_);
    source_files_ = std::move(other.source_files_);
    source_file_lookup_ = std::move(other.source_file_lookup_);
    stage_nodes_ = std::move(other.stage_nodes_);
    stage_lookup_ = std::move(other.stage_lookup_);
    pipeline_nodes_ = std::move(other.pipeline_nodes_);
    pipeline_free_head_ = other.pipeline_free_head_;
    retiring_ = std::move(other.retiring_);
    frames_in_flight_ = other.frames_in_flight_;
    change_generation_ = other.change_generation_;

    other.pipeline_free_head_ = 0;
    other.frames_in_flight_ = 1;
    other.change_generation_ = 0;

    return *this;
}

auto PipelineGraphRepository::create(VulkanContext &context, PipelineGraphCreateInfo const &create_info)
        -> std::expected<PipelineGraphRepository, PipelineGraphError> {
    if (create_info.pipeline_capacity < 1 || create_info.frames_in_flight == 0) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::invalid_argument,
        });
    }

    auto storage = PipelineStorage::create(
            context, PipelineStorageCreateInfo{
                             .capacity = create_info.pipeline_capacity,
                             .global_descriptor_set_layout = create_info.global_descriptor_set_layout,
                             .debug_name = create_info.debug_name,
                     });

    if (!storage) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::pipeline_storage_error,
                .cause = ErrorCause{Boxed<PipelineStorageError>{storage.error()}},
        });
    }

    PipelineGraphRepository repository;

    repository.storage_ = std::move(*storage);
    repository.frames_in_flight_ = create_info.frames_in_flight;

    repository.pipeline_nodes_.resize(create_info.pipeline_capacity);

    for (std::uint32_t index = 1; index < create_info.pipeline_capacity; ++index) {
        repository.pipeline_nodes_[index].next_free = index + 1 < create_info.pipeline_capacity ? index + 1 : 0;
    }

    repository.pipeline_free_head_ = create_info.pipeline_capacity > 1 ? 1 : 0;

    return repository;
}

auto PipelineGraphRepository::find_or_create_source_file(std::filesystem::path const &path) -> std::uint32_t {
    auto const key = to_lookup_key(path);

    auto const existing = source_file_lookup_.find(key);

    if (existing != source_file_lookup_.end()) {
        return existing->second;
    }

    auto const file_index = static_cast<std::uint32_t>(source_files_.size());

    std::error_code error_code;
    auto canonical = std::filesystem::weakly_canonical(path, error_code);

    source_files_.push_back(SourceFileNode{
            .path = error_code ? path : canonical,
            .dependent_stages = {},
    });

    source_file_lookup_.emplace(key, file_index);

    return file_index;
}

auto PipelineGraphRepository::link_stage_source_files(std::uint32_t stage_index) -> void {
    auto &stage = stage_nodes_[stage_index];

    auto const file_index = find_or_create_source_file(stage.request.source_path);

    stage.source_file_indices.push_back(file_index);

    auto &file = source_files_[file_index];

    if (std::ranges::find(file.dependent_stages, stage_index) == file.dependent_stages.end()) {
        file.dependent_stages.push_back(stage_index);
    }

    /*
     * This only watches the entry-point file itself, not #include'd
     * headers. Slang can report a module's file dependencies after a
     * successful compile (its include graph) -- once SlangCompiler
     * exposes that list on CompiledShader, feed each path through
     * find_or_create_source_file() here and link it to stage_index too,
     * so an edit to a shared .slangh invalidates every stage that
     * includes it.
     */
}

auto PipelineGraphRepository::find_or_create_stage(renderer::ShaderCompileRequest const &request,
                                                   std::uint32_t owning_pipeline) -> std::uint32_t {
    auto const key = to_stage_key(request);

    auto const existing = stage_lookup_.find(key);

    if (existing != stage_lookup_.end()) {
        auto const stage_index = existing->second;
        auto &stage = stage_nodes_[stage_index];

        if (std::ranges::find(stage.dependent_pipelines, owning_pipeline) == stage.dependent_pipelines.end()) {
            stage.dependent_pipelines.push_back(owning_pipeline);
        }

        return stage_index;
    }

    auto const stage_index = static_cast<std::uint32_t>(stage_nodes_.size());

    stage_nodes_.push_back(ShaderStageNode{
            .request = request,
            .spirv = {},
            .entry_point = request.entry_point,
            .source_file_indices = {},
            .dependent_pipelines = {owning_pipeline},
            .dirty = true,
            .has_compiled_once = false,
            .last_change_generation = change_generation_,
            .last_attempt_generation = 0,
    });

    stage_lookup_.emplace(key, stage_index);

    link_stage_source_files(stage_index);

    return stage_index;
}

auto PipelineGraphRepository::to_vk_stage(renderer::ShaderStage stage) noexcept -> VkShaderStageFlagBits {
    switch (stage) {
        case renderer::ShaderStage::vertex:
            return VK_SHADER_STAGE_VERTEX_BIT;

        case renderer::ShaderStage::fragment:
            return VK_SHADER_STAGE_FRAGMENT_BIT;

        case renderer::ShaderStage::compute:
            return VK_SHADER_STAGE_COMPUTE_BIT;

        case renderer::ShaderStage::task:
            return VK_SHADER_STAGE_TASK_BIT_EXT;

        case renderer::ShaderStage::mesh:
            return VK_SHADER_STAGE_MESH_BIT_EXT;
    }

    return VK_SHADER_STAGE_VERTEX_BIT;
}

auto PipelineGraphRepository::build_pipeline(PipelineNode const &node)
        -> std::expected<PipelineHandle, PipelineGraphError> {
    std::vector<ShaderStageInfo> stage_infos;
    stage_infos.reserve(node.stage_indices.size());

    for (auto const stage_index: node.stage_indices) {
        auto const &stage = stage_nodes_[stage_index];

        stage_infos.push_back(ShaderStageInfo{
                .stage = to_vk_stage(stage.request.stage),
                .spirv = stage.spirv,
                .entry_point = FlyString{stage.entry_point},
                .flags = 0,
                .specialization_info = nullptr,
        });
    }

    auto const is_compute =
            node.stage_indices.size() == 1 && stage_nodes_[node.stage_indices[0]].request.stage == renderer::ShaderStage::compute;

    auto created = is_compute
            ? storage_.create_compute(ComputePipelineCreateInfo{
                      .shader = stage_infos[0],
                      .additional_descriptor_set_layouts = node.register_info.additional_descriptor_set_layouts,
                      .push_constant_ranges = node.register_info.push_constant_ranges,
                      .debug_name = node.register_info.debug_name,
              })
            : storage_.create_graphics(GraphicsPipelineCreateInfo{
                      .shaders = stage_infos,
                      .additional_descriptor_set_layouts = node.register_info.additional_descriptor_set_layouts,
                      .push_constant_ranges = node.register_info.push_constant_ranges,
                      .dynamic_states = {},
                      .colour_formats = node.register_info.colour_formats,
                      .depth_format = node.register_info.depth_format,
                      .stencil_format = node.register_info.stencil_format,
                      .samples = node.register_info.samples,
                      .blending = node.register_info.blending,
                      .debug_name = node.register_info.debug_name,
              });

    if (!created) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::pipeline_storage_error,
                .cause = ErrorCause{Boxed<PipelineStorageError>{created.error()}},
        });
    }

    return *created;
}

auto PipelineGraphRepository::register_pipeline(renderer::SlangCompiler const &compiler,
                                                PipelineRegisterInfo register_info)
        -> std::expected<PipelineNodeHandle, PipelineGraphError> {
    if (register_info.stages.empty()) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::invalid_argument,
        });
    }

    if (pipeline_free_head_ == 0) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::capacity_exceeded,
        });
    }

    auto const node_index = pipeline_free_head_;
    auto &node = pipeline_nodes_[node_index];

    std::vector<std::uint32_t> stage_indices;
    stage_indices.reserve(register_info.stages.size());

    for (auto const &request: register_info.stages) {
        stage_indices.push_back(find_or_create_stage(request, node_index));
    }

    for (auto const stage_index: stage_indices) {
        auto &stage = stage_nodes_[stage_index];

        if (!stage.dirty) {
            continue; // already compiled by a prior register_pipeline() call sharing this stage
        }

        auto compiled = compiler.compile(stage.request);

        if (!compiled) {
            return std::unexpected(PipelineGraphError{
                    .type = PipelineGraphErrorType::compiler_error,
                    .cause = ErrorCause{Boxed<renderer::ShaderCompileError>{compiled.error()}},
            });
        }

        stage.spirv = std::move(compiled->spirv);
        stage.entry_point = compiled->entry_point;
        stage.dirty = false;
        stage.has_compiled_once = true;
    }

    node.stage_indices = std::move(stage_indices);
    node.register_info = std::move(register_info);
    node.pending_rebuild = false;
    node.occupied = true;

    auto built = build_pipeline(node);

    if (!built) {
        node.occupied = false;
        return std::unexpected(built.error());
    }

    node.live_handle = *built;

    pipeline_free_head_ = node.next_free;
    node.next_free = 0;

    return PipelineNodeHandle{
            .index = node_index,
            .generation = node.generation,
    };
}

auto PipelineGraphRepository::resolve(PipelineNodeHandle handle) const noexcept -> Pipeline const * {
    if (handle.index >= pipeline_nodes_.size()) {
        return nullptr;
    }

    auto const &node = pipeline_nodes_[handle.index];

    if (!node.occupied || node.generation != handle.generation) {
        return nullptr;
    }

    return storage_.get(node.live_handle);
}

auto PipelineGraphRepository::pipeline_handle(PipelineNodeHandle handle) const noexcept -> PipelineHandle {
    if (handle.index >= pipeline_nodes_.size()) {
        return {};
    }

    auto const &node = pipeline_nodes_[handle.index];

    if (!node.occupied || node.generation != handle.generation) {
        return {};
    }

    return node.live_handle;
}

auto PipelineGraphRepository::on_files_changed(std::span<std::filesystem::path const> changed_files) -> void {
    if (changed_files.empty()) {
        return;
    }

    ++change_generation_;

    for (auto const &path: changed_files) {
        auto const key = to_lookup_key(path);

        auto const existing = source_file_lookup_.find(key);

        if (existing == source_file_lookup_.end()) {
            debug("Not found for {}", key);
            continue; // not part of any registered pipeline
        }

        auto const &file = source_files_[existing->second];

        for (auto const stage_index: file.dependent_stages) {
            auto &stage = stage_nodes_[stage_index];

            stage.dirty = true;
            stage.last_change_generation = change_generation_;

            for (auto const pipeline_index: stage.dependent_pipelines) {
                pipeline_nodes_[pipeline_index].pending_rebuild = true;
            }
        }
    }
}

auto PipelineGraphRepository::process_dirty(renderer::SlangCompiler const &compiler) -> void {
    for (auto &node: pipeline_nodes_) {
        if (!node.occupied || !node.pending_rebuild) {
            continue;
        }

        auto all_stages_clean = true;

        for (auto const stage_index: node.stage_indices) {
            auto &stage = stage_nodes_[stage_index];

            if (!stage.dirty) {
                continue;
            }

            if (stage.has_compiled_once && stage.last_attempt_generation >= stage.last_change_generation) {
                all_stages_clean = false;
                continue;
            }

            stage.last_attempt_generation = change_generation_;

            auto compiled = compiler.compile(stage.request);

            if (!compiled) {
                error("Shader reload failed for {} ({}): {}", stage.request.source_path.string(),
                      stage.request.entry_point, describe(compiled.error()));

                all_stages_clean = false;
                continue;
            }

            stage.spirv = std::move(compiled->spirv);
            stage.entry_point = compiled->entry_point;
            stage.dirty = false;
            stage.has_compiled_once = true;
        }

        if (!all_stages_clean) {
            continue;
        }

        auto rebuilt = build_pipeline(node);

        if (!rebuilt) {
            error("Pipeline rebuild failed for {} after successful shader compile(s)", node.register_info.debug_name);
            continue; // stays pending_rebuild; will retry next process_dirty() call
        }

        retire(node.live_handle);

        node.live_handle = *rebuilt;
        node.pending_rebuild = false;

        debug("Recompiled {}", node.register_info.debug_name);
    }
}

auto PipelineGraphRepository::retire(PipelineHandle handle) -> void {
    if (!handle.valid()) {
        return;
    }

    retiring_.push_back(RetiringPipeline{
            .handle = handle,
            .frames_remaining = frames_in_flight_,
    });
}

auto PipelineGraphRepository::tick_retirement() -> void {
    for (auto &entry: retiring_) {
        if (entry.frames_remaining > 0) {
            --entry.frames_remaining;
        }
    }

    std::erase_if(retiring_, [this](RetiringPipeline const &entry) {
        if (entry.frames_remaining > 0) {
            return false;
        }

        auto const result = storage_.destroy_pipeline(entry.handle);

        if (!result) {
            error("Failed to destroy a retired pipeline");
        }

        return true;
    });
}

auto PipelineGraphRepository::destroy() noexcept -> void {
    retiring_.clear();

    pipeline_nodes_.clear();
    pipeline_free_head_ = 0;

    stage_nodes_.clear();
    stage_lookup_.clear();

    source_files_.clear();
    source_file_lookup_.clear();

    change_generation_ = 0;

    storage_.destroy();
}
