#include "pipeline_graph_repository.hxx"

#include <algorithm>
#include <utility>

#include "error_describe.hxx"
#include "logger.hxx"
#include "renderer.hxx"

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
    storage_(std::move(other.storage_)), shader_object_storage_(std::move(other.shader_object_storage_)),
    source_files_(std::move(other.source_files_)), source_file_lookup_(std::move(other.source_file_lookup_)),
    stage_nodes_(std::move(other.stage_nodes_)), stage_lookup_(std::move(other.stage_lookup_)),
    pipeline_nodes_(std::move(other.pipeline_nodes_)), pipeline_free_head_(other.pipeline_free_head_),
    retiring_(std::move(other.retiring_)), frames_in_flight_(other.frames_in_flight_),
    change_generation_(other.change_generation_) {
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
    shader_object_storage_ = std::move(other.shader_object_storage_);
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
                             .cache_file_path = create_info.cache_file_path,
                             .debug_name = create_info.debug_name,
                     });

    if (!storage) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::pipeline_storage_error,
                .cause = ErrorCause{Boxed<PipelineStorageError>{storage.error()}},
        });
    }

    auto shader_object_storage = ShaderObjectStorage::create(
            context, ShaderObjectStorageCreateInfo{
                             .capacity = create_info.pipeline_capacity,
                             .global_descriptor_set_layout = create_info.global_descriptor_set_layout,
                             .debug_name = create_info.debug_name,
                     });

    if (!shader_object_storage) {
        return std::unexpected(PipelineGraphError{
                .type = PipelineGraphErrorType::pipeline_storage_error,
                .cause = ErrorCause{Boxed<ShaderObjectStorageError>{shader_object_storage.error()}},
        });
    }

    PipelineGraphRepository repository;

    repository.storage_ = std::move(*storage);
    repository.shader_object_storage_ = std::move(*shader_object_storage);
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

auto PipelineGraphRepository::build_node(PipelineNode const &node) -> std::expected<BuiltNode, PipelineGraphError> {
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

    auto const is_compute = node.stage_indices.size() == 1 &&
                            stage_nodes_[node.stage_indices[0]].request.stage == renderer::ShaderStage::compute;

    if (node.register_info.use_shader_objects) {
        auto created = is_compute ? shader_object_storage_.create_compute(ComputeShaderCreateInfo{
                                            .shader = stage_infos[0],
                                            .additional_descriptor_set_layouts =
                                                    node.register_info.additional_descriptor_set_layouts,
                                            .push_constant_ranges = node.register_info.push_constant_ranges,
                                            .debug_name = node.register_info.debug_name,
                                    })
                                  : shader_object_storage_.create_linked(ShaderObjectCreateInfo{
                                            .shaders = stage_infos,
                                            .additional_descriptor_set_layouts =
                                                    node.register_info.additional_descriptor_set_layouts,
                                            .push_constant_ranges = node.register_info.push_constant_ranges,
                                            .debug_name = node.register_info.debug_name,
                                    });

        if (!created) {
            return std::unexpected(PipelineGraphError{
                    .type = PipelineGraphErrorType::pipeline_storage_error,
                    .cause = ErrorCause{Boxed<ShaderObjectStorageError>{created.error()}},
            });
        }

        return BuiltNode{.handle = {}, .shader_object_handle = *created};
    }

    auto created =
            is_compute
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
                              .dynamic_states = node.register_info.dynamic_states,
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

    return BuiltNode{.handle = *created, .shader_object_handle = {}};
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

    auto built = build_node(node);

    if (!built) {
        node.occupied = false;
        return std::unexpected(built.error());
    }

    node.live_handle = built->handle;
    node.live_shader_object_handle = built->shader_object_handle;

    pipeline_free_head_ = node.next_free;
    node.next_free = 0;

    return PipelineNodeHandle{
            .index = node_index,
            .generation = node.generation,
    };
}

auto PipelineGraphRepository::register_pipelines_parallel(renderer::SlangCompiler const &compiler,
                                                          std::span<PipelineRegisterInfo> register_infos)
        -> std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>> {

    std::vector<std::expected<PipelineNodeHandle, PipelineGraphError>> results(register_infos.size());

    std::vector<std::uint32_t> node_indices(register_infos.size(), 0);
    std::vector<std::vector<std::uint32_t>> stage_indices_per_entry(register_infos.size());
    std::vector<bool> reserved(register_infos.size(), false);

    // Phase 1 (sequential): find_or_create_stage/find_or_create_source_file
    // mutate shared lookup maps (stage_lookup_, source_file_lookup_), so
    // this whole phase must run single-threaded -- see
    // docs/parallel-pipeline.md Task 3.
    for (std::size_t i = 0; i < register_infos.size(); ++i) {
        auto const &info = register_infos[i];

        if (info.stages.empty()) {
            results[i] = std::unexpected(PipelineGraphError{
                    .type = PipelineGraphErrorType::invalid_argument,
            });
            continue;
        }

        if (pipeline_free_head_ == 0) {
            results[i] = std::unexpected(PipelineGraphError{
                    .type = PipelineGraphErrorType::capacity_exceeded,
            });
            continue;
        }

        auto const node_index = pipeline_free_head_;
        auto &node = pipeline_nodes_[node_index];

        pipeline_free_head_ = node.next_free;

        std::vector<std::uint32_t> stage_indices;
        stage_indices.reserve(info.stages.size());

        for (auto const &request: info.stages) {
            stage_indices.push_back(find_or_create_stage(request, node_index));
        }

        node_indices[i] = node_index;
        stage_indices_per_entry[i] = std::move(stage_indices);
        reserved[i] = true;
    }

    // Phase 2 (parallel): every distinct dirty stage across the batch
    // compiles exactly once, concurrently, on thread_pool.
    std::vector<std::uint32_t> dirty_stage_indices;
    {
        std::vector<bool> seen(stage_nodes_.size(), false);

        for (auto const &stage_indices: stage_indices_per_entry) {
            for (auto const stage_index: stage_indices) {
                if (seen[stage_index]) {
                    continue;
                }

                seen[stage_index] = true;

                if (stage_nodes_[stage_index].dirty) {
                    dirty_stage_indices.push_back(stage_index);
                }
            }
        }
    }


    auto &thread_pool = Renderer::thread_pool();

    if (!dirty_stage_indices.empty()) {
        std::vector<std::future<std::expected<renderer::CompiledShader, renderer::ShaderCompileError>>> futures;
        futures.reserve(dirty_stage_indices.size());

        for (auto const stage_index: dirty_stage_indices) {
            auto const &request = stage_nodes_[stage_index].request;


            futures.push_back(thread_pool.submit_task([&compiler, &request] { return compiler.compile(request); }));
        }


        std::optional<PipelineGraphError> first_error;

        for (std::size_t i = 0; i < dirty_stage_indices.size(); ++i) {
            auto compiled = futures[i].get();
            auto const stage_index = dirty_stage_indices[i];


            if (!compiled) {
                if (!first_error) {
                    first_error = PipelineGraphError{
                            .type = PipelineGraphErrorType::compiler_error,
                            .cause = ErrorCause{Boxed<renderer::ShaderCompileError>{compiled.error()}},
                    };
                }

                continue;
            }

            auto &stage = stage_nodes_[stage_index];

            stage.spirv = std::move(compiled->spirv);
            stage.entry_point = compiled->entry_point;
            stage.dirty = false;
            stage.has_compiled_once = true;
        }


        // A compile failure anywhere in the batch is all-or-nothing (unlike
        // a Phase 3 build failure below): it usually means a shared shader
        // file is broken, which affects every pipeline in the batch that
        // depends on it, so free every node this batch reserved and bail
        // before Phase 3.
        if (first_error) {
            for (std::size_t i = 0; i < register_infos.size(); ++i) {
                if (!reserved[i]) {
                    continue;
                }

                auto const node_index = node_indices[i];
                auto &node = pipeline_nodes_[node_index];

                node.next_free = pipeline_free_head_;
                pipeline_free_head_ = node_index;
                node.occupied = false;

                results[i] = std::unexpected(*first_error);
            }


            return results;
        }
    }

    // Phase 3 (parallel): build every successfully-compiled node's
    // VkPipeline/ShaderObjectSet concurrently. Safe now that
    // PipelineStorage/ShaderObjectStorage synchronize their own free-list
    // bookkeeping internally, and Pipeline::create_graphics/create_compute
    // synchronize the shared VkPipelineCache internally (see pipeline.cxx).
    for (std::size_t i = 0; i < register_infos.size(); ++i) {
        if (!reserved[i]) {
            continue;
        }

        auto const node_index = node_indices[i];
        auto &node = pipeline_nodes_[node_index];

        node.stage_indices = std::move(stage_indices_per_entry[i]);
        node.register_info = std::move(register_infos[i]);
        node.pending_rebuild = false;
        node.occupied = true;
    }

    std::vector<std::size_t> build_order;
    build_order.reserve(register_infos.size());

    for (std::size_t i = 0; i < register_infos.size(); ++i) {
        if (reserved[i]) {
            build_order.push_back(i);
        }
    }


    std::vector<std::future<std::expected<BuiltNode, PipelineGraphError>>> build_futures;
    build_futures.reserve(build_order.size());

    for (auto const i: build_order) {
        auto const node_index = node_indices[i];


        build_futures.push_back(
                thread_pool.submit_task([this, node_index] { return build_node(pipeline_nodes_[node_index]); }));
    }

    for (std::size_t k = 0; k < build_order.size(); ++k) {
        auto const i = build_order[k];
        auto const node_index = node_indices[i];
        auto &node = pipeline_nodes_[node_index];

        auto built = build_futures[k].get();


        if (!built) {
            node.occupied = false;
            node.next_free = pipeline_free_head_;
            pipeline_free_head_ = node_index;

            results[i] = std::unexpected(built.error());
            continue;
        }

        node.live_handle = built->handle;
        node.live_shader_object_handle = built->shader_object_handle;

        results[i] = PipelineNodeHandle{
                .index = node_index,
                .generation = node.generation,
        };
    }


    return results;
}

auto PipelineGraphRepository::save_pipeline_cache() const -> void {
    debug("[save_pipeline_cache] enter");

    auto const result = storage_.save_cache_to_disk();

    debug("[save_pipeline_cache] exit: {}", result ? "ok" : "FAILED");

    if (!result) {
        error("Failed to save pipeline cache to disk");
    }
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

auto PipelineGraphRepository::resolve_shader_objects(PipelineNodeHandle handle) const noexcept
        -> ShaderObjectSet const * {
    if (handle.index >= pipeline_nodes_.size()) {
        return nullptr;
    }

    auto const &node = pipeline_nodes_[handle.index];

    if (!node.occupied || node.generation != handle.generation || !node.register_info.use_shader_objects) {
        return nullptr;
    }

    return shader_object_storage_.get(node.live_shader_object_handle);
}

auto PipelineGraphRepository::shader_object_handle(PipelineNodeHandle handle) const noexcept -> ShaderObjectHandle {
    if (handle.index >= pipeline_nodes_.size()) {
        return {};
    }

    auto const &node = pipeline_nodes_[handle.index];

    if (!node.occupied || node.generation != handle.generation) {
        return {};
    }

    return node.live_shader_object_handle;
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

        auto rebuilt = build_node(node);

        if (!rebuilt) {
            error("Pipeline rebuild failed for {} after successful shader compile(s)", node.register_info.debug_name);
            continue; // stays pending_rebuild; will retry next process_dirty() call
        }

        if (node.register_info.use_shader_objects) {
            retire(node.live_shader_object_handle);
        } else {
            retire(node.live_handle);
        }

        node.live_handle = rebuilt->handle;
        node.live_shader_object_handle = rebuilt->shader_object_handle;
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
            .shader_object_handle = {},
            .is_shader_object = false,
            .frames_remaining = frames_in_flight_,
    });
}

auto PipelineGraphRepository::retire(ShaderObjectHandle handle) -> void {
    if (!handle.valid()) {
        return;
    }

    retiring_.push_back(RetiringPipeline{
            .handle = {},
            .shader_object_handle = handle,
            .is_shader_object = true,
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

        if (entry.is_shader_object) {
            auto const result = shader_object_storage_.destroy_shader_object(entry.shader_object_handle);

            if (!result) {
                error("Failed to destroy a retired shader object set");
            }
        } else {
            auto const result = storage_.destroy_pipeline(entry.handle);

            if (!result) {
                error("Failed to destroy a retired pipeline");
            }
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
    shader_object_storage_.destroy();
}
