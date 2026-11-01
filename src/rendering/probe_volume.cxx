#include "rendering/probe_volume.hxx"

#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <glm/geometric.hpp>

#include "core/logger.hxx"
#include "core/paths.hxx"
#include "gpu/context.hxx"
#include "gpu/shader_stage.hxx"
#include "rendering/renderer.hxx"

namespace {
    constexpr auto trace_slice = std::uint32_t{64};
    constexpr auto bytes_per_ray = VkDeviceSize{8};
    constexpr auto bytes_per_probe_state = VkDeviceSize{64};
    constexpr auto group = std::uint32_t{8};

    // Mirrors PushConstants in gi_probe_trace.slang.
    struct TracePush {
        VkDeviceAddress state = 0;
        VkDeviceAddress rays = 0;
        VkDeviceAddress instances = 0;
        VkDeviceAddress materials = 0;
        VkDeviceAddress ubo = 0;
        VkDeviceAddress emitters = 0;
        VkDeviceAddress lights = 0;
        std::int32_t base[3]{};
        std::uint32_t counts[3]{};
        float spacing = 1.0F;
        std::uint32_t rays_per_probe = 64;
        float rotation[4]{};
        std::uint32_t emitter_count = 0;
        std::uint32_t light_count = 0;
        std::uint32_t flags = 0;
        float emitter_intensity = 200.0F;
    };
    static_assert(sizeof(TracePush) == 120);
    static_assert(sizeof(TracePush) <= 128);

    // Mirrors Push in gi_probe_classify.slang.
    struct ClassifyPush {
        VkDeviceAddress state = 0;
        VkDeviceAddress rays = 0;
        std::int32_t base[3]{};
        std::uint32_t counts[3]{};
        float spacing = 1.0F;
        std::uint32_t rays_per_probe = 64;
        float back_face_limit = 0.25F;
        float rotation[4]{};
    };

    // Mirrors Push in gi_probe_blend.slang.
    struct BlendPush {
        VkDeviceAddress state = 0;
        VkDeviceAddress rays = 0;
        std::uint32_t atlas = 0;
        std::uint32_t counts[3]{};
        float spacing = 1.0F;
        std::uint32_t rays_per_probe = 64;
        float hysteresis = 0.97F;
        std::uint32_t tile = 10;
        float rotation[4]{};
    };

    auto hash(std::uint32_t value) -> std::uint32_t {
        value = value * 747796405U + 2891336453U;
        value = ((value >> ((value >> 28U) + 4U)) ^ value) * 277803737U;
        return (value >> 22U) ^ value;
    }

    auto unit_random(std::uint32_t seed) -> float {
        return static_cast<float>(hash(seed) & 0xFFFFFFU) / 16777216.0F;
    }

    // A uniformly distributed random rotation (Shoemake), a new one per frame so the rays sample the sphere evenly over
    // time.
    auto random_rotation(std::uint32_t frame) -> glm::quat {
        auto const u1 = unit_random(frame * 3U + 1U);
        auto const u2 = unit_random(frame * 3U + 2U) * glm::two_pi<float>();
        auto const u3 = unit_random(frame * 3U + 3U) * glm::two_pi<float>();
        auto const a = std::sqrt(1.0F - u1);
        auto const b = std::sqrt(u1);
        return glm::quat{b * std::cos(u3), a * std::sin(u2), a * std::cos(u2), b * std::sin(u3)};
    }

    auto groups_of(std::uint32_t threads) -> std::uint32_t { return (threads + group - 1U) / group; }
}

ProbeVolume::~ProbeVolume() { destroy(); }

auto ProbeVolume::irradiance_extent() const noexcept -> VkExtent2D {
    return {settings_.counts[0] * settings_.counts[1] * irradiance_tile, settings_.counts[2] * irradiance_tile};
}

auto ProbeVolume::distance_extent() const noexcept -> VkExtent2D {
    return {settings_.counts[0] * settings_.counts[1] * distance_tile, settings_.counts[2] * distance_tile};
}

auto ProbeVolume::irradiance_image() const -> Image const & { return *irradiance_; }

auto ProbeVolume::distance_image() const -> Image const & { return *distance_; }

auto ProbeVolume::create(Renderer &renderer) -> std::expected<void, RendererError> {
    if (!renderer.context().ray_query_supported || !renderer.scene_acceleration().ready()) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_argument});
    }

    auto const make = [&](char const *source, char const *entry,
                          char const *name) -> std::expected<PipelineNodeHandle, RendererError> {
        auto registered = renderer.register_pipeline(PipelineRegisterInfo{
                .stages = {renderer::ShaderCompileRequest{
                        .source_path = data_path(source),
                        .entry_point = FlyString{std::string_view{entry}},
                        .stage = renderer::ShaderStage::compute,
                        .include_directories = Paths::current().data_directories("assets/shaders"),
                }},
                .push_constant_ranges = {global_push_constant_range},
                .debug_name = name,
        });
        if (!registered) {
            return std::unexpected(registered.error());
        }
        return *registered;
    };
    auto trace = make("assets/shaders/gi_probe_trace.slang", "main_cs", "gi_probe_trace");
    auto classify = make("assets/shaders/gi_probe_classify.slang", "main_cs", "gi_probe_classify");
    auto blend_irradiance = make("assets/shaders/gi_probe_blend.slang", "irradiance_cs", "gi_probe_blend_irradiance");
    auto blend_distance = make("assets/shaders/gi_probe_blend.slang", "distance_cs", "gi_probe_blend_distance");
    auto border = make("assets/shaders/gi_probe_blend.slang", "border_cs", "gi_probe_border");
    if (!trace || !classify || !blend_irradiance || !blend_distance || !border) {
        return std::unexpected(RendererError{.type = RendererErrorType::invalid_pipeline});
    }
    trace_ = *trace;
    classify_ = *classify;
    blend_irradiance_ = *blend_irradiance;
    blend_distance_ = *blend_distance;
    border_ = *border;

    auto &context = renderer.context();
    auto const probes = probe_count();
    auto state = Buffer::create(context, BufferCreateInfo{
                                                 .size = probes * bytes_per_probe_state,
                                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 .memory = BufferMemory::device,
                                                 .debug_name = "gi_probe_state",
                                         });
    auto rays = Buffer::create(context, BufferCreateInfo{
                                                .size = VkDeviceSize{probes} * max_rays_per_probe * bytes_per_ray,
                                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                .memory = BufferMemory::device,
                                                .debug_name = "gi_ray_data",
                                        });
    if (!state || !rays) {
        error("ProbeVolume: could not allocate the probe buffers");
        return std::unexpected(RendererError{.type = RendererErrorType::device_error});
    }
    state_ = std::move(*state);
    rays_ = std::move(*rays);

    auto const atlas = [&](VkExtent2D extent, char const *name) {
        return create_held_image(
                renderer.image_storage(),
                ImageCreateInfo{
                        .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
                        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .image_type = VK_IMAGE_TYPE_2D,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                            image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                        .flags = 0,
                        .samples = VK_SAMPLE_COUNT_1_BIT,
                        .tiling = VK_IMAGE_TILING_OPTIMAL,
                        .mip_levels = 1,
                        .array_layers = 1,
                        .debug_name = name,
                });
    };
    auto irradiance = atlas(irradiance_extent(), "gi_irradiance_atlas");
    auto distance = atlas(distance_extent(), "gi_distance_atlas");
    if (!irradiance || !distance) {
        error("ProbeVolume: could not create the atlases");
        return std::unexpected(RendererError{.type = RendererErrorType::device_error});
    }
    irradiance_ = std::move(*irradiance);
    distance_ = std::move(*distance);

    renderer_ = &renderer;
    renderer.set_ray_tracing_scene_enabled(true);
    info("Probe volume ready: {}x{}x{} probes, {} rays each, spacing {} m", settings_.counts[0], settings_.counts[1],
         settings_.counts[2], settings_.rays_per_probe, settings_.spacing);
    return {};
}

auto ProbeVolume::destroy() noexcept -> void {
    irradiance_ = {};
    distance_ = {};
    state_.destroy();
    rays_.destroy();
    for (auto &buffer: emitter_buffers_) {
        buffer.destroy();
    }
    atlases_initialised_ = false;
    state_cleared_ = false;
    active_ = false;
    renderer_ = nullptr;
}

auto ProbeVolume::begin_frame(glm::vec3 camera, glm::vec3 sun_direction, glm::vec3 sun_radiance,
                              EnvironmentUboBlock const &environment) -> void {
    active_ = false;
    if (!ready()) {
        return;
    }
    settings_.rays_per_probe = std::clamp(settings_.rays_per_probe / trace_slice * trace_slice, trace_slice,
                                          max_rays_per_probe);
    settings_.spacing = std::max(settings_.spacing, 0.05F);

    // A volume that skips frames keeps its window too: the probes that scroll in are only reassigned by a classify pass.
    auto const interval = std::max(settings_.update_interval, 1U);
    updating_ = !state_cleared_ || frame_ % interval == 0;
    // The grid is centred on the camera and moves in whole probe steps.
    for (auto axis = 0; updating_ && axis < 3; ++axis) {
        auto const index = static_cast<std::size_t>(axis);
        auto const centre = static_cast<std::int32_t>(std::floor(camera[axis] / settings_.spacing));
        base_[index] = centre - static_cast<std::int32_t>(settings_.counts[index] / 2);
    }
    sun_direction_ = glm::normalize(sun_direction);
    sun_radiance_ = sun_radiance;
    environment_ = environment;
    rotation_ = random_rotation(frame_++);
    active_ = true;
    if (frame_ % 300 == 1) {
        info("Probe volume: {} explicit emitters", emitter_counts_[emitter_slot_]);
    }
}

auto ProbeVolume::begin_emitters(std::uint32_t frame_index) -> void {
    emitter_slot_ = frame_index % frames_in_flight;
    emitter_counts_[emitter_slot_] = 0;
    emitters_this_frame_.clear();
}

auto ProbeVolume::add_emitter(Emitter const &emitter) -> bool {
    if (!ready()) {
        return false;
    }
    auto &buffer = emitter_buffers_[emitter_slot_];
    if (!buffer.valid()) {
        auto created = Buffer::create(renderer_->context(),
                                      BufferCreateInfo{.size = VkDeviceSize{emitter_buffer_words} * sizeof(std::uint32_t),
                                                       .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       .memory = BufferMemory::upload,
                                                       .debug_name = "gi_emitters"});
        if (!created) {
            return false;
        }
        buffer = std::move(*created);
    }
    auto &count = emitter_counts_[emitter_slot_];
    if (count >= max_emitters) {
        return false;
    }
    std::memcpy(buffer.mapped_data() + VkDeviceSize{count} * sizeof(Emitter), &emitter, sizeof(Emitter));
    emitters_this_frame_.push_back(emitter);
    ++count;
    return true;
}

auto ProbeVolume::finish_emitters() -> void {
    auto &buffer = emitter_buffers_[emitter_slot_];
    if (!ready() || !buffer.valid() || emitters_this_frame_.empty()) {
        return;
    }
    // A bulb is a light that reaches max(80 radius, 1) m (the trace's falloff): the cell size is the largest reach, so
    // an emitter touches at most two cells per axis, and a hit finds every emitter that can light it in its own cell.
    auto reach = [](Emitter const &emitter) { return std::max(80.0F * emitter.radius, 1.0F); };
    auto low = glm::vec3{std::numeric_limits<float>::max()};
    auto high = glm::vec3{std::numeric_limits<float>::lowest()};
    auto cell = 4.0F;
    for (auto const &emitter: emitters_this_frame_) {
        auto const r = reach(emitter);
        low = glm::min(low, emitter.centre - glm::vec3{r});
        high = glm::max(high, emitter.centre + glm::vec3{r});
        cell = std::max(cell, r);
    }
    auto dims = glm::uvec3{1};
    for (;;) {
        dims = glm::max(glm::uvec3{glm::ceil((high - low) / cell)}, glm::uvec3{1});
        if (std::uint64_t{dims.x} * dims.y * dims.z <= max_grid_cells) {
            break;
        }
        cell *= 1.5F;
    }
    auto const cells = dims.x * dims.y * dims.z;
    auto const cell_of = [&](glm::ivec3 const &c) { return static_cast<std::uint32_t>(c.x) + dims.x * (static_cast<std::uint32_t>(c.y) + dims.y * static_cast<std::uint32_t>(c.z)); };
    auto const span_of = [&](Emitter const &emitter) {
        auto const r = glm::vec3{reach(emitter)};
        auto const first = glm::clamp(glm::ivec3{glm::floor((emitter.centre - r - low) / cell)}, glm::ivec3{0}, glm::ivec3{dims} - 1);
        auto const last = glm::clamp(glm::ivec3{glm::floor((emitter.centre + r - low) / cell)}, glm::ivec3{0}, glm::ivec3{dims} - 1);
        return std::pair{first, last};
    };

    std::vector<std::uint32_t> counts(cells, 0U);
    for (auto const &emitter: emitters_this_frame_) {
        auto const [first, last] = span_of(emitter);
        for (auto z = first.z; z <= last.z; ++z) {
            for (auto y = first.y; y <= last.y; ++y) {
                for (auto x = first.x; x <= last.x; ++x) {
                    ++counts[cell_of({x, y, z})];
                }
            }
        }
    }
    std::vector<std::uint32_t> first_index(cells, 0U);
    auto total = std::uint32_t{0};
    for (auto i = std::uint32_t{0}; i < cells; ++i) {
        first_index[i] = total;
        total += counts[i];
    }
    if (total > max_emitters * 8U) {
        return;
    }

    auto *words = reinterpret_cast<std::uint32_t *>(buffer.mapped_data());
    auto const put_float = [&](std::uint32_t at, float value) { std::memcpy(&words[at], &value, sizeof(float)); };
    put_float(grid_header_word + 0, low.x);
    put_float(grid_header_word + 1, low.y);
    put_float(grid_header_word + 2, low.z);
    put_float(grid_header_word + 3, 1.0F / cell);
    words[grid_header_word + 4] = dims.x;
    words[grid_header_word + 5] = dims.y;
    words[grid_header_word + 6] = dims.z;
    words[grid_header_word + 7] = cells;
    std::vector<std::uint32_t> fill(cells, 0U);
    for (auto i = std::uint32_t{0}; i < cells; ++i) {
        words[grid_ranges_word + i * 2] = first_index[i];
        words[grid_ranges_word + i * 2 + 1] = counts[i];
    }
    for (auto e = std::uint32_t{0}; e < emitters_this_frame_.size(); ++e) {
        auto const [first, last] = span_of(emitters_this_frame_[e]);
        for (auto z = first.z; z <= last.z; ++z) {
            for (auto y = first.y; y <= last.y; ++y) {
                for (auto x = first.x; x <= last.x; ++x) {
                    auto const index = cell_of({x, y, z});
                    words[grid_indices_word + first_index[index] + fill[index]++] = e;
                }
            }
        }
    }
}

auto ProbeVolume::gpu_block() const noexcept -> GpuBlock {
    if (!active_) {
        return {};
    }
    return GpuBlock{
            .probe_state = state_.device_address,
            .irradiance_texture = irradiance_.handle().index,
            .distance_texture = distance_.handle().index,
            .base_x = base_[0],
            .base_y = base_[1],
            .base_z = base_[2],
            .flags = 1U | (settings_.debug_view << 1U),
            .count_x = settings_.counts[0],
            .count_y = settings_.counts[1],
            .count_z = settings_.counts[2],
            .spacing = settings_.spacing,
            .intensity = settings_.intensity,
            .normal_bias = settings_.normal_bias,
            .view_bias = settings_.view_bias,
    };
}

auto ProbeVolume::bind(VkCommandBuffer command_buffer, std::uint32_t frame_index, PipelineNodeHandle node)
        -> VkPipelineLayout {
    auto const *set = renderer_->resolve_pipeline(node);
    if (set == nullptr || !set->valid()) {
        return VK_NULL_HANDLE;
    }
    set->bind(command_buffer);
    renderer_->resource_table().bind(command_buffer, frame_index, VK_PIPELINE_BIND_POINT_COMPUTE, set->layout());
    return set->layout();
}

auto ProbeVolume::record_trace(VkCommandBuffer command_buffer, std::uint32_t frame_index,
                               TraceInputs const &inputs) -> void {
    ZoneScopedNC("Probe trace", tracy::Color::Gold);

    // The state buffer is created with undefined contents; zero means "slot not assigned yet" to every shader.
    if (!state_cleared_) {
        vkCmdFillBuffer(command_buffer, state_.buffer, 0, VK_WHOLE_SIZE, 0);
        auto const barrier = VkMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        };
        auto const dependency = VkDependencyInfo{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(command_buffer, &dependency);
        state_cleared_ = true;
    }

    auto const layout = bind(command_buffer, frame_index, trace_);
    if (layout == VK_NULL_HANDLE) {
        return;
    }
    auto const &emitters = emitter_buffers_[frame_index % frames_in_flight];
    auto push = TracePush{
            .state = state_.device_address,
            .rays = rays_.device_address,
            .instances = inputs.instance_table,
            .materials = inputs.materials,
            .ubo = inputs.ubo,
            .emitters = emitters.valid() ? emitters.device_address : 0,
            .lights = inputs.lights,
            .base = {base_[0], base_[1], base_[2]},
            .counts = {settings_.counts[0], settings_.counts[1], settings_.counts[2]},
            .spacing = settings_.spacing,
            .rays_per_probe = settings_.rays_per_probe,
            .rotation = {rotation_.x, rotation_.y, rotation_.z, rotation_.w},
            .emitter_count = emitter_counts_[frame_index % frames_in_flight],
            .light_count = inputs.lights != 0 ? inputs.light_count : 0,
            .flags = settings_.cascade,
            .emitter_intensity = settings_.emitter_intensity,
    };
    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer, probe_count(), settings_.rays_per_probe / trace_slice, 1);
}

auto ProbeVolume::record_classify(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void {
    ZoneScopedNC("Probe classify", tracy::Color::Gold);
    auto const layout = bind(command_buffer, frame_index, classify_);
    if (layout == VK_NULL_HANDLE) {
        return;
    }
    auto const push = ClassifyPush{
            .state = state_.device_address,
            .rays = rays_.device_address,
            .base = {base_[0], base_[1], base_[2]},
            .counts = {settings_.counts[0], settings_.counts[1], settings_.counts[2]},
            .spacing = settings_.spacing,
            .rays_per_probe = settings_.rays_per_probe,
            .back_face_limit = settings_.back_face_limit,
            .rotation = {rotation_.x, rotation_.y, rotation_.z, rotation_.w},
    };
    vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    vkCmdDispatch(command_buffer, (probe_count() + 63U) / 64U, 1, 1);
}

auto ProbeVolume::record_blend(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void {
    ZoneScopedNC("Probe blend", tracy::Color::Gold);
    struct Target {
        PipelineNodeHandle node;
        Image const *image;
        VkExtent2D extent;
        std::uint32_t tile;
    };
    for (auto const &target: {Target{blend_irradiance_, irradiance_.get(), irradiance_extent(), irradiance_tile},
                              Target{blend_distance_, distance_.get(), distance_extent(), distance_tile}}) {
        auto const layout = bind(command_buffer, frame_index, target.node);
        if (layout == VK_NULL_HANDLE) {
            continue;
        }
        auto const push = BlendPush{
                .state = state_.device_address,
                .rays = rays_.device_address,
                .atlas = (target.tile == irradiance_tile ? irradiance_ : distance_).handle().index,
                .counts = {settings_.counts[0], settings_.counts[1], settings_.counts[2]},
                .spacing = settings_.spacing,
                .rays_per_probe = settings_.rays_per_probe,
                .hysteresis = settings_.hysteresis,
                .tile = target.tile,
                .rotation = {rotation_.x, rotation_.y, rotation_.z, rotation_.w},
        };
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
        vkCmdDispatch(command_buffer, probe_count(), 1, 1);
    }
}

auto ProbeVolume::record_border(VkCommandBuffer command_buffer, std::uint32_t frame_index) -> void {
    ZoneScopedNC("Probe border", tracy::Color::Gold);
    auto const layout = bind(command_buffer, frame_index, border_);
    if (layout == VK_NULL_HANDLE) {
        return;
    }
    struct Target {
        ImageHolder const *image;
        VkExtent2D extent;
        std::uint32_t tile;
    };
    for (auto const &target: {Target{&irradiance_, irradiance_extent(), irradiance_tile},
                              Target{&distance_, distance_extent(), distance_tile}}) {
        auto const push = BlendPush{
                .state = state_.device_address,
                .rays = rays_.device_address,
                .atlas = target.image->handle().index,
                .counts = {settings_.counts[0], settings_.counts[1], settings_.counts[2]},
                .spacing = settings_.spacing,
                .rays_per_probe = settings_.rays_per_probe,
                .hysteresis = settings_.hysteresis,
                .tile = target.tile,
                .rotation = {rotation_.x, rotation_.y, rotation_.z, rotation_.w},
        };
        vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
        vkCmdDispatch(command_buffer, groups_of(target.extent.width), groups_of(target.extent.height), 1);
        // The next target's pass uses another image: no barrier needed between them.
    }
}
