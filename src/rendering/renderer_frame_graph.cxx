#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include <array>
#include <expected>
#include <optional>

#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include "rendering/frame_graph/executor.hxx"
#include "rendering/frame_graph/pass_context.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/renderer.hxx"
#include "rendering/screenshot.hxx"

// The frame as a graph (docs/frame-graph.md, phase 4). What has not been migrated yet stays inside the "frame_legacy"
// pass, which runs first; the passes declared here follow it and derive their own barriers.

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    constexpr auto fragment_stage = static_cast<frame_graph::ShaderStages>(frame_graph::ShaderStage::fragment);
    constexpr auto compute_stage = static_cast<frame_graph::ShaderStages>(frame_graph::ShaderStage::compute);

    // The editor clears the swapchain under its UI to this.
    constexpr auto ui_clear_colour = VkClearValue{.color = {.float32 = {0.0F, 0.0F, 0.0F, 1.0F}}};

    // The state the legacy pass and the previous frame's readers leave a sampled image in.
    constexpr auto sampled_by_fragment_or_compute = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    };

    constexpr auto color_attachment_state = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    };

    // What the legacy pass leaves the depth buffer in for the forward pass's LOAD_OP_LOAD.
    constexpr auto depth_attachment_state = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    constexpr auto forward_clear_colour = VkClearValue{.color = {.float32 = {0.015F, 0.025F, 0.050F, 1.0F}}};

    // Buffers enter and leave the graph with nothing outstanding: whatever wrote them before the frame graph (the
    // prologue, the legacy pass) is ordered by the legacy pass's fences, and nothing outside the graph touches them
    // after it before the frame slot's fence.
    constexpr auto buffer_idle = frame_graph::ResourceState{};

    // The host reads the occlusion statistics back once the slot's fence has passed.
    constexpr auto host_reads = frame_graph::ResourceState{
            .stages = VK_PIPELINE_STAGE_2_HOST_BIT,
            .access = VK_ACCESS_2_HOST_READ_BIT,
    };

    auto physical_buffer(Buffer const &buffer) -> frame_graph::PhysicalBuffer {
        return frame_graph::PhysicalBuffer{
                .buffer = buffer.buffer,
                .address = buffer.device_address,
                .size = buffer.size(),
        };
    }

    auto physical_image(Image const &image) -> frame_graph::PhysicalImage {
        return frame_graph::PhysicalImage{
                .image = image.image(),
                .view = image.view(),
                .format = image.format(),
                .extent = image.extent(),
                .mip_levels = image.mip_levels(),
                .array_layers = image.array_layers(),
        };
    }

    // The Hi-Z pyramid as the occlusion tests (compute, task shaders) and the editor's debug view leave it.
    constexpr auto hiz_readers = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    };

    // What a viewport (or swapchain) image looks like to the graph: the sampled state the editor panel leaves it in.
    constexpr auto sampled_by_fragment = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    };
} // namespace

auto Renderer::record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError> {
    submit_batches_.clear();

    auto const &swapchain_image = info.swapchain_image;

    if (!initialized_ || info.command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || info.frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    // Reads back and resets this slot's pass timestamps: the slot's earlier work has finished.
    pass_profiler_.begin_slot(info.frame_index);

    screenshot_->try_resolve(info.frame_index);

    auto &frame = frames_[info.frame_index];
    consume_culled_readback(frame);

    auto const targets = resolve_frame_targets(frame);
    if (!targets) {
        return std::unexpected(targets.error());
    }

    // Registration changes made by overlay callbacks land after recording, so prepare, stages and timing all see the
    // same set. It spans every pass of the frame.
    auto const overlay_iteration = overlays_.iterate();

    // Fullscreen play composites into the swapchain with the UI on top. Otherwise the scene goes into the viewport
    // target the editor's Viewport panel samples and a second pass draws the UI onto the swapchain.
    auto const fullscreen = info.composite_target == CompositeTarget::swapchain;

    frame_graph_.reset();

    auto swapchain = frame_graph_.import_image({
            .entry = {.layout = VK_IMAGE_LAYOUT_UNDEFINED},
            .exit = {.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
            .swapchain = true,
            .debug_name = "swapchain",
            .image =
                    frame_graph::PhysicalImage{
                            .image = swapchain_image.image,
                            .view = swapchain_image.view,
                            .format = swapchain_image.format,
                            .extent = {swapchain_image.extent.width, swapchain_image.extent.height, 1},
                    },
    });

    auto viewport = frame_graph::ImageId{};
    if (!fullscreen) {
        viewport = frame_graph_.import_image({
                .entry = sampled_by_fragment,
                .exit = sampled_by_fragment,
                .debug_name = "viewport",
                .image =
                        frame_graph::PhysicalImage{
                                .image = targets->viewport->image(),
                                .view = targets->viewport->view(),
                                .format = targets->viewport->format(),
                                .extent = {targets->viewport->extent().width, targets->viewport->extent().height, 1},
                        },
        });
    }

    // Everything the legacy pass leaves for the passes after it enters the graph in the state it is left in: the
    // legacy pass declares nothing, and its fences order its work before theirs.
    auto const multisampled = targets->multisampled;
    auto const bloom_enabled = bloom_settings_.enabled;
    auto const ao_enabled = ao_settings_.enabled;

    // Single-sample HDR: forward resolves (or draws) into it, bloom and composition sample it.
    auto hdr_image = frame_graph_.import_image({
            .entry = sampled_by_fragment_or_compute,
            .exit = sampled_by_fragment_or_compute,
            .debug_name = "resolved_hdr",
            .image = physical_image(*targets->resolved_hdr),
    });

    auto msaa_hdr_image = frame_graph::ImageId{};
    if (multisampled) {
        msaa_hdr_image = frame_graph_.import_image({
                .entry = color_attachment_state,
                .exit = color_attachment_state,
                .debug_name = "hdr_msaa",
                .image = physical_image(*targets->hdr),
        });
    }

    auto depth_image = frame_graph_.import_image({
            .entry = depth_attachment_state,
            .exit = depth_attachment_state,
            .debug_name = "depth",
            .image = physical_image(*targets->depth),
    });

    // The single-sample depth AO samples: its own image under MSAA, the depth buffer itself otherwise. One graph
    // resource per image, so it must not be imported twice.
    auto resolved_depth_image = depth_image;
    if (multisampled) {
        resolved_depth_image = frame_graph_.import_image({
                .entry = depth_attachment_state,
                .exit = depth_attachment_state,
                .debug_name = "resolved_depth",
                .image = physical_image(*targets->resolved_depth),
        });
    }

    // Shadows are always rendered before the first frame's forward pass, so the atlas is sampled by then.
    auto const shadow_image = frame_graph_.import_image({
            .entry = sampled_by_fragment,
            .exit = sampled_by_fragment,
            .read_only = true,
            .debug_name = "shadow_atlas",
            .image = physical_image(*targets->shadow_atlas),
    });

    // With AO off forward samples a white texture that is not part of the graph. Both images are rebuilt every
    // frame; the raw one is only sampled by the denoise, the denoised one by forward.
    auto ao_raw_image = frame_graph::ImageId{};
    auto ao_image = frame_graph::ImageId{};
    if (ao_enabled) {
        ao_raw_image = frame_graph_.import_image({
                .entry = sampled_by_fragment_or_compute,
                .exit = sampled_by_fragment_or_compute,
                .debug_name = "ao_raw",
                .image = physical_image(*targets->ao_raw),
        });
        ao_image = frame_graph_.import_image({
                .entry = sampled_by_fragment,
                .exit = sampled_by_fragment,
                .debug_name = "ao_denoised",
                .image = physical_image(*targets->ao_denoised),
        });
    }

    // Rebuilt every frame by the bloom pass, which leaves it sampled. Not imported when bloom is off: nothing
    // touches it.
    auto bloom_image = frame_graph::ImageId{};
    if (bloom_enabled) {
        bloom_image = frame_graph_.import_image({
                .entry = sampled_by_fragment_or_compute,
                .exit = sampled_by_fragment_or_compute,
                .debug_name = "bloom",
                .image = physical_image(*frame.bloom_target.image),
        });
    }

    // The per-frame buffers the occlusion chain and the draws reach by device address. Every pass that dereferences one
    // must declare it, because neither sync validation nor the compiler can see a device-address access otherwise.
    // The ones nothing in the graph writes are imports marked read-only.
    using frame_graph::ShaderStage;
    constexpr auto geometry_stages = ShaderStage::vertex | ShaderStage::task | ShaderStage::mesh;
    constexpr auto draw_stages = geometry_stages | ShaderStage::fragment;

    auto const import_frame_buffer = [&](Buffer const &buffer, std::string_view name,
                                         bool read_only) -> frame_graph::BufferId {
        return frame_graph_.import_buffer({
                .entry = buffer_idle,
                .exit = buffer_idle,
                .read_only = read_only,
                .debug_name = name,
                .buffer = physical_buffer(buffer),
        });
    };

    // late_cs appends to the visible draws and transforms, so they are written when occlusion culling is on.
    auto visible_draws = import_frame_buffer(frame.visible_draw_buffer, "visible_draws", false);
    auto visible_transforms = import_frame_buffer(frame.visible_transform_buffer, "visible_transforms", false);
    auto const culled_indirect = import_frame_buffer(frame.culled_indirect_buffer, "culled_indirect", true);
    auto const frustum_planes = import_frame_buffer(frame.frustum_planes_buffer, "frustum_planes", true);

    auto const occlusion_active = frame.occlusion_active;
    auto const meshlet_occlusion_active = frame.meshlet_occlusion_active;

    // Phase 2 (late_cs): re-tests main_cs's candidates against this frame's Hi-Z. Everything it reads but does not
    // write was produced before the frame graph.
    auto late_indirect = frame_graph::BufferId{};
    auto merged_indirect = frame_graph::BufferId{};
    auto occlusion_views = frame_graph::BufferId{};
    auto source_draws = frame_graph::BufferId{};
    auto source_transforms = frame_graph::BufferId{};
    auto batch_bounds = frame_graph::BufferId{};
    auto source_indirect = frame_graph::BufferId{};
    auto occlusion_candidates = frame_graph::BufferId{};
    auto candidate_counts = frame_graph::BufferId{};
    auto hiz_image = frame_graph::ImageId{};
    if (occlusion_active) {
        late_indirect = import_frame_buffer(frame.late_indirect_buffer, "late_indirect", false);
        merged_indirect = import_frame_buffer(frame.merged_indirect_buffer, "merged_indirect", false);
        occlusion_views = import_frame_buffer(frame.occlusion_views_buffer, "occlusion_views", true);
        source_draws = import_frame_buffer(frame.draw_buffer, "draws", true);
        source_transforms = import_frame_buffer(frame.transform_buffer, "transforms", true);
        batch_bounds = import_frame_buffer(frame.batch_bounds_buffer, "batch_bounds", true);
        source_indirect = import_frame_buffer(frame.indirect_buffer, "indirect", true);
        occlusion_candidates = import_frame_buffer(frame.occlusion_candidates_buffer, "occlusion_candidates", true);
        candidate_counts = import_frame_buffer(frame.candidate_counts_buffer, "candidate_counts", true);

        // Rebuilt from the depth every frame; last frame's readers are what the build has to wait for.
        hiz_image = frame_graph_.import_image({
                .entry = hiz_readers,
                .exit = hiz_readers,
                .debug_name = "hiz",
                .image = physical_image(*hiz_.image),
        });
    }

    // Meshlet-level occlusion: the prepass phases' task shaders write the visibility bitset and the statistics
    // counters, forward replays the bitset.
    auto meshlet_bits = frame_graph::BufferId{};
    if (meshlet_occlusion_active) {
        meshlet_bits = import_frame_buffer(frame.meshlet_visibility_buffer, "meshlet_visibility", false);
    }

    auto stats_buffer = import_frame_buffer(frame.occlusion_stats_buffer, "occlusion_stats", false);
    auto stats_readback = frame_graph_.import_buffer({
            .entry = host_reads,
            .exit = host_reads,
            .debug_name = "occlusion_stats_readback",
            .buffer = physical_buffer(frame.occlusion_stats_readback_buffer),
    });

    // Shared by the record lambdas, which all run inside frame_graph::record() below.
    struct FrameState {
        std::expected<void, RendererError> result{};
        PassHandoff handoff;
    } state;

    frame_graph_.add_pass("frame_legacy", frame_graph::PassType::raster,
                          {
                                  .name_id = "frame_legacy",
                                  .label = "Frame (legacy)",
                                  .color = static_cast<std::uint32_t>(tracy::Color::RoyalBlue),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              // Touches nothing the graph knows about yet; the fences around it do the ordering.
                              pass.legacy();
                              pass.side_effect();

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  auto legacy_info = info;
                                  legacy_info.command_buffer = context.command_buffer;
                                  state.result = record_frame_legacy(legacy_info, frame, *targets, state.handoff);
                              }};
                          });

    // The Hi-Z pyramid: the single-sample depth reduced into a mip chain, one dispatch per level. The levels are
    // written and sampled one at a time inside the pass; the graph sees it enter writable and leave sampled.
    if (occlusion_active) {
        frame_graph_.add_pass("hiz_build", frame_graph::PassType::compute,
                              {
                                      .name_id = "hiz_build",
                                      .label = "Hi-Z build",
                                      .color = static_cast<std::uint32_t>(tracy::Color::DarkOrange),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const depth =
                                          pass.read(resolved_depth_image, frame_graph::Use::sampled, compute);
                                  hiz_image = pass.write(hiz_image, frame_graph::Use::storage_write, compute,
                                                         frame_graph::ExitUse{frame_graph::Use::sampled});

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context =
                                              make_pass_context(context.command_buffer, info.frame_index);
                                      if (auto const done = record_hiz_build(pass_context, *targets); !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });
    }

    // Phase 2 of occlusion culling, culling: re-tests the candidates main_cs deferred against this frame's Hi-Z and
    // appends the survivors to the visible draws, with the late and merged indirect commands.
    if (occlusion_active) {
        frame_graph_.add_pass(
                "late_cs", frame_graph::PassType::compute,
                {
                        .name_id = "occlusion_culling",
                        .label = "Occlusion culling",
                        .color = static_cast<std::uint32_t>(tracy::Color::SlateBlue),
                },
                [&](frame_graph::PassBuilder &pass) {
                    constexpr auto compute = stages_of(ShaderStage::compute);
                    using frame_graph::Use;

                    [[maybe_unused]] auto const hiz = pass.read(hiz_image, Use::sampled, compute);
                    [[maybe_unused]] auto const draws = pass.read(source_draws, Use::shader_read, compute);
                    [[maybe_unused]] auto const transforms = pass.read(source_transforms, Use::shader_read, compute);
                    [[maybe_unused]] auto const bounds = pass.read(batch_bounds, Use::shader_read, compute);
                    [[maybe_unused]] auto const commands = pass.read(source_indirect, Use::shader_read, compute);
                    [[maybe_unused]] auto const culled = pass.read(culled_indirect, Use::shader_read, compute);
                    [[maybe_unused]] auto const planes = pass.read(frustum_planes, Use::shader_read, compute);
                    [[maybe_unused]] auto const views = pass.read(occlusion_views, Use::shader_read, compute);
                    [[maybe_unused]] auto const candidates = pass.read(occlusion_candidates, Use::shader_read, compute);
                    [[maybe_unused]] auto const counts = pass.read(candidate_counts, Use::shader_read, compute);

                    // Appends past the ranges the early prepass reads, but device-address accesses are not tracked
                    // per range, so the early prepass's reads are ordered before these writes as a whole.
                    visible_draws = pass.write(visible_draws, Use::shader_read_write, compute);
                    visible_transforms = pass.write(visible_transforms, Use::shader_read_write, compute);
                    late_indirect = pass.write(late_indirect, Use::shader_write, compute);
                    merged_indirect = pass.write(merged_indirect, Use::shader_write, compute);
                    stats_buffer = pass.write(stats_buffer, Use::shader_read_write, compute);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = record_occlusion_cull_pass(pass_context, frame); !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

    // Phase 2 of occlusion culling: draws what late_cs added on top of the early prepass's depth, and leaves the final
    // single-sample depth.
    if (occlusion_active) {
        frame_graph_.add_pass(
                "depth_prepass_late", frame_graph::PassType::raster,
                {
                        .name_id = "depth_prepass_late",
                        .label = "Depth prepass (late)",
                        .color = static_cast<std::uint32_t>(tracy::Color::SlateGray),
                },
                [&](frame_graph::PassBuilder &pass) {
                    [[maybe_unused]] auto const draws =
                            pass.read(visible_draws, frame_graph::Use::shader_read, geometry_stages);
                    [[maybe_unused]] auto const transforms =
                            pass.read(visible_transforms, frame_graph::Use::shader_read, geometry_stages);
                    [[maybe_unused]] auto const commands = pass.read(late_indirect, frame_graph::Use::indirect_read);
                    [[maybe_unused]] auto const planes =
                            pass.read(frustum_planes, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                    if (meshlet_occlusion_active) {
                        [[maybe_unused]] auto const views =
                                pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                        meshlet_bits = pass.write(meshlet_bits, frame_graph::Use::shader_read_write,
                                                  stages_of(ShaderStage::task));
                        stats_buffer = pass.write(stats_buffer, frame_graph::Use::shader_read_write,
                                                  stages_of(ShaderStage::task));
                    }

                    depth_image = pass.write_depth(depth_image, frame_graph::LoadOp::load, frame_graph::StoreOp::store);
                    if (multisampled) {
                        resolved_depth_image =
                                pass.resolve(depth_image, resolved_depth_image, VK_RESOLVE_MODE_SAMPLE_ZERO_BIT);
                    } else {
                        resolved_depth_image = depth_image;
                    }
                    pass.render_area({.offset = {0, 0}, .extent = targets->extent});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = record_depth_prepass(pass_context, frame, *targets,
                                                                   render_pass::DepthPrepassPhase::late);
                            !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

    if (ao_enabled) {
        frame_graph_.add_pass(
                "gtao", frame_graph::PassType::compute,
                {
                        .name_id = "gtao",
                        .label = "GTAO",
                        .color = static_cast<std::uint32_t>(tracy::Color::DarkSlateGray),
                },
                [&](frame_graph::PassBuilder &pass) {
                    [[maybe_unused]] auto const depth =
                            pass.read(resolved_depth_image, frame_graph::Use::sampled, compute_stage);
                    ao_raw_image = pass.write(ao_raw_image, frame_graph::Use::storage_write, compute_stage);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = render_pass::gtao(
                                    pass_context, ambient_occlusion_info(frame, *targets, info.frame_index));
                            !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });

        frame_graph_.add_pass(
                "gtao_denoise", frame_graph::PassType::compute,
                {
                        .name_id = "gtao_denoise",
                        .label = "GTAO denoise",
                        .color = static_cast<std::uint32_t>(tracy::Color::SlateGray),
                },
                [&](frame_graph::PassBuilder &pass) {
                    [[maybe_unused]] auto const raw = pass.read(ao_raw_image, frame_graph::Use::sampled, compute_stage);
                    [[maybe_unused]] auto const depth =
                            pass.read(resolved_depth_image, frame_graph::Use::sampled, compute_stage);
                    ao_image = pass.write(ao_image, frame_graph::Use::storage_write, compute_stage,
                                          frame_graph::ExitUse{frame_graph::Use::sampled});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = render_pass::gtao_denoise(
                                    pass_context, ambient_occlusion_info(frame, *targets, info.frame_index));
                            !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

    frame_graph_.add_pass(
            "forward", frame_graph::PassType::raster,
            {
                    .name_id = "forward_pass",
                    .label = "Forward",
                    .color = static_cast<std::uint32_t>(tracy::Color::RoyalBlue),
            },
            [&](frame_graph::PassBuilder &pass) {
                // The buffers forward reads by device address. Declared so far: the occlusion chain's (draws,
                // transforms, indirect commands, culling planes, meshlet views and bits). Lights, cluster lists and
                // the UBO are still ordered by the legacy pass's fences; whichever pass takes over producing one must
                // declare it here too.
                [[maybe_unused]] auto const shadows =
                        pass.read(shadow_image, frame_graph::Use::sampled, fragment_stage);
                [[maybe_unused]] auto const draws =
                        pass.read(visible_draws, frame_graph::Use::shader_read, draw_stages);
                [[maybe_unused]] auto const transforms =
                        pass.read(visible_transforms, frame_graph::Use::shader_read, draw_stages);
                [[maybe_unused]] auto const commands = pass.read(occlusion_active ? merged_indirect : culled_indirect,
                                                                 frame_graph::Use::indirect_read);
                [[maybe_unused]] auto const planes =
                        pass.read(frustum_planes, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                if (meshlet_occlusion_active) {
                    [[maybe_unused]] auto const views =
                            pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                    [[maybe_unused]] auto const bits =
                            pass.read(meshlet_bits, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                }
                if (ao_enabled) {
                    [[maybe_unused]] auto const ao = pass.read(ao_image, frame_graph::Use::sampled, fragment_stage);
                }

                // Multisampled: draw into the MSAA target and resolve; the MSAA contents are not kept.
                if (multisampled) {
                    auto const msaa = pass.color(msaa_hdr_image, frame_graph::LoadOp::clear,
                                                 frame_graph::StoreOp::dont_care, forward_clear_colour);
                    hdr_image = pass.resolve(msaa, hdr_image, VK_RESOLVE_MODE_AVERAGE_BIT);
                } else {
                    hdr_image = pass.color(hdr_image, frame_graph::LoadOp::clear, frame_graph::StoreOp::store,
                                           forward_clear_colour);
                }
                depth_image = pass.write_depth(depth_image, frame_graph::LoadOp::load, frame_graph::StoreOp::store);
                pass.render_area({.offset = {0, 0}, .extent = targets->extent});

                return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                    if (!state.result) {
                        return;
                    }

                    auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);

                    OverlayScope const scene_scope{
                            .extent = targets->extent,
                            .colour_format = frame.forward_target.hdr_format(),
                            .depth_format = frame.forward_target.depth_format(),
                            .samples = samples_,
                    };
                    auto scene_overlays = [&] {
                        record_overlay_stage(pass_context, OverlayStage::scene, scene_scope, frame.view_projection);
                    };

                    auto const hdr = record_forward_pass(pass_context, frame, *targets, state.handoff.ao_texture_index,
                                                         render_pass::Callback::bind(scene_overlays));
                    if (hdr) {
                        state.handoff.hdr = *hdr;
                    } else {
                        state.result = std::unexpected(hdr.error());
                    }
                }};
            });

    if (bloom_enabled) {
        frame_graph_.add_pass(
                "bloom", frame_graph::PassType::compute,
                {
                        .name_id = "bloom",
                        .label = "Bloom",
                        .color = static_cast<std::uint32_t>(tracy::Color::Orange),
                },
                [&](frame_graph::PassBuilder &pass) {
                    [[maybe_unused]] auto const input = pass.read(hdr_image, frame_graph::Use::sampled, compute_stage);
                    // The mip chain is written, sampled and written again inside the pass, one level at a time; the
                    // graph only sees it enter writable and leave sampled.
                    bloom_image = pass.write(bloom_image, frame_graph::Use::storage_write, compute_stage,
                                             frame_graph::ExitUse{frame_graph::Use::sampled});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        auto const bloom_result = record_bloom_pass(pass_context, frame, *targets, state.handoff.hdr);
                        if (bloom_result) {
                            state.handoff.bloom = *bloom_result;
                        } else {
                            state.result = std::unexpected(bloom_result.error());
                        }
                    }};
                });
    }

    frame_graph_.add_pass(
            "composition", frame_graph::PassType::raster,
            {
                    .name_id = "composition",
                    .label = "Composition",
                    .color = static_cast<std::uint32_t>(tracy::Color::SeaGreen),
            },
            [&](frame_graph::PassBuilder &pass) {
                if (bloom_enabled) {
                    [[maybe_unused]] auto const bloom =
                            pass.read(bloom_image, frame_graph::Use::sampled, fragment_stage);
                }
                if (fullscreen) {
                    swapchain = pass.color(swapchain, frame_graph::LoadOp::dont_care, frame_graph::StoreOp::store);
                    pass.render_area({.offset = {0, 0}, .extent = swapchain_image.extent});
                } else {
                    viewport = pass.color(viewport, frame_graph::LoadOp::dont_care, frame_graph::StoreOp::store);
                    pass.render_area({.offset = {0, 0}, .extent = targets->extent});
                }

                return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                    if (!state.result) {
                        return;
                    }

                    auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);

                    OverlayScope const ui_scope{
                            .extent = swapchain_image.extent,
                            .colour_format = swapchain_image.format,
                            .depth_format = VK_FORMAT_UNDEFINED,
                            .samples = VK_SAMPLE_COUNT_1_BIT,
                    };
                    auto ui_overlays = [&] {
                        record_overlay_stage(pass_context, OverlayStage::ui, ui_scope, frame.view_projection);
                    };
                    auto const ui_callback = render_pass::Callback::bind(ui_overlays);

                    auto const composited = render_pass::composite(
                            pass_context,
                            render_pass::CompositePassInfo{
                                    .extent = fullscreen ? swapchain_image.extent : targets->extent,
                                    .hdr = state.handoff.hdr,
                                    .bloom = state.handoff.bloom,
                                    .bloom_fallback_texture_index = image_storage_.emissive().index,
                                    .linear_sampler_index = sampler_storage_.linear_clamp().index,
                                    .pipeline = composite_pipeline_,
                                    .exposure = 1.0F,
                                    .bloom_intensity = bloom_settings_.intensity,
                            },
                            fullscreen ? ui_callback : render_pass::Callback{});
                    if (!composited) {
                        state.result = std::unexpected(composited.error());
                    }
                }};
            });

    if (!fullscreen) {
        frame_graph_.add_pass(
                "ui", frame_graph::PassType::raster,
                {
                        .name_id = "ui",
                        .label = "UI",
                        .color = static_cast<std::uint32_t>(tracy::Color::Orchid),
                },
                [&](frame_graph::PassBuilder &pass) {
                    // The UI draws the viewport panel by sampling the target the composition pass just wrote.
                    [[maybe_unused]] auto const sampled =
                            pass.read(viewport, frame_graph::Use::sampled, fragment_stage);
                    swapchain = pass.color(swapchain, frame_graph::LoadOp::clear, frame_graph::StoreOp::store,
                                           ui_clear_colour);
                    pass.render_area({.offset = {0, 0}, .extent = swapchain_image.extent});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);

                        OverlayScope const ui_scope{
                                .extent = swapchain_image.extent,
                                .colour_format = swapchain_image.format,
                                .depth_format = VK_FORMAT_UNDEFINED,
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                        };
                        record_overlay_stage(pass_context, OverlayStage::ui, ui_scope, frame.view_projection);
                    }};
                });
    }

    // The host reads the stats a frame later; the copy and the host-visibility barrier are the graph's.
    frame_graph_.add_pass("occlusion_stats_readback", frame_graph::PassType::transfer,
                          {
                                  .name_id = "occlusion_stats_readback",
                                  .label = "Occlusion stats readback",
                                  .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              pass.side_effect();
                              [[maybe_unused]] auto const source =
                                      pass.read(stats_buffer, frame_graph::Use::transfer_read);
                              stats_readback = pass.write(stats_readback, frame_graph::Use::transfer_write);

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  record_occlusion_stats_readback(context.command_buffer, frame);
                              }};
                          });

    // A pending capture of the viewport target is only honoured in the editor; fullscreen takes the swapchain.
    if (auto const pending = screenshot_->pending_source(); pending.has_value()) {
        auto const from_viewport = *pending == ScreenshotSource::viewport && !fullscreen;

        frame_graph_.add_pass(
                "screenshot", frame_graph::PassType::transfer,
                {
                        .name_id = "screenshot",
                        .label = "Screenshot",
                        .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                },
                [&](frame_graph::PassBuilder &pass) {
                    pass.side_effect();
                    [[maybe_unused]] auto const source =
                            pass.read(from_viewport ? viewport : swapchain, frame_graph::Use::transfer_src);

                    return frame_graph::RecordFn{[&, from_viewport](frame_graph::PassContext &context) {
                        // The graph has moved the image to TRANSFER_SRC_OPTIMAL and moves it on afterwards.
                        auto const image = from_viewport
                                                   ? ScreenshotImage{.image = targets->viewport->image(),
                                                                     .format = targets->viewport->format(),
                                                                     .extent = {targets->viewport->extent().width,
                                                                                targets->viewport->extent().height},
                                                                     .managed_by_graph = true}
                                                   : ScreenshotImage{.image = swapchain_image.image,
                                                                     .format = swapchain_image.format,
                                                                     .extent = swapchain_image.extent,
                                                                     .managed_by_graph = true};
                        (void) screenshot_->record(context_, context.command_buffer, image, info.frame_index);
                    }};
                });
    }

    // Last, so the full-frame timestamp closes the frame.
    frame_graph_.add_pass("frame_end", frame_graph::PassType::transfer,
                          {
                                  .name_id = "frame_end",
                                  .label = "Frame end",
                                  .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              pass.side_effect();

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  record_frame_end(context.command_buffer, info.frame_index);

                                  TracyVkCollectHost(context_.host_query_context.context);
                                  if (context_.compute_host_query_context.context != nullptr) {
                                      TracyVkCollectHost(context_.compute_host_query_context.context);
                                  }
                              }};
                          });

    auto const compiled = plan_cache_.compile(frame_graph_, context_.queue_set.topology(),
                                              {.async_compute = context_.async_compute_mode != AsyncComputeMode::off,
                                               .serialize = context_.frame_graph_serialize});
    if (!compiled) {
        error("Could not compile the frame graph: {}", compiled.error());
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }
    frame_plan_ = *compiled;

    auto const resources = frame_graph::physical_resources_of(frame_graph_.description());

    auto const graphics_tracy = context_.host_query_context.context;
    auto const compute_tracy = context_.compute_queue != context_.graphics_queue
                                       ? context_.compute_host_query_context.context
                                       : graphics_tracy;

    auto batches = frame_graph::record(frame_graph::ExecuteInfo{
            .graph = frame_graph_.description(),
            .compiled = *frame_plan_,
            .records = frame_graph_.records(),
            .resources = resources,
            .queue_set = context_.queue_set,
            .profiler = pass_profiler_,
            .tracy_contexts = {graphics_tracy, compute_tracy},
            .prologue = info.command_buffer,
            .frame_index = info.frame_index,
    });
    if (!batches) {
        error("Could not record the frame graph: {}", batches.error().message);
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }
    submit_batches_ = std::move(*batches);

    return state.result;
}
