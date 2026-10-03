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

// The frame as a graph (docs/frame-graph.md, phase 4): every pass is declared here, in recording order, and the
// compiler derives the barriers between them.

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

    // The state the previous frame's readers leave a sampled image in.
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

    // The depth buffer between the prepass that writes it and forward's LOAD_OP_LOAD.
    constexpr auto depth_attachment_state = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    constexpr auto forward_clear_colour = VkClearValue{.color = {.float32 = {0.015F, 0.025F, 0.050F, 1.0F}}};

    // Buffers enter and leave the graph with nothing outstanding: whatever wrote them before it (prepare_frame's
    // uploads, which end with their own barriers to every consumer stage, or the host, which needs none) is already
    // visible, and nothing outside the graph touches them after it before the frame slot's fence.
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

    // The targets, the shadow atlas and the AO images persist across frames. Most are written without loading first
    // (discarding what the previous frame left), so what matters about their entry state is the readers it must wait
    // for; the exit state is what the editor and the next frame find.
    auto const multisampled = targets->multisampled;
    auto const bloom_enabled = bloom_settings_.enabled;
    auto const ao_enabled = ao_settings_.enabled;
    auto const environment_pending = environment_.has_pending_record();

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

    // The shadow atlas persists across frames: only the cascades that moved are redrawn, so the pass loads it, and
    // forward samples it whether or not any was. Before the first shadow pass it has no contents (and no layout).
    auto shadow_image = frame_graph_.import_image({
            .entry = shadow_atlas_initialized_ ? sampled_by_fragment : frame_graph::ResourceState{},
            .exit = sampled_by_fragment,
            .debug_name = "shadow_atlas",
            .image = physical_image(*targets->shadow_atlas),
    });

    // With AO off forward samples a white texture that is not part of the graph. Otherwise both AO images are
    // transients created by the passes that write them (GTAO's raw image, the denoise's output); the allocator gives
    // them memory and bindless indices after the graph is compiled.
    auto ao_raw_image = frame_graph::ImageId{};
    auto ao_image = frame_graph::ImageId{};
    auto const ao_description = [&](std::string_view name) {
        return frame_graph::TransientImageDesc{
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent = {targets->extent.width, targets->extent.height, 1},
                .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                    image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                .debug_name = name,
        };
    };
    // The bindless index of a transient, once the allocator has made it.
    auto const transient_index = [&](frame_graph::ImageId image) {
        return transient_allocator_.handle(info.frame_index, image.index).index;
    };

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

    // main_cs always runs (it is plain frustum culling with occlusion off), so what it reads and writes is always
    // imported: the batch bounds, the occlusion views (disabled with occlusion off) and the candidate lists.
    auto const batch_bounds = import_frame_buffer(frame.batch_bounds_buffer, "batch_bounds", true);
    auto const occlusion_views = import_frame_buffer(frame.occlusion_views_buffer, "occlusion_views", true);
    auto occlusion_candidates = import_frame_buffer(frame.occlusion_candidates_buffer, "occlusion_candidates", false);
    auto candidate_counts = import_frame_buffer(frame.candidate_counts_buffer, "candidate_counts", false);

    // Clustered lighting: light_cull writes the visible lights, light_cluster the per-cluster lists and statistics
    // (the first bytes of the one buffer), which the host reads back a frame later.
    auto const clustered = clustered_lighting_;
    auto visible_lights = frame_graph::BufferId{};
    auto cluster_lights = frame_graph::BufferId{};
    auto cluster_stats_readback = frame_graph::BufferId{};
    if (clustered) {
        visible_lights = import_frame_buffer(frame.visible_lights_buffer, "visible_lights", false);
        cluster_lights = import_frame_buffer(frame.cluster_lights_buffer, "cluster_lights", false);
        cluster_stats_readback = frame_graph_.import_buffer({
                .entry = host_reads,
                .exit = host_reads,
                .debug_name = "cluster_stats_readback",
                .buffer = physical_buffer(frame.cluster_stats_readback_buffer),
        });
    }

    // late_cs appends to the visible draws and transforms, so they are written when occlusion culling is on.
    auto visible_draws = import_frame_buffer(frame.visible_draw_buffer, "visible_draws", false);
    auto visible_transforms = import_frame_buffer(frame.visible_transform_buffer, "visible_transforms", false);
    // Every caster, un-culled: the shadow pass draws these, and late_cs culls from them.
    auto const source_draws = import_frame_buffer(frame.draw_buffer, "draws", true);
    auto const source_transforms = import_frame_buffer(frame.transform_buffer, "transforms", true);
    auto const source_indirect = import_frame_buffer(frame.indirect_buffer, "indirect", true);
    auto culled_indirect = import_frame_buffer(frame.culled_indirect_buffer, "culled_indirect", false);
    auto const frustum_planes = import_frame_buffer(frame.frustum_planes_buffer, "frustum_planes", true);

    auto const occlusion_active = frame.occlusion_active;
    auto const meshlet_occlusion_active = frame.meshlet_occlusion_active;

    // Phase 2 (late_cs): re-tests main_cs's candidates against this frame's Hi-Z. Everything it reads but does not
    // write was produced before the frame graph.
    auto late_indirect = frame_graph::BufferId{};
    auto merged_indirect = frame_graph::BufferId{};
    auto hiz_image = frame_graph::ImageId{};
    if (occlusion_active) {
        late_indirect = import_frame_buffer(frame.late_indirect_buffer, "late_indirect", false);
        merged_indirect = import_frame_buffer(frame.merged_indirect_buffer, "merged_indirect", false);

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

    // CPU bookkeeping that used to sit in the legacy body.
    if (frame.occlusion_active) {
        // Next frame's phase 1 tests against this pyramid, projected as it was built.
        hiz_history_view_projection_ = frame.view_projection;
        hiz_history_valid_ = true;
    } else {
        // A pyramid from before this gap may not match what is on screen when culling resumes.
        hiz_history_valid_ = false;
    }

    // Overlays' prepare() hooks run before every pass, outside any rendering scope, and may write GPU data (debug
    // geometry, indirect arguments) the overlay draws read. The token orders them before those draws.
    auto overlay_data = frame_graph_.import_token("overlay_data", {}, {});
    frame_graph_.add_pass("overlay_prepare", frame_graph::PassType::compute,
                          {
                                  .name_id = "overlay_prepare",
                                  .label = "Overlay prepare",
                                  .color = static_cast<std::uint32_t>(tracy::Color::Orchid),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              pass.side_effect();
                              overlay_data = pass.write(overlay_data, frame_graph::Use::token_write);

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                                  record_overlay_prepares(pass_context);
                              }};
                          });

    // Temporary (deleted with RenderStage in phase 7): every stage of the old timings panel needs both of its
    // timestamps every frame, so the stages whose pass is not part of this frame's graph write them empty.
    frame_graph_.add_pass("stage_timestamps", frame_graph::PassType::compute,
                          {
                                  .name_id = "stage_timestamps",
                                  .label = "Stage timestamps",
                                  .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              pass.side_effect();

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  auto const empty = [&](RenderStage stage) {
                                      write_empty_stage(context.command_buffer, info.frame_index, stage);
                                  };

                                  if (!environment_pending) {
                                      empty(RenderStage::Environment);
                                  }
                                  if (frame.shadow_update_mask == 0) {
                                      empty(RenderStage::ShadowPass);
                                  }
                                  if (!occlusion_active) {
                                      empty(RenderStage::HiZBuild);
                                      empty(RenderStage::OcclusionCulling);
                                      empty(RenderStage::DepthPrepassLate);
                                  }
                                  if (!clustered) {
                                      empty(RenderStage::LightClustering);
                                  }
                                  if (!ao_enabled) {
                                      empty(RenderStage::AmbientOcclusion);
                                  }
                                  if (!bloom_enabled) {
                                      empty(RenderStage::BloomPass);
                                  }
                              }};
                          });

    // Image-based lighting and the procedural sky: compute that (re)builds the radiance cube, the prefiltered specular
    // cubes, the BRDF LUT and the SH coefficients, only on the frames the system planned work. It manages its own
    // layouts per mip and face, and its rebuilds are amortized over frames (a partly filled set must survive between
    // them), so the graph does not own those images: a token orders the build before the pass that samples the result.
    auto environment_token = frame_graph::BufferId{};
    if (environment_pending) {
        environment_token = frame_graph_.import_token("environment", {}, {});

        frame_graph_.add_pass("environment", frame_graph::PassType::compute,
                              {
                                      .name_id = "environment",
                                      .label = "Environment",
                                      .color = static_cast<std::uint32_t>(tracy::Color::SkyBlue),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  environment_token = pass.write(environment_token, frame_graph::Use::token_write);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      auto const pass_context =
                                              make_pass_context(context.command_buffer, info.frame_index);
                                      record_environment_pass(pass_context, frame);
                                  }};
                              });
    }

    // The statistics and the meshlet bitset are accumulated into by the culling and prepass shaders, so they start
    // empty.
    frame_graph_.add_pass("occlusion_stats_clear", frame_graph::PassType::transfer,
                          {
                                  .name_id = "occlusion_stats_clear",
                                  .label = "Occlusion stats clear",
                                  .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              stats_buffer = pass.write_discard(stats_buffer, frame_graph::Use::transfer_write);

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  record_occlusion_stats_clear(context.command_buffer, frame);
                              }};
                          });

    if (meshlet_occlusion_active) {
        frame_graph_.add_pass("meshlet_visibility_clear", frame_graph::PassType::transfer,
                              {
                                      .name_id = "meshlet_visibility_clear",
                                      .label = "Meshlet visibility clear",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  meshlet_bits = pass.write_discard(meshlet_bits, frame_graph::Use::transfer_write);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      record_meshlet_visibility_clear(context.command_buffer, frame);
                                  }};
                              });
    }

    // main_cs: frustum culling of every batch and, with occlusion culling on, phase 1 against last frame's Hi-Z (the
    // instances it defers become late_cs's candidates).
    frame_graph_.add_pass("gpu_culling", frame_graph::PassType::compute,
                          {
                                  .name_id = "culling",
                                  .label = "Culling",
                                  .color = static_cast<std::uint32_t>(tracy::Color::SlateBlue),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              constexpr auto compute = stages_of(ShaderStage::compute);
                              using frame_graph::Use;

                              [[maybe_unused]] auto const draws = pass.read(source_draws, Use::shader_read, compute);
                              [[maybe_unused]] auto const transforms =
                                      pass.read(source_transforms, Use::shader_read, compute);
                              [[maybe_unused]] auto const bounds = pass.read(batch_bounds, Use::shader_read, compute);
                              [[maybe_unused]] auto const commands =
                                      pass.read(source_indirect, Use::shader_read, compute);
                              [[maybe_unused]] auto const planes = pass.read(frustum_planes, Use::shader_read, compute);
                              [[maybe_unused]] auto const views = pass.read(occlusion_views, Use::shader_read, compute);
                              if (occlusion_active) {
                                  // View [0] holds last frame's pyramid, sampled by the occlusion test.
                                  [[maybe_unused]] auto const history = pass.read(hiz_image, Use::sampled, compute);
                              }

                              visible_draws = pass.write(visible_draws, Use::shader_write, compute);
                              visible_transforms = pass.write(visible_transforms, Use::shader_write, compute);
                              culled_indirect = pass.write(culled_indirect, Use::shader_write, compute);
                              occlusion_candidates = pass.write(occlusion_candidates, Use::shader_write, compute);
                              candidate_counts = pass.write(candidate_counts, Use::shader_write, compute);
                              stats_buffer = pass.write(stats_buffer, Use::shader_read_write, compute);

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                                  if (auto const done = record_gpu_culling(pass_context, frame); !done) {
                                      state.result = std::unexpected(done.error());
                                  }
                              }};
                          });

    // Clustered lighting: cull the lights to the frustum, then bin them into the screen-space clusters forward reads.
    if (clustered) {
        frame_graph_.add_pass("cluster_stats_clear", frame_graph::PassType::transfer,
                              {
                                      .name_id = "cluster_stats_clear",
                                      .label = "Cluster stats clear",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  cluster_lights = pass.write(cluster_lights, frame_graph::Use::transfer_write);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      auto const pass_context =
                                              make_pass_context(context.command_buffer, info.frame_index);
                                      record_cluster_stats_clear(pass_context, frame);
                                  }};
                              });

        frame_graph_.add_pass("light_cull", frame_graph::PassType::compute,
                              {
                                      .name_id = "light_cull",
                                      .label = "Light cull",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const planes =
                                          pass.read(frustum_planes, frame_graph::Use::shader_read, compute);
                                  visible_lights = pass.write(visible_lights, frame_graph::Use::shader_write, compute);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context =
                                              make_pass_context(context.command_buffer, info.frame_index);
                                      if (auto const done = record_light_cull(pass_context, frame); !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });

        frame_graph_.add_pass("light_cluster", frame_graph::PassType::compute,
                              {
                                      .name_id = "light_cluster",
                                      .label = "Light cluster",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const lights =
                                          pass.read(visible_lights, frame_graph::Use::shader_read, compute);
                                  cluster_lights =
                                          pass.write(cluster_lights, frame_graph::Use::shader_read_write, compute);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context =
                                              make_pass_context(context.command_buffer, info.frame_index);
                                      if (auto const done = record_light_cluster(pass_context, frame); !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });

        frame_graph_.add_pass(
                "cluster_stats_readback", frame_graph::PassType::transfer,
                {
                        .name_id = "cluster_stats_readback",
                        .label = "Cluster stats readback",
                        .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                },
                [&](frame_graph::PassBuilder &pass) {
                    pass.side_effect();
                    [[maybe_unused]] auto const source = pass.read(cluster_lights, frame_graph::Use::transfer_read);
                    cluster_stats_readback = pass.write(cluster_stats_readback, frame_graph::Use::transfer_write);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        record_cluster_stats_readback(pass_context, frame);
                    }};
                });
    }

    // Cascaded shadow maps into the atlas. Only the cascades in the update mask are cleared and redrawn; the rest keep
    // their contents, so the pass loads the atlas once it has any.
    if (frame.shadow_update_mask != 0) {
        frame_graph_.add_pass(
                "shadow_pass", frame_graph::PassType::raster,
                {
                        .name_id = "shadow_pass",
                        .label = "Shadows",
                        .color = static_cast<std::uint32_t>(tracy::Color::Purple),
                },
                [&](frame_graph::PassBuilder &pass) {
                    [[maybe_unused]] auto const draws =
                            pass.read(source_draws, frame_graph::Use::shader_read, draw_stages);
                    [[maybe_unused]] auto const transforms =
                            pass.read(source_transforms, frame_graph::Use::shader_read, draw_stages);
                    [[maybe_unused]] auto const commands = pass.read(source_indirect, frame_graph::Use::indirect_read);
                    [[maybe_unused]] auto const planes =
                            pass.read(frustum_planes, frame_graph::Use::shader_read, stages_of(ShaderStage::task));

                    shadow_image = pass.write_depth(shadow_image,
                                                    shadow_atlas_initialized_ ? frame_graph::LoadOp::load
                                                                              : frame_graph::LoadOp::dont_care,
                                                    frame_graph::StoreOp::store);
                    pass.render_area({.offset = {0, 0}, .extent = {shadow_atlas_width, shadow_atlas_height}});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = record_shadow_pass(pass_context, frame); !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

    // The depth prepass (phase 1 with occlusion culling, the only phase without): clears the depth buffer and draws
    // what main_cs kept. Under MSAA it resolves into the single-sample depth; MIN keeps each pixel's farthest sample,
    // which the Hi-Z needs to stay conservative (reverse-Z), otherwise SAMPLE_ZERO is what the rest expects.
    frame_graph_.add_pass(
            "depth_prepass", frame_graph::PassType::raster,
            {
                    .name_id = "depth_prepass",
                    .label = occlusion_active ? "Depth prepass (early)" : "Depth prepass",
                    .color = static_cast<std::uint32_t>(tracy::Color::SlateGray),
            },
            [&](frame_graph::PassBuilder &pass) {
                [[maybe_unused]] auto const draws =
                        pass.read(visible_draws, frame_graph::Use::shader_read, geometry_stages);
                [[maybe_unused]] auto const transforms =
                        pass.read(visible_transforms, frame_graph::Use::shader_read, geometry_stages);
                [[maybe_unused]] auto const commands = pass.read(culled_indirect, frame_graph::Use::indirect_read);
                [[maybe_unused]] auto const planes =
                        pass.read(frustum_planes, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                if (meshlet_occlusion_active) {
                    // View [0], the history Hi-Z's (sampled by the task shaders' meshlet test), and the bits and
                    // counters this phase's task shaders record.
                    [[maybe_unused]] auto const views =
                            pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                    [[maybe_unused]] auto const history =
                            pass.read(hiz_image, frame_graph::Use::sampled, stages_of(ShaderStage::task));
                    meshlet_bits =
                            pass.write(meshlet_bits, frame_graph::Use::shader_read_write, stages_of(ShaderStage::task));
                    stats_buffer =
                            pass.write(stats_buffer, frame_graph::Use::shader_read_write, stages_of(ShaderStage::task));
                }

                depth_image = pass.write_depth(depth_image, frame_graph::LoadOp::clear, frame_graph::StoreOp::store);
                if (multisampled) {
                    resolved_depth_image =
                            pass.resolve(depth_image, resolved_depth_image,
                                         occlusion_active ? VK_RESOLVE_MODE_MIN_BIT : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT);
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
                                                               occlusion_active ? render_pass::DepthPrepassPhase::early
                                                                                : render_pass::DepthPrepassPhase::only);
                        !done) {
                        state.result = std::unexpected(done.error());
                    }
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
                        // View [1], this frame's pyramid, sampled by the task shaders' meshlet test.
                        [[maybe_unused]] auto const views =
                                pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                        [[maybe_unused]] auto const pyramid =
                                pass.read(hiz_image, frame_graph::Use::sampled, stages_of(ShaderStage::task));
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
                    ao_raw_image = pass.create(ao_description("ao_raw"));
                    ao_raw_image = pass.write(ao_raw_image, frame_graph::Use::storage_write, compute_stage);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = render_pass::gtao(pass_context,
                                                                ambient_occlusion_info(*targets, info.frame_index,
                                                                                       transient_index(ao_raw_image),
                                                                                       transient_index(ao_image)));
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
                    ao_image = pass.create(ao_description("ao_denoised"));
                    ao_image = pass.write(ao_image, frame_graph::Use::storage_write, compute_stage,
                                          frame_graph::ExitUse{frame_graph::Use::sampled});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = make_pass_context(context.command_buffer, info.frame_index);
                        if (auto const done = render_pass::gtao_denoise(
                                    pass_context,
                                    ambient_occlusion_info(*targets, info.frame_index, transient_index(ao_raw_image),
                                                           transient_index(ao_image)));
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
                // The scene overlays draw inside this pass, from what their prepare() hooks wrote.
                [[maybe_unused]] auto const overlays = pass.read(overlay_data, frame_graph::Use::token_read);
                if (environment_pending) {
                    // Samples the cubes, the LUT and the SH the environment pass just built.
                    [[maybe_unused]] auto const environment =
                            pass.read(environment_token, frame_graph::Use::token_read);
                }
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
                if (clustered) {
                    // The per-cluster light lists (past the statistics) the fragment shader walks.
                    [[maybe_unused]] auto const clusters =
                            pass.read(cluster_lights, frame_graph::Use::shader_read, fragment_stage);
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

                    // The denoised AO image, or white when AO is off.
                    state.handoff.ao_texture_index =
                            ao_enabled ? transient_index(ao_image) : image_storage_.white().index;

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
                if (fullscreen) {
                    // The UI overlays draw inside this pass in fullscreen play.
                    [[maybe_unused]] auto const overlays = pass.read(overlay_data, frame_graph::Use::token_read);
                }
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
                    // The UI draws the viewport panel by sampling the target the composition pass just wrote. ImGui
                    // may only sample imports in SHADER_READ_ONLY (the viewport, the Hi-Z debug view) or images outside
                    // the graph; see docs/frame-graph-status.md before making any ImGui-visible image a transient.
                    [[maybe_unused]] auto const sampled =
                            pass.read(viewport, frame_graph::Use::sampled, fragment_stage);
                    [[maybe_unused]] auto const overlays = pass.read(overlay_data, frame_graph::Use::token_read);
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

    // Memory and images for the transients, recreated only when the compiled plan or a description changed. New
    // images have new bindless slots, which this frame's descriptor set has not seen yet.
    auto const allocated = transient_allocator_.prepare(info.frame_index, frame_graph_.description(), *frame_plan_,
                                                        transient_aliasing_);
    if (!allocated) {
        error("Could not allocate the frame graph's transients: {}", allocated.error().message);
        return std::unexpected(make_error(RendererErrorType::image_error));
    }
    if (*allocated) {
        if (auto const refreshed =
                    gpu_resource_table_.prepare_frame(info.frame_index, image_storage_, sampler_storage_);
            !refreshed) {
            error("Could not refresh the resource table for new transients");
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    auto resources = frame_graph::physical_resources_of(frame_graph_.description());
    transient_allocator_.fill(info.frame_index, resources);

    auto const graphics_tracy = context_.host_query_context.context;
    auto const compute_tracy = context_.compute_queue != context_.graphics_queue
                                       ? context_.compute_host_query_context.context
                                       : graphics_tracy;

    auto batches = frame_graph::record(frame_graph::ExecuteInfo{
            .graph = frame_graph_.description(),
            .compiled = *frame_plan_,
            .records = frame_graph_.records(),
            .resources = resources,
            .transients = &transient_allocator_.plan(info.frame_index),
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
