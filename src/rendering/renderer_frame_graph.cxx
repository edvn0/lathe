#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#include <array>
#include <expected>
#include <optional>

#include "core/logger.hxx"
#include "core/perf_events.hxx"
#include "gpu/context.hxx"
#include "rendering/frame_graph/compiler.hxx"
#include <fstream>

#include "rendering/frame_graph/describe.hxx"
#include "rendering/frame_graph/executor.hxx"
#include "rendering/frame_graph/pass_context.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/renderer.hxx"
#include "rendering/screenshot.hxx"

namespace {
    auto make_error(RendererErrorType type) -> RendererError {
        return RendererError{
                .type = type,
        };
    }

    constexpr auto fragment_stage = static_cast<frame_graph::ShaderStages>(frame_graph::ShaderStage::fragment);
    constexpr auto compute_stage = static_cast<frame_graph::ShaderStages>(frame_graph::ShaderStage::compute);

    constexpr auto ui_clear_colour = VkClearValue{.color = {.float32 = {0.0F, 0.0F, 0.0F, 1.0F}}};

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

    constexpr auto depth_attachment_state = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    constexpr auto outline_clear_colour = VkClearValue{.color = {.float32 = {0.0F, 0.0F, 0.0F, 0.0F}}};
    constexpr auto forward_clear_colour = VkClearValue{.color = {.float32 = {0.015F, 0.025F, 0.050F, 1.0F}}};

    constexpr auto buffer_idle = frame_graph::ResourceState{};

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

    constexpr auto hiz_readers = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    };

    constexpr auto sampled_by_fragment = frame_graph::ResourceState{
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    };
}

auto Renderer::record_frame(FrameRecordInfo const &info) -> std::expected<void, RendererError> {
    submit_batches_.clear();

    auto const &swapchain_image = info.swapchain_image;

    if (!initialized_ || info.command_buffer == VK_NULL_HANDLE || swapchain_image.image == VK_NULL_HANDLE ||
        swapchain_image.view == VK_NULL_HANDLE || swapchain_image.format == VK_FORMAT_UNDEFINED ||
        swapchain_image.extent.width == 0 || swapchain_image.extent.height == 0 || info.frame_index >= frames_.size()) {
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }

    pass_profiler_.begin_slot(info.frame_index);
    last_frame_timings_.passes.assign(pass_profiler_.timings().begin(), pass_profiler_.timings().end());

    screenshot_->try_resolve(info.frame_index);

    auto &frame = frames_[info.frame_index];
    consume_culled_readback(frame);

    auto const targets = resolve_frame_targets(frame);
    if (!targets) {
        return std::unexpected(targets.error());
    }

    auto const overlay_iteration = overlays_.iterate();

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

    auto const multisampled = targets->multisampled;
    auto const bloom_enabled = bloom_settings_.enabled;
    auto const ao_enabled = ao_settings_.enabled;
    auto const environment_pending = environment_.has_pending_record();

    auto const async_occlusion_enabled = (async_candidates_ & async_occlusion) != 0;
    auto const async_gtao_enabled = (async_candidates_ & async_gtao) != 0;
    auto const async_light_enabled = (async_candidates_ & async_light_clustering) != 0;

    auto hdr_image = frame_graph::ImageId{};

    auto outline_image = frame_graph::ImageId{};
    auto depth_image = frame_graph::ImageId{};
    auto resolved_depth_image = frame_graph::ImageId{};
    auto const target_description = [&](VkFormat format, VkSampleCountFlagBits samples, bool bindless,
                                        std::string_view name) {
        return frame_graph::TransientImageDesc{
                .format = format,
                .extent = {targets->extent.width, targets->extent.height, 1},
                .samples = samples,
                .descriptor_views = bindless ? image_descriptor_view_bit(ImageDescriptorView::sampled_2d) : 0U,
                .debug_name = name,
        };
    };

    auto shadow_image = frame_graph_.import_image({
            .entry = shadow_atlas_initialized_ ? sampled_by_fragment : frame_graph::ResourceState{},
            .exit = sampled_by_fragment,
            .debug_name = "shadow_atlas",
            .image = physical_image(*targets->shadow_atlas),
    });

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
    auto const transient_index = [&](frame_graph::ImageId image) {
        return transient_allocator_.handle(info.frame_index, image.index).index;
    };

    auto bloom_image = frame_graph::ImageId{};

    using frame_graph::ShaderStage;
    constexpr auto geometry_stages = ShaderStage::vertex | ShaderStage::task | ShaderStage::mesh;
    constexpr auto draw_stages = geometry_stages | ShaderStage::fragment;

    auto const buffers_concurrent = context_.queue_families.compute != context_.queue_families.graphics;

    auto const import_frame_buffer = [&](Buffer const &buffer, std::string_view name,
                                         bool read_only) -> frame_graph::BufferId {
        return frame_graph_.import_buffer({
                .entry = buffer_idle,
                .exit = buffer_idle,
                .sharing = buffers_concurrent ? frame_graph::Sharing::concurrent : frame_graph::Sharing::exclusive,
                .read_only = read_only,
                .debug_name = name,
                .buffer = physical_buffer(buffer),
        });
    };

    auto const batch_bounds = import_frame_buffer(frame.batch_bounds_buffer, "batch_bounds", true);
    auto const occlusion_views = import_frame_buffer(frame.occlusion_views_buffer, "occlusion_views", true);
    auto occlusion_candidates = import_frame_buffer(frame.occlusion_candidates_buffer, "occlusion_candidates", false);
    auto cull_chunks = import_frame_buffer(frame.cull_chunks_buffer, "cull_chunks", false);

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

    auto visible_draws = import_frame_buffer(frame.visible_draw_buffer, "visible_draws", false);
    auto visible_transforms = import_frame_buffer(frame.visible_transform_buffer, "visible_transforms", false);
    auto source_draws = import_frame_buffer(frame.draw_buffer, "draws", false);
    auto source_transforms = import_frame_buffer(frame.transform_buffer, "transforms", false);
    auto source_indirect = import_frame_buffer(frame.indirect_buffer, "indirect", false);
    auto culled_indirect = import_frame_buffer(frame.culled_indirect_buffer, "culled_indirect", false);
    auto const frustum_planes = import_frame_buffer(frame.frustum_planes_buffer, "frustum_planes", true);

    auto const skinning_active = !frame.skin_jobs.empty();
    auto skin_upload_source = frame_graph::BufferId{};
    auto skin_input = frame_graph::BufferId{};
    auto skin_scratch = frame_graph::BufferId{};
    if (skinning_active) {
        skin_upload_source = import_frame_buffer(frame.skin_upload_buffer, "skin_upload", true);
        skin_input = import_frame_buffer(frame.skin_input_buffer, "skin_input", false);
        skin_scratch = import_frame_buffer(frame.skin_scratch_buffer, "skin_scratch", false);
    }
    auto const read_skinned_vertices = [&](frame_graph::PassBuilder &pass, frame_graph::ShaderStages stages) {
        if (skinning_active) {
            [[maybe_unused]] auto const skinned = pass.read(skin_scratch, frame_graph::Use::shader_read, stages);
        }
    };

    auto const occlusion_active = frame.occlusion_active;
    auto const meshlet_occlusion_active = frame.meshlet_occlusion_active;

    auto late_indirect = frame_graph::BufferId{};
    auto merged_indirect = frame_graph::BufferId{};
    auto hiz_image = frame_graph::ImageId{};
    if (occlusion_active) {
        late_indirect = import_frame_buffer(frame.late_indirect_buffer, "late_indirect", false);
        merged_indirect = import_frame_buffer(frame.merged_indirect_buffer, "merged_indirect", false);

        hiz_image = frame_graph_.import_image({
                .entry = hiz_readers,
                .exit = hiz_readers,
                .debug_name = "hiz",
                .image = physical_image(*hiz_.image),
        });
    }

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

    auto const topology = context_.queue_set.topology();
    auto const pass_context_of = [&](frame_graph::PassContext const &context) {
        auto const compute_only =
                context.queue == frame_graph::LogicalQueue::compute &&
                !topology.same_family(frame_graph::LogicalQueue::graphics, frame_graph::LogicalQueue::compute);
        return make_pass_context(context.command_buffer, info.frame_index, compute_only);
    };

    struct FrameState {
        std::expected<void, RendererError> result{};
        PassHandoff handoff;
    } state;

    if (frame.occlusion_active) {
        hiz_history_view_projection_ = frame.view_projection;
        hiz_history_valid_ = true;
    } else {
        hiz_history_valid_ = false;
    }

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
                                  auto const pass_context = pass_context_of(context);
                                  record_overlay_prepares(pass_context);
                              }};
                          });

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
                                      auto const pass_context = pass_context_of(context);
                                      record_environment_pass(pass_context, frame);
                                  }};
                              });
    }

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

    if (skinning_active) {
        frame_graph_.add_pass("skin_upload", frame_graph::PassType::transfer,
                              {
                                      .name_id = "skin_upload",
                                      .label = "Skin palette upload",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gray),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  [[maybe_unused]] auto const source =
                                          pass.read(skin_upload_source, frame_graph::Use::transfer_read);
                                  skin_input = pass.write_discard(skin_input, frame_graph::Use::transfer_write);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      record_skin_upload(context.command_buffer, frame);
                                  }};
                              });

        frame_graph_.add_pass("skin", frame_graph::PassType::compute,
                              {
                                      .name_id = "skin",
                                      .label = "Skinning",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Orange),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  constexpr auto compute = stages_of(ShaderStage::compute);
                                  [[maybe_unused]] auto const input =
                                          pass.read(skin_input, frame_graph::Use::shader_read, compute);
                                  skin_scratch = pass.write_discard(skin_scratch, frame_graph::Use::shader_write, compute);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      auto const pass_context = pass_context_of(context);
                                      if (auto const done = record_skin(pass_context, frame); !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });
    }

    frame_graph_.add_pass("gpu_culling", frame_graph::PassType::compute,
                          {
                                  .name_id = "gpu_culling",
                                  .label = "Culling",
                                  .color = static_cast<std::uint32_t>(tracy::Color::SlateBlue),
                          },
                          [&](frame_graph::PassBuilder &pass) {
                              constexpr auto compute = stages_of(ShaderStage::compute);
                              using frame_graph::Use;

                              source_draws = pass.write(source_draws, Use::shader_read_write, compute);
                              source_transforms = pass.write(source_transforms, Use::shader_read_write, compute);
                              source_indirect = pass.write(source_indirect, Use::shader_read_write, compute);
                              [[maybe_unused]] auto const bounds = pass.read(batch_bounds, Use::shader_read, compute);
                              [[maybe_unused]] auto const planes = pass.read(frustum_planes, Use::shader_read, compute);
                              [[maybe_unused]] auto const views = pass.read(occlusion_views, Use::shader_read, compute);
                              if (occlusion_active) {
                                  [[maybe_unused]] auto const history = pass.read(hiz_image, Use::sampled, compute);
                              }

                              visible_draws = pass.write(visible_draws, Use::shader_write, compute);
                              visible_transforms = pass.write(visible_transforms, Use::shader_write, compute);
                              culled_indirect = pass.write(culled_indirect, Use::shader_write, compute);
                              occlusion_candidates = pass.write(occlusion_candidates, Use::shader_write, compute);
                              cull_chunks = pass.write(cull_chunks, Use::shader_read_write, compute);
                              stats_buffer = pass.write(stats_buffer, Use::shader_read_write, compute);

                              return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                  auto const pass_context = pass_context_of(context);
                                  if (auto const done = record_gpu_culling(pass_context, frame); !done) {
                                      state.result = std::unexpected(done.error());
                                  }
                              }};
                          });

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
                                      auto const pass_context = pass_context_of(context);
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
                                  if (async_light_enabled) {
                                      pass.queue(frame_graph::QueueAffinity::compute_preferred);
                                  }

                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const planes =
                                          pass.read(frustum_planes, frame_graph::Use::shader_read, compute);
                                  visible_lights = pass.write(visible_lights, frame_graph::Use::shader_write, compute);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context = pass_context_of(context);
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
                                  if (async_light_enabled) {
                                      pass.queue(frame_graph::QueueAffinity::compute_preferred);
                                  }

                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const lights =
                                          pass.read(visible_lights, frame_graph::Use::shader_read, compute);
                                  cluster_lights =
                                          pass.write(cluster_lights, frame_graph::Use::shader_read_write, compute);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context = pass_context_of(context);
                                      if (auto const done = record_light_cluster(pass_context, frame); !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });

        frame_graph_.add_pass("cluster_stats_readback", frame_graph::PassType::transfer,
                              {
                                      .name_id = "cluster_stats_readback",
                                      .label = "Cluster stats readback",
                                      .color = static_cast<std::uint32_t>(tracy::Color::Gold),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  pass.side_effect();
                                  [[maybe_unused]] auto const source =
                                          pass.read(cluster_lights, frame_graph::Use::transfer_read);
                                  cluster_stats_readback =
                                          pass.write(cluster_stats_readback, frame_graph::Use::transfer_write);

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      auto const pass_context = pass_context_of(context);
                                      record_cluster_stats_readback(pass_context, frame);
                                  }};
                              });
    }

    auto const declare_shadows = [&] {
        if (frame.shadow_update_mask == 0) {
            return;
        }
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
                    read_skinned_vertices(pass, draw_stages);

                    shadow_image = pass.write_depth(shadow_image,
                                                    shadow_atlas_initialized_ ? frame_graph::LoadOp::load
                                                                              : frame_graph::LoadOp::dont_care,
                                                    frame_graph::StoreOp::store);
                    pass.render_area({.offset = {0, 0}, .extent = {shadow_atlas_width, shadow_atlas_height}});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = pass_context_of(context);
                        if (auto const done = record_shadow_pass(pass_context, frame); !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    };

    auto const shadows_after_early_prepass = async_occlusion_enabled;
    auto const shadows_after_late_prepass = !async_occlusion_enabled && async_gtao_enabled;
    if (!shadows_after_early_prepass && !shadows_after_late_prepass) {
        declare_shadows();
    }

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
                read_skinned_vertices(pass, geometry_stages);
                if (meshlet_occlusion_active) {
                    [[maybe_unused]] auto const views =
                            pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                    [[maybe_unused]] auto const history =
                            pass.read(hiz_image, frame_graph::Use::sampled, stages_of(ShaderStage::task));
                    meshlet_bits =
                            pass.write(meshlet_bits, frame_graph::Use::shader_read_write, stages_of(ShaderStage::task));
                    stats_buffer =
                            pass.write(stats_buffer, frame_graph::Use::shader_read_write, stages_of(ShaderStage::task));
                }

                depth_image = pass.create(target_description(depth_format_, samples_, !multisampled, "depth"));
                depth_image = pass.write_depth(depth_image, frame_graph::LoadOp::clear, frame_graph::StoreOp::store);
                if (multisampled) {
                    resolved_depth_image = pass.create(
                            target_description(depth_format_, VK_SAMPLE_COUNT_1_BIT, true, "resolved_depth"));
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

                    auto const pass_context = pass_context_of(context);
                    if (auto const done = record_depth_prepass(pass_context, frame, *targets,
                                                               occlusion_active ? render_pass::DepthPrepassPhase::early
                                                                                : render_pass::DepthPrepassPhase::only);
                        !done) {
                        state.result = std::unexpected(done.error());
                    }
                }};
            });

    if (shadows_after_early_prepass) {
        declare_shadows();
    }

    if (occlusion_active) {
        frame_graph_.add_pass("hiz_build", frame_graph::PassType::compute,
                              {
                                      .name_id = "hiz_build",
                                      .label = "Hi-Z build",
                                      .color = static_cast<std::uint32_t>(tracy::Color::DarkOrange),
                              },
                              [&](frame_graph::PassBuilder &pass) {
                                  if (async_occlusion_enabled) {
                                      pass.queue(frame_graph::QueueAffinity::compute_preferred);
                                  }

                                  constexpr auto compute = stages_of(ShaderStage::compute);

                                  [[maybe_unused]] auto const depth =
                                          pass.read(resolved_depth_image, frame_graph::Use::sampled, compute);
                                  hiz_image = pass.write(hiz_image, frame_graph::Use::storage_write, compute,
                                                         frame_graph::ExitUse{frame_graph::Use::sampled});

                                  return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                                      if (!state.result) {
                                          return;
                                      }

                                      auto const pass_context = pass_context_of(context);
                                      if (auto const done = record_hiz_build(pass_context, *targets,
                                                                             transient_index(resolved_depth_image));
                                          !done) {
                                          state.result = std::unexpected(done.error());
                                      }
                                  }};
                              });
    }

    if (occlusion_active) {
        frame_graph_.add_pass(
                "late_cs", frame_graph::PassType::compute,
                {
                        .name_id = "occlusion_culling",
                        .label = "Occlusion culling",
                        .color = static_cast<std::uint32_t>(tracy::Color::SlateBlue),
                },
                [&](frame_graph::PassBuilder &pass) {
                    if (async_occlusion_enabled) {
                        pass.queue(frame_graph::QueueAffinity::compute_preferred);
                    }

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
                    cull_chunks = pass.write(cull_chunks, Use::shader_read_write, compute);

                    visible_draws = pass.write(visible_draws, Use::shader_read_write, compute);
                    visible_transforms = pass.write(visible_transforms, Use::shader_read_write, compute);
                    late_indirect = pass.write(late_indirect, Use::shader_write, compute);
                    merged_indirect = pass.write(merged_indirect, Use::shader_write, compute);
                    stats_buffer = pass.write(stats_buffer, Use::shader_read_write, compute);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = pass_context_of(context);
                        if (auto const done = record_occlusion_cull_pass(pass_context, frame); !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

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
                    read_skinned_vertices(pass, geometry_stages);
                    if (meshlet_occlusion_active) {
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

                        auto const pass_context = pass_context_of(context);
                        if (auto const done = record_depth_prepass(pass_context, frame, *targets,
                                                                   render_pass::DepthPrepassPhase::late);
                            !done) {
                            state.result = std::unexpected(done.error());
                        }
                    }};
                });
    }

    if (shadows_after_late_prepass) {
        declare_shadows();
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
                    if (async_gtao_enabled) {
                        pass.queue(frame_graph::QueueAffinity::compute_preferred);
                    }

                    [[maybe_unused]] auto const depth =
                            pass.read(resolved_depth_image, frame_graph::Use::sampled, compute_stage);
                    ao_raw_image = pass.create(ao_description("ao_raw"));
                    ao_raw_image = pass.write(ao_raw_image, frame_graph::Use::storage_write, compute_stage);

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = pass_context_of(context);
                        if (auto const done = render_pass::gtao(
                                    pass_context,
                                    ambient_occlusion_info(*targets, info.frame_index,
                                                           transient_index(resolved_depth_image),
                                                           transient_index(ao_raw_image), transient_index(ao_image)));
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
                    if (async_gtao_enabled) {
                        pass.queue(frame_graph::QueueAffinity::compute_preferred);
                    }

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

                        auto const pass_context = pass_context_of(context);
                        if (auto const done = render_pass::gtao_denoise(
                                    pass_context,
                                    ambient_occlusion_info(*targets, info.frame_index,
                                                           transient_index(resolved_depth_image),
                                                           transient_index(ao_raw_image), transient_index(ao_image)));
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
                [[maybe_unused]] auto const shadows =
                        pass.read(shadow_image, frame_graph::Use::sampled, fragment_stage);
                [[maybe_unused]] auto const overlays = pass.read(overlay_data, frame_graph::Use::token_read);
                if (environment_pending) {
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
                read_skinned_vertices(pass, draw_stages);
                if (meshlet_occlusion_active) {
                    [[maybe_unused]] auto const views =
                            pass.read(occlusion_views, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                    [[maybe_unused]] auto const bits =
                            pass.read(meshlet_bits, frame_graph::Use::shader_read, stages_of(ShaderStage::task));
                }
                if (clustered) {
                    [[maybe_unused]] auto const clusters =
                            pass.read(cluster_lights, frame_graph::Use::shader_read, fragment_stage);
                }
                if (ao_enabled) {
                    [[maybe_unused]] auto const ao = pass.read(ao_image, frame_graph::Use::sampled, fragment_stage);
                }

                if (multisampled) {
                    auto const msaa = pass.color(
                            pass.create(target_description(hdr_format_, samples_, false, "hdr_msaa")),
                            frame_graph::LoadOp::clear, frame_graph::StoreOp::dont_care, forward_clear_colour);
                    hdr_image =
                            pass.create(target_description(hdr_format_, VK_SAMPLE_COUNT_1_BIT, true, "resolved_hdr"));
                    hdr_image = pass.resolve(msaa, hdr_image, VK_RESOLVE_MODE_AVERAGE_BIT);
                } else {
                    hdr_image = pass.create(target_description(hdr_format_, VK_SAMPLE_COUNT_1_BIT, true, "hdr"));
                    hdr_image = pass.color(hdr_image, frame_graph::LoadOp::clear, frame_graph::StoreOp::store,
                                           forward_clear_colour);
                }

                {
                    if (multisampled) {
                        auto const outline_msaa = pass.color(
                                pass.create(target_description(VK_FORMAT_R8_UNORM, samples_, false, "outline_msaa")),
                                frame_graph::LoadOp::clear, frame_graph::StoreOp::dont_care, outline_clear_colour);
                        outline_image = pass.create(
                                target_description(VK_FORMAT_R8_UNORM, VK_SAMPLE_COUNT_1_BIT, true, "resolved_outline"));
                        outline_image = pass.resolve(outline_msaa, outline_image, VK_RESOLVE_MODE_AVERAGE_BIT);
                    } else {
                        outline_image = pass.create(
                                target_description(VK_FORMAT_R8_UNORM, VK_SAMPLE_COUNT_1_BIT, true, "outline"));
                        outline_image = pass.color(outline_image, frame_graph::LoadOp::clear,
                                                   frame_graph::StoreOp::store, outline_clear_colour);
                    }
                }
                depth_image = pass.write_depth(depth_image, frame_graph::LoadOp::load, frame_graph::StoreOp::store);
                pass.render_area({.offset = {0, 0}, .extent = targets->extent});

                return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                    if (!state.result) {
                        return;
                    }

                    auto const pass_context = pass_context_of(context);

                    OverlayScope const scene_scope{
                            .extent = targets->extent,
                            .colour_format = hdr_format_,
                            .depth_format = depth_format_,
                            .samples = samples_,
                            .colour_attachment_count = 2U,
                    };
                    auto scene_overlays = [&] {
                        record_overlay_stage(pass_context, OverlayStage::scene, scene_scope, frame.view_projection);
                    };

                    state.handoff.ao_texture_index =
                            ao_enabled ? transient_index(ao_image) : image_storage_.white().index;

                    auto const hdr = record_forward_pass(pass_context, frame, *targets, state.handoff.ao_texture_index,
                                                         transient_index(hdr_image),
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
                    bloom_image = pass.create({
                            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                            .extent = {targets->extent.width / 2, targets->extent.height / 2, 1},
                            .mip_levels = render_pass::bloom_mip_count,
                            .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d) |
                                                image_descriptor_view_bit(ImageDescriptorView::storage_2d),
                            .mip_layer_views = true,
                            .mip_slots = true,
                            .debug_name = "bloom",
                    });
                    bloom_image = pass.write(bloom_image, frame_graph::Use::storage_write, compute_stage,
                                             frame_graph::ExitUse{frame_graph::Use::sampled});

                    return frame_graph::RecordFn{[&](frame_graph::PassContext &context) {
                        if (!state.result) {
                            return;
                        }

                        auto const pass_context = pass_context_of(context);
                        auto mip_texture_indices = std::array<std::uint32_t, render_pass::bloom_mip_count>{};
                        for (auto mip = std::uint32_t{0}; mip < render_pass::bloom_mip_count; ++mip) {
                            mip_texture_indices[mip] =
                                    transient_allocator_.mip_handle(info.frame_index, bloom_image.index, mip).index;
                        }

                        auto const bloom_result = record_bloom_pass(
                                pass_context, *targets, state.handoff.hdr,
                                *transient_allocator_.image(info.frame_index, bloom_image.index), mip_texture_indices);
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
                    [[maybe_unused]] auto const overlays = pass.read(overlay_data, frame_graph::Use::token_read);
                }
                if (bloom_enabled) {
                    [[maybe_unused]] auto const bloom =
                            pass.read(bloom_image, frame_graph::Use::sampled, fragment_stage);
                }
                [[maybe_unused]] auto const outline = pass.read(outline_image, frame_graph::Use::sampled, fragment_stage);
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

                    auto const pass_context = pass_context_of(context);

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
                                    .outline_texture_index = transient_index(outline_image),
                                    .outline_thickness_pixels = outline_settings_.thickness_pixels,
                                    .outline_colour = outline_settings_.colour,
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

                        auto const pass_context = pass_context_of(context);

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
                                              {.async_compute = context_.async_compute_mode != AsyncComputeMode::off});
    if (!compiled) {
        error("Could not compile the frame graph: {}", compiled.error());
        return std::unexpected(make_error(RendererErrorType::invalid_argument));
    }
    frame_plan_ = *compiled;

    auto const plan_changed = plan_cache_.misses() != logged_plan_misses_;
    if (plan_changed) {
        perf_events::record(PerfEvent::frame_graph_compile);
        logged_plan_misses_ = plan_cache_.misses();
        frame_graph_view_ = frame_graph::FrameGraphView{
                .graph = frame_graph_.description(),
                .compiled = *frame_plan_,
                .revision = frame_graph_view_.revision + 1,
        };
        auto per_queue = std::array<std::size_t, frame_graph::logical_queue_count>{};
        auto passes_per_queue = std::array<std::size_t, frame_graph::logical_queue_count>{};
        auto waits = std::size_t{0};
        for (auto const &batch: frame_plan_->batches) {
            ++per_queue[static_cast<std::size_t>(batch.queue)];
            passes_per_queue[static_cast<std::size_t>(batch.queue)] += batch.passes.size();
            waits += batch.waits.size();
        }
        ::info("Frame graph plan: {} batches (graphics {}, compute {}), {} passes on graphics, {} on compute, {} "
               "waits, {} ownership transfers",
               frame_plan_->batches.size(), per_queue[0], per_queue[1], passes_per_queue[0], passes_per_queue[1], waits,
               frame_plan_->transfers.size());
    }

    auto const allocated = transient_allocator_.prepare(info.frame_index, frame_graph_.description(), *frame_plan_,
                                                        transient_aliasing_);
    if (!allocated) {
        error("Could not allocate the frame graph's transients: {}", allocated.error().message);
        return std::unexpected(make_error(RendererErrorType::image_error));
    }
    if (plan_changed || *allocated) {
        frame_graph_view_.transients = transient_allocator_.plan(info.frame_index);
    }
    if (*allocated) {
        perf_events::record(PerfEvent::transient_allocation);
        auto const total_bytes = transient_allocator_.total_bytes();
        ::info("Frame graph transients: {:.1f} MiB for all frame slots ({:+.1f} MiB), {:.1f} MiB without aliasing",
               static_cast<double>(total_bytes) / (1024.0 * 1024.0),
               (static_cast<double>(total_bytes) - static_cast<double>(logged_transient_bytes_)) / (1024.0 * 1024.0),
               static_cast<double>(transient_allocator_.unaliased_bytes()) / (1024.0 * 1024.0));
        logged_transient_bytes_ = total_bytes;
        if (auto const refreshed =
                    gpu_resource_table_.prepare_frame(info.frame_index, image_storage_, sampler_storage_);
            !refreshed) {
            error("Could not refresh the resource table for new transients");
            return std::unexpected(make_error(RendererErrorType::device_error));
        }
    }

    if (plan_changed && dump_frame_graph_) {
        ::info("Frame graph (slot {}):\n{}", info.frame_index,
               frame_graph::describe(frame_graph_.description(), *frame_plan_,
                                     &transient_allocator_.plan(info.frame_index)));
    }

    if (plan_changed && !frame_graph_dot_path_.empty()) {
        auto const dot = frame_graph::to_dot(frame_graph_.description(), *frame_plan_,
                                             &transient_allocator_.plan(info.frame_index));
        if (auto file = std::ofstream{frame_graph_dot_path_, std::ios::binary | std::ios::trunc};
            file && file << dot) {
            ::info("Frame graph written to {} (render with: dot -Tsvg {} -o frame_graph.svg)", frame_graph_dot_path_,
                   frame_graph_dot_path_);
        } else {
            error("Could not write the frame graph to {}", frame_graph_dot_path_);
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
