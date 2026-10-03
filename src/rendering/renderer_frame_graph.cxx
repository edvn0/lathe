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

    // The editor clears the swapchain under its UI to this.
    constexpr auto ui_clear_colour = VkClearValue{.color = {.float32 = {0.0F, 0.0F, 0.0F, 1.0F}}};

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

    // Shared by the record lambdas, which all run inside frame_graph::record() below.
    struct FrameState {
        std::expected<void, RendererError> result{};
        CompositeInputs composite;
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
                                  state.result = record_frame_legacy(legacy_info, frame, *targets, state.composite);
                              }};
                          });

    frame_graph_.add_pass(
            "composition", frame_graph::PassType::raster,
            {
                    .name_id = "composition",
                    .label = "Composition",
                    .color = static_cast<std::uint32_t>(tracy::Color::SeaGreen),
            },
            [&](frame_graph::PassBuilder &pass) {
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
                                    .hdr = state.composite.hdr,
                                    .bloom = state.composite.bloom,
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
