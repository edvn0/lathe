#include <volk.h>

#include "rendering/imgui_renderer.hxx"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>

#include <imgui.h>

#include "core/config.hxx"
#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "gpu/swapchain.hxx"
#include "gpu/upload_ring.hxx"
#include "rendering/render_passes.hxx"
#include "rendering/renderer.hxx"

// ImGui's multi-viewport support: a window dragged out of the main one becomes a window of its own, which this
// backend gives a surface, a swapchain and its own command buffers. A detached window is rendered and presented on
// its own, apart from the frame graph, as ImGui's stock Vulkan backend does.
namespace gui {

    namespace {

        constexpr auto clear_colour = VkClearColorValue{.float32 = {0.012F, 0.012F, 0.014F, 1.0F}};

        auto framebuffer_extent(ImGuiViewport const &viewport) -> VkExtent2D {
            return {
                    .width = std::max(1U, static_cast<std::uint32_t>(viewport.Size.x * viewport.FramebufferScale.x)),
                    .height = std::max(1U, static_cast<std::uint32_t>(viewport.Size.y * viewport.FramebufferScale.y)),
            };
        }

        void transition(VkCommandBuffer command_buffer, VkImage image, VkImageLayout from, VkImageLayout to,
                        VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                        VkAccessFlags2 dst_access) {
            auto const barrier = VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = src_stage,
                    .srcAccessMask = src_access,
                    .dstStageMask = dst_stage,
                    .dstAccessMask = dst_access,
                    .oldLayout = from,
                    .newLayout = to,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            auto const dependency = VkDependencyInfo{
                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                    .imageMemoryBarrierCount = 1,
                    .pImageMemoryBarriers = &barrier,
            };
            vkCmdPipelineBarrier2(command_buffer, &dependency);
        }

    }

    // What the backend keeps for one detached window (ImGuiViewport::RendererUserData).
    struct ViewportData {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        Swapchain swapchain;

        VkCommandPool pool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, frames_in_flight> commands{};
        // Signalled when the submission of the slot has finished, so the slot's buffers can be reused.
        std::array<VkFence, frames_in_flight> fences{};

        // Geometry of this window only, so windows never wait for each other's frames.
        UploadRing vertices{UploadRing::Info{.name = "imgui.viewport.vertices"}};
        UploadRing indices{UploadRing::Info{.name = "imgui.viewport.indices", .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT}};

        // The slot (fence) of the last submission recorded with each descriptor table slot bound, or -1. The main
        // window's frame for that table slot updates the table, which must not happen while this window's work that
        // bound it is still running.
        std::array<std::int32_t, frames_in_flight> table_use = [] {
            auto none = std::array<std::int32_t, frames_in_flight>{};
            none.fill(-1);
            return none;
        }();

        // Rendered by RenderWindow, presented by SwapBuffers.
        std::optional<SwapchainFrame> frame;
    };

    struct ViewportBackend {
        static auto self() -> ImGuiRenderer & {
            return *static_cast<ImGuiRenderer *>(ImGui::GetIO().BackendRendererUserData);
        }

        static auto data_of(ImGuiViewport *viewport) -> ViewportData * {
            return static_cast<ViewportData *>(viewport->RendererUserData);
        }

        static void destroy(VulkanContext &gpu, ViewportData &data) {
            // A present may still be waiting on the semaphores the swapchain is about to free.
            vkDeviceWaitIdle(gpu.device);

            for (auto const fence: data.fences) {
                if (fence != VK_NULL_HANDLE) {
                    vkDestroyFence(gpu.device, fence, nullptr);
                }
            }
            if (data.pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(gpu.device, data.pool, nullptr);
            }
            data.swapchain.destroy();
            if (data.surface != VK_NULL_HANDLE) {
                vkDestroySurfaceKHR(gpu.instance, data.surface, nullptr);
            }
        }

        static void create_window(ImGuiViewport *viewport) {
            auto &gpu = self().renderer.context();
            auto data = std::make_unique<ViewportData>();

            auto const &platform = ImGui::GetPlatformIO();
            auto raw_surface = ImU64{0};
            if (platform.Platform_CreateVkSurface == nullptr ||
                platform.Platform_CreateVkSurface(viewport, reinterpret_cast<ImU64>(gpu.instance), nullptr,
                                                  &raw_surface) != VK_SUCCESS) {
                error("[ImGui] Could not create a surface for a detached window");
                return;
            }
            data->surface = reinterpret_cast<VkSurfaceKHR>(raw_surface);

            auto supported = VkBool32{VK_FALSE};
            vkGetPhysicalDeviceSurfaceSupportKHR(gpu.physical_device, gpu.queue_families.present, data->surface,
                                                 &supported);

            if (supported == VK_FALSE ||
                !data->swapchain.initialize(SwapchainCreateInfo{
                        .physical_device = gpu.physical_device,
                        .device = gpu.device,
                        .surface = data->surface,
                        .graphics_queue = gpu.graphics_queue,
                        .present_queue = gpu.present_queue,
                        .graphics_queue_family = gpu.queue_families.graphics,
                        .present_queue_family = gpu.queue_families.present,
                        .framebuffer_extent = framebuffer_extent(*viewport),
                        .vsync = gpu.vsync,
                        .preferred_present_mode = gpu.present_mode,
                        .image_count = 0,
                })) {
                error("[ImGui] Could not create a swapchain for a detached window");
                destroy(gpu, *data);
                return;
            }

            auto const pool_info = VkCommandPoolCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    .queueFamilyIndex = gpu.queue_families.graphics,
            };
            auto const allocate_info = [&] {
                return VkCommandBufferAllocateInfo{
                        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                        .commandPool = data->pool,
                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        .commandBufferCount = static_cast<std::uint32_t>(data->commands.size()),
                };
            };
            auto const fence_info = VkFenceCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                    .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };

            auto ok = vkCreateCommandPool(gpu.device, &pool_info, nullptr, &data->pool) == VK_SUCCESS;
            if (ok) {
                auto const info = allocate_info();
                ok = vkAllocateCommandBuffers(gpu.device, &info, data->commands.data()) == VK_SUCCESS;
            }
            for (auto &fence: data->fences) {
                ok = ok && vkCreateFence(gpu.device, &fence_info, nullptr, &fence) == VK_SUCCESS;
            }
            if (!ok) {
                error("[ImGui] Could not create the command buffers of a detached window");
                destroy(gpu, *data);
                return;
            }

            viewport->RendererUserData = data.release();
        }

        static void destroy_window(ImGuiViewport *viewport) {
            if (auto *data = data_of(viewport); data != nullptr) {
                destroy(self().renderer.context(), *data);
                delete data;
                viewport->RendererUserData = nullptr;
            }
        }

        static void set_window_size(ImGuiViewport *viewport, ImVec2 /*size*/) {
            if (auto *data = data_of(viewport); data != nullptr) {
                data->swapchain.request_recreate(framebuffer_extent(*viewport));
            }
        }

        static void render_window(ImGuiViewport *viewport, void * /*arg*/) {
            auto *const data = data_of(viewport);
            auto const *const draw_data = viewport->DrawData;
            if (data == nullptr || draw_data == nullptr || (viewport->Flags & ImGuiViewportFlags_IsMinimized) != 0) {
                return;
            }

            auto &renderer = self();
            auto &gpu = renderer.renderer.context();
            auto const slot = data->swapchain.current_slot();

            vkWaitForFences(gpu.device, 1, &data->fences[slot], VK_TRUE, UINT64_MAX);

            // A swapchain that had to be recreated or is out of date has nothing to draw to this frame.
            auto frame = data->swapchain.acquire(slot);
            if (!frame) {
                return;
            }
            vkResetFences(gpu.device, 1, &data->fences[slot]);

            auto const command_buffer = data->commands[slot];
            vkResetCommandBuffer(command_buffer, 0);
            auto const begin_info = VkCommandBufferBeginInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(command_buffer, &begin_info);

            transition(command_buffer, frame->image, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            auto const colour = VkRenderingAttachmentInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .imageView = frame->image_view,
                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .clearValue = {.color = clear_colour},
            };
            auto const rendering = VkRenderingInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                    .renderArea = {{0, 0}, frame->extent},
                    .layerCount = 1,
                    .colorAttachmentCount = 1,
                    .pColorAttachments = &colour,
            };
            vkCmdBeginRendering(command_buffer, &rendering);

            if (draw_data->TotalIdxCount > 0) {
                // The table of the frame being built: its descriptors are the ones the main window's UI uses too.
                auto const context = OverlayRecordContext{
                        .command_buffer = command_buffer,
                        .frame_index = renderer.current_frame_index,
                        .scope = {.extent = frame->extent,
                                  .colour_format = frame->format,
                                  .depth_format = VK_FORMAT_UNDEFINED,
                                  .samples = VK_SAMPLE_COUNT_1_BIT},
                };
                render_pass::set_overlay_baseline_state(command_buffer, OverlayStage::ui, context.scope);
                if (auto const bound = renderer.viewport_shader.bind(context)) {
                    // The preview image is a transient of the main window's frame graph, which is not there for
                    // this window to sample.
                    renderer.draw(*bound, *draw_data,
                                  ImGuiRenderer::DrawTarget{
                                          .vertices = data->vertices,
                                          .indices = data->indices,
                                          .ring_slot = slot,
                                          .preview_index = renderer.renderer.image_storage().black().index,
                                  });
                }
            }

            vkCmdEndRendering(command_buffer);
            transition(command_buffer, frame->image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
            vkEndCommandBuffer(command_buffer);

            auto const wait_semaphore = data->swapchain.image_available(frame->frame_index);
            auto const signal_semaphore = data->swapchain.render_finished(frame->image_index);
            auto const wait_stage = VkPipelineStageFlags{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            auto const submit = VkSubmitInfo{
                    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .waitSemaphoreCount = 1,
                    .pWaitSemaphores = &wait_semaphore,
                    .pWaitDstStageMask = &wait_stage,
                    .commandBufferCount = 1,
                    .pCommandBuffers = &command_buffer,
                    .signalSemaphoreCount = 1,
                    .pSignalSemaphores = &signal_semaphore,
            };
            if (vkQueueSubmit(gpu.graphics_queue, 1, &submit, data->fences[slot]) != VK_SUCCESS) {
                error("[ImGui] Could not submit a detached window's frame");
                return;
            }

            data->table_use[renderer.current_frame_index] = static_cast<std::int32_t>(slot);
            data->frame = *frame;
        }

        static void swap_buffers(ImGuiViewport *viewport, void * /*arg*/) {
            auto *const data = data_of(viewport);
            if (data == nullptr || !data->frame) {
                return;
            }

            auto const result = data->swapchain.present(*data->frame);
            data->frame.reset();
            if (result == SwapchainFrameResult::device_lost) {
                self().renderer.context().mark_device_lost();
            }
        }
    };

    auto ImGuiRenderer::init_viewports() -> void {
        if (!viewports_enabled) {
            return;
        }

        auto &io = ImGui::GetIO();
        io.BackendRendererUserData = this;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

        auto &platform = ImGui::GetPlatformIO();
        platform.Renderer_CreateWindow = ViewportBackend::create_window;
        platform.Renderer_DestroyWindow = ViewportBackend::destroy_window;
        platform.Renderer_SetWindowSize = ViewportBackend::set_window_size;
        platform.Renderer_RenderWindow = ViewportBackend::render_window;
        platform.Renderer_SwapBuffers = ViewportBackend::swap_buffers;
    }

    auto ImGuiRenderer::wait_for_viewports(std::uint32_t const frame_index) -> void {
        if (!viewports_enabled) {
            return;
        }

        auto &gpu = renderer.context();
        for (auto *const viewport: ImGui::GetPlatformIO().Viewports) {
            if (auto *const data = ViewportBackend::data_of(viewport); data != nullptr) {
                if (auto const slot = data->table_use[frame_index]; slot >= 0) {
                    vkWaitForFences(gpu.device, 1, &data->fences[static_cast<std::size_t>(slot)], VK_TRUE,
                                    UINT64_MAX);
                    data->table_use[frame_index] = -1;
                }
            }
        }
    }

    auto ImGuiRenderer::shutdown_viewports() -> void {
        if (!viewports_enabled) {
            return;
        }

        // Before the platform backend goes: the windows are destroyed through its callbacks too.
        ImGui::DestroyPlatformWindows();
        ImGui::GetIO().BackendRendererUserData = nullptr;
    }

}
