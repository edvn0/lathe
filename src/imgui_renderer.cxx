#include <volk.h>

#include "imgui_renderer.hxx"

#include <backends/imgui_impl_glfw.h>
#include <misc/freetype/imgui_freetype.h>

#include <imgui.h>

#include <ImGuizmo.h>
#include <bit>
#include <filesystem>
#include <implot.h>
#include <unordered_map>
#include <utility>

#include "context.hxx"
#include "error_describe.hxx"
#include "logger.hxx"
#include "pipeline.hxx"
#include "renderer.hxx"

namespace gui {

    namespace {
        [[nodiscard]] constexpr auto next_power_of_two(std::size_t value) noexcept -> std::size_t {
            if (value <= 1) {
                return 1;
            }

            --value;
            for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
                value |= value >> shift;
            }

            return value + 1;
        }

        struct PC {
            std::array<float, 4> lrtb{};
            VkDeviceAddress vb;
            std::uint32_t base_vertex;
            std::uint32_t texture_id;
            std::uint32_t sampler_id;
            std::uint32_t _pad = 0;
        };

        auto apply_dark_theme() -> void {
            ImGui::StyleColorsDark();
            ImGuiStyle &style = ImGui::GetStyle();

            style.WindowPadding = {8.f, 8.f};
            style.FramePadding = {6.f, 4.f};
            style.CellPadding = {6.f, 4.f};
            style.ItemSpacing = {8.f, 4.f};
            style.ItemInnerSpacing = {4.f, 4.f};
            style.IndentSpacing = 16.f;
            style.ScrollbarSize = 12.f;
            style.GrabMinSize = 8.f;

            style.WindowRounding = 4.f;
            style.ChildRounding = 4.f;
            style.FrameRounding = 3.f;
            style.PopupRounding = 4.f;
            style.ScrollbarRounding = 6.f;
            style.GrabRounding = 3.f;
            style.TabRounding = 4.f;

            style.WindowBorderSize = 1.f;
            style.ChildBorderSize = 1.f;
            style.PopupBorderSize = 1.f;
            style.FrameBorderSize = 0.f;
            style.TabBorderSize = 0.f;

            auto *c = style.Colors;
            c[ImGuiCol_Text] = {0.82f, 0.82f, 0.82f, 1.00f};
            c[ImGuiCol_TextDisabled] = {0.42f, 0.42f, 0.44f, 1.00f};
            c[ImGuiCol_WindowBg] = {0.13f, 0.13f, 0.14f, 1.00f};
            c[ImGuiCol_ChildBg] = {0.10f, 0.10f, 0.11f, 1.00f};
            c[ImGuiCol_PopupBg] = {0.11f, 0.11f, 0.12f, 0.96f};
            c[ImGuiCol_Border] = {0.25f, 0.25f, 0.27f, 0.60f};
            c[ImGuiCol_BorderShadow] = {0.00f, 0.00f, 0.00f, 0.00f};
            c[ImGuiCol_FrameBg] = {0.18f, 0.18f, 0.20f, 1.00f};
            c[ImGuiCol_FrameBgHovered] = {0.24f, 0.24f, 0.26f, 1.00f};
            c[ImGuiCol_FrameBgActive] = {0.28f, 0.28f, 0.31f, 1.00f};
            c[ImGuiCol_TitleBg] = {0.09f, 0.09f, 0.10f, 1.00f};
            c[ImGuiCol_TitleBgActive] = {0.09f, 0.09f, 0.10f, 1.00f};
            c[ImGuiCol_TitleBgCollapsed] = {0.09f, 0.09f, 0.10f, 0.75f};
            c[ImGuiCol_MenuBarBg] = {0.11f, 0.11f, 0.12f, 1.00f};
            c[ImGuiCol_ScrollbarBg] = {0.00f, 0.00f, 0.00f, 0.00f};
            c[ImGuiCol_ScrollbarGrab] = {0.28f, 0.28f, 0.30f, 1.00f};
            c[ImGuiCol_ScrollbarGrabHovered] = {0.34f, 0.34f, 0.37f, 1.00f};
            c[ImGuiCol_ScrollbarGrabActive] = {0.40f, 0.40f, 0.44f, 1.00f};
            c[ImGuiCol_CheckMark] = {0.26f, 0.59f, 0.98f, 1.00f};
            c[ImGuiCol_SliderGrab] = {0.26f, 0.59f, 0.98f, 0.90f};
            c[ImGuiCol_SliderGrabActive] = {0.46f, 0.54f, 0.80f, 1.00f};
            c[ImGuiCol_Button] = {0.24f, 0.24f, 0.27f, 1.00f};
            c[ImGuiCol_ButtonHovered] = {0.26f, 0.59f, 0.98f, 0.55f};
            c[ImGuiCol_ButtonActive] = {0.26f, 0.59f, 0.98f, 1.00f};
            c[ImGuiCol_Header] = {0.26f, 0.59f, 0.98f, 0.25f};
            c[ImGuiCol_HeaderHovered] = {0.26f, 0.59f, 0.98f, 0.50f};
            c[ImGuiCol_HeaderActive] = {0.26f, 0.59f, 0.98f, 0.90f};
            c[ImGuiCol_Separator] = {0.25f, 0.25f, 0.27f, 0.60f};
            c[ImGuiCol_SeparatorHovered] = {0.26f, 0.59f, 0.98f, 0.60f};
            c[ImGuiCol_SeparatorActive] = {0.26f, 0.59f, 0.98f, 1.00f};
            c[ImGuiCol_ResizeGrip] = {0.26f, 0.59f, 0.98f, 0.20f};
            c[ImGuiCol_ResizeGripHovered] = {0.26f, 0.59f, 0.98f, 0.67f};
            c[ImGuiCol_ResizeGripActive] = {0.26f, 0.59f, 0.98f, 0.95f};
            c[ImGuiCol_Tab] = {0.09f, 0.09f, 0.10f, 1.00f};
            c[ImGuiCol_TabHovered] = {0.30f, 0.30f, 0.34f, 1.00f};
            c[ImGuiCol_TabActive] = {0.20f, 0.20f, 0.23f, 1.00f};
            c[ImGuiCol_TabUnfocused] = {0.09f, 0.09f, 0.10f, 1.00f};
            c[ImGuiCol_TabUnfocusedActive] = {0.14f, 0.14f, 0.16f, 1.00f};
            c[ImGuiCol_DockingPreview] = {0.26f, 0.59f, 0.98f, 0.60f};
            c[ImGuiCol_DockingEmptyBg] = {0.10f, 0.10f, 0.11f, 1.00f};
            c[ImGuiCol_PlotLines] = {0.61f, 0.61f, 0.61f, 1.00f};
            c[ImGuiCol_PlotLinesHovered] = {1.00f, 0.43f, 0.35f, 1.00f};
            c[ImGuiCol_PlotHistogram] = {0.26f, 0.59f, 0.98f, 1.00f};
            c[ImGuiCol_PlotHistogramHovered] = {1.00f, 0.43f, 0.35f, 1.00f};
            c[ImGuiCol_TableHeaderBg] = {0.13f, 0.13f, 0.15f, 1.00f};
            c[ImGuiCol_TableBorderStrong] = {0.25f, 0.25f, 0.27f, 1.00f};
            c[ImGuiCol_TableBorderLight] = {0.20f, 0.20f, 0.22f, 1.00f};
            c[ImGuiCol_TableRowBg] = {0.00f, 0.00f, 0.00f, 0.00f};
            c[ImGuiCol_TableRowBgAlt] = {1.00f, 1.00f, 1.00f, 0.03f};
            c[ImGuiCol_TextSelectedBg] = {0.26f, 0.59f, 0.98f, 0.35f};
            c[ImGuiCol_DragDropTarget] = {0.26f, 0.59f, 0.98f, 0.90f};
            c[ImGuiCol_NavHighlight] = {0.26f, 0.59f, 0.98f, 1.00f};
            c[ImGuiCol_NavWindowingHighlight] = {1.00f, 1.00f, 1.00f, 0.70f};
            c[ImGuiCol_NavWindowingDimBg] = {0.80f, 0.80f, 0.80f, 0.20f};
            c[ImGuiCol_ModalWindowDimBg] = {0.10f, 0.10f, 0.10f, 0.45f};
        }

        auto create_pipeline(Renderer &r, VkFormat fb) -> std::expected<PipelineNodeHandle, RendererError> {
            VkPushConstantRange const pc_range{
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    .offset = 0,
                    .size = sizeof(PC),
            };

            return r.register_pipeline(PipelineRegisterInfo{
                    .stages =
                            {
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/gui.slang",
                                            .entry_point = "vs_main",
                                            .stage = renderer::ShaderStage::vertex,
                                    },
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/gui.slang",
                                            .entry_point = "fs_main",
                                            .stage = renderer::ShaderStage::fragment,
                                    },
                            },
                    .push_constant_ranges = {pc_range},
                    .colour_formats = {fb},
                    .depth_format = VK_FORMAT_UNDEFINED,
                    .stencil_format = VK_FORMAT_UNDEFINED,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .blending = true,
                    .debug_name = "gui.pipeline",
            });
        }
    } // namespace

    ImGuiRenderer::ImGuiRenderer(Renderer &r, FontChoice font) :
        ImGuiRenderer(r.context().window, r.context().swapchain.frame_count(), r, font) {}

    ImGuiRenderer::ImGuiRenderer(GLFWwindow *w, std::uint32_t initial_slot_count, Renderer &r, FontChoice font) :
        renderer(r) {

        std::ignore = ImGui::CreateContext();
        std::ignore = ImPlot::CreateContext();
        apply_dark_theme();

        ImGuiIO &io = ImGui::GetIO();
        io.BackendRendererName = "imgui-custom-vulkan";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

        // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports;
            io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

            ImGuiStyle &style = ImGui::GetStyle();
            style.WindowRounding = 0.0F;
            style.Colors[ImGuiCol_WindowBg].w = 1.0F;
        }

        update_font(std::move(font));
        ImGui_ImplGlfw_InitForVulkan(w, true);
        slots_per_frame = std::max(1u, initial_slot_count);
        drawables.resize(frames_in_flight * slots_per_frame);
    }

    ImGuiRenderer::~ImGuiRenderer() {
        ImGuiIO &io = ImGui::GetIO();
        io.Fonts->TexID = ImTextureID{0};

        ImGui_ImplGlfw_Shutdown();

        ImGui::DestroyPlatformWindows();

        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }

    auto ImGuiRenderer::begin_frame(ImGuiFramebuffer fb) -> void {
        const auto &dim = std::get<VkExtent2D>(fb);

        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize =
                ImVec2(static_cast<float>(dim.width) / display_scale, static_cast<float>(dim.height) / display_scale);
        io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);

        if (force_recompile_primary || !main_pipeline.valid()) {
            auto created = create_pipeline(renderer, std::get<1>(fb));

            if (!created) {
                error("(ImGui) Failed to create UI pipeline");
            } else {
                main_pipeline = *created;
                force_recompile_primary = false;
            }
        }

        slot_cursor = 0;
        frame_cursor = (frame_cursor + 1) % frames_in_flight;

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
    }

    auto ImGuiRenderer::acquire_draw_slot() -> DrawableData & {
        if (slot_cursor >= slots_per_frame) {
            std::uint32_t new_slots_per_frame = std::max(slots_per_frame * 2u, slot_cursor + 1u);
            std::vector<DrawableData> new_drawables(frames_in_flight * new_slots_per_frame);

            for (std::uint32_t f = 0; f < frames_in_flight; ++f) {
                for (std::uint32_t s = 0; s < slots_per_frame; ++s) {
                    new_drawables[f * new_slots_per_frame + s] = std::move(drawables[f * slots_per_frame + s]);
                }
            }

            drawables = std::move(new_drawables);
            slots_per_frame = new_slots_per_frame;
        }

        DrawableData &out = drawables[frame_cursor * slots_per_frame + slot_cursor];
        slot_cursor++;
        return out;
    }

    auto ImGuiRenderer::end_frame() -> void {
        ImGui::EndFrame();
        ImGui::Render();

        if (auto &io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
        }
    }

    auto ImGuiRenderer::render(VkCommandBuffer cmd, std::uint32_t frame_index) -> void {
        if (auto const *pipeline = renderer.resolve_pipeline(main_pipeline)) {
            render_draw_data(cmd, ImGui::GetDrawData(), *pipeline, frame_index);
        }
    }

    auto ImGuiRenderer::render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, Pipeline const &pipeline,
                                         std::uint32_t frame_index) -> void {
        if (!dd || dd->TotalIdxCount == 0) {
            return;
        }

        const float fb_width = dd->DisplaySize.x * dd->FramebufferScale.x;
        const float fb_height = dd->DisplaySize.y * dd->FramebufferScale.y;

        VkViewport vp{
                .x = 0,
                .y = fb_height,
                .width = fb_width,
                .height = -fb_height,
                .minDepth = 0.0F,
                .maxDepth = 1.0F,
        };
        vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_ALWAYS);
        vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
        vkCmdSetDepthTestEnable(cmd, VK_FALSE);
        vkCmdSetDepthWriteEnable(cmd, VK_FALSE);

        vkCmdSetViewport(cmd, 0, 1, &vp);

        const float L = dd->DisplayPos.x;
        const float R = dd->DisplayPos.x + dd->DisplaySize.x;
        const float T = dd->DisplayPos.y;
        const float B = dd->DisplayPos.y + dd->DisplaySize.y;
        const ImVec2 clip_offset = dd->DisplayPos;
        const ImVec2 clip_scale = dd->FramebufferScale;

        DrawableData &drawable = acquire_draw_slot();

        if (std::cmp_less(drawable.index_count, dd->TotalIdxCount)) {
            const auto size = static_cast<std::size_t>(dd->TotalIdxCount * 4) * sizeof(ImDrawIdx);
            const auto actual_size = next_power_of_two(size);
            info("(ImGui) Reallocating index buffer to {} bytes", actual_size);

            auto created = Buffer::create(renderer.context(), BufferCreateInfo{
                                                                      .size = actual_size,
                                                                      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                      .memory = BufferMemory::upload,
                                                                      .debug_name = "imgui_index_buffer",
                                                              });

            if (!created) {
                error("(ImGui) Failed to allocate index buffer");
                return;
            }

            drawable.index = std::make_unique<Buffer>(std::move(*created));
            drawable.index_count = static_cast<std::uint32_t>(actual_size / sizeof(ImDrawIdx));
        }

        if (static_cast<std::int32_t>(drawable.vertex_count) < dd->TotalVtxCount) {
            const auto size = static_cast<std::size_t>(dd->TotalVtxCount * 4) * sizeof(ImDrawVert);
            const auto actual_size = next_power_of_two(size);
            info("(ImGui) Reallocating vertex buffer to {} bytes", actual_size);

            auto created =
                    Buffer::create(renderer.context(), BufferCreateInfo{
                                                               .size = actual_size,
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                               .memory = BufferMemory::upload,
                                                               .debug_name = "imgui_vertex_buffer",
                                                       });

            if (!created) {
                error("(ImGui) Failed to allocate vertex buffer");
                return;
            }

            drawable.vertex = std::make_unique<Buffer>(std::move(*created));
            drawable.vertex_count = static_cast<std::uint32_t>(actual_size / sizeof(ImDrawVert));
        }

        {
            std::vector<ImDrawVert> all_vtx;
            std::vector<ImDrawIdx> all_itx;

            all_vtx.reserve(static_cast<std::size_t>(dd->TotalVtxCount));
            all_itx.reserve(static_cast<std::size_t>(dd->TotalIdxCount));

            for (int n = 0; n < dd->CmdListsCount; n++) {
                const auto *imgui_cmd = dd->CmdLists[n];
                all_vtx.insert(all_vtx.end(), imgui_cmd->VtxBuffer.Data,
                               imgui_cmd->VtxBuffer.Data + imgui_cmd->VtxBuffer.Size);
                all_itx.insert(all_itx.end(), imgui_cmd->IdxBuffer.Data,
                               imgui_cmd->IdxBuffer.Data + imgui_cmd->IdxBuffer.Size);
            }

            if (!drawable.vertex->write(0, std::span<const ImDrawVert>(all_vtx))) {
                error("(ImGui) Failed to write vertex buffer");
            }
            if (!drawable.index->write(0, std::span<const ImDrawIdx>(all_itx))) {
                error("(ImGui) Failed to write index buffer");
            }
        }

        vkCmdBindIndexBuffer(cmd, drawable.index->buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline());

        renderer.resource_table().bind(cmd, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout());

        std::uint32_t index_offset = 0;
        std::uint32_t vertex_offset = 0;

        for (int n = 0; n < dd->CmdListsCount; n++) {
            const auto *command_list = dd->CmdLists[n];

            for (int cmd_i = 0; cmd_i < command_list->CmdBuffer.Size; cmd_i++) {
                const auto &imgui_cmd = command_list->CmdBuffer[cmd_i];

                ImVec2 clip_min((imgui_cmd.ClipRect.x - clip_offset.x) * clip_scale.x,
                                (imgui_cmd.ClipRect.y - clip_offset.y) * clip_scale.y);
                ImVec2 clip_max((imgui_cmd.ClipRect.z - clip_offset.x) * clip_scale.x,
                                (imgui_cmd.ClipRect.w - clip_offset.y) * clip_scale.y);

                clip_min.x = std::max(clip_min.x, 0.0F);
                clip_min.y = std::max(clip_min.y, 0.0F);
                clip_max.x = std::min(clip_max.x, fb_width);
                clip_max.y = std::min(clip_max.y, fb_height);

                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) {
                    continue;
                }

                PC pc{
                        .lrtb = {L, R, T, B},
                        .vb = drawable.vertex->device_address,
                        .base_vertex = vertex_offset + imgui_cmd.VtxOffset,
                        .texture_id = static_cast<std::uint32_t>(imgui_cmd.GetTexID()),
                        .sampler_id = sampler.index,
                };

                vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(pc), &pc);

                VkRect2D scissor{
                        .offset =
                                {
                                        .x = static_cast<std::int32_t>(clip_min.x),
                                        .y = static_cast<std::int32_t>(clip_min.y),
                                },
                        .extent =
                                {
                                        .width = static_cast<std::uint32_t>(clip_max.x - clip_min.x),
                                        .height = static_cast<std::uint32_t>(clip_max.y - clip_min.y),
                                },
                };
                vkCmdSetScissor(cmd, 0, 1, &scissor);

                vkCmdDrawIndexed(cmd, imgui_cmd.ElemCount, 1, index_offset + imgui_cmd.IdxOffset,
                                 static_cast<std::int32_t>(vertex_offset + imgui_cmd.VtxOffset), 0);
            }

            index_offset += static_cast<std::uint32_t>(command_list->IdxBuffer.Size);
            vertex_offset += static_cast<std::uint32_t>(command_list->VtxBuffer.Size);
        }
    }

    auto ImGuiRenderer::set_app_name(const std::string_view name) -> void {
        config_name = std::format("{}.ini", name);
        config_path = std::make_unique<std::filesystem::path>(config_name);
    }

    auto ImGuiRenderer::update_font(FontChoice f) -> void {
        ImGuiIO &io = ImGui::GetIO();
        ImFontConfig cfg{};
        cfg.FontDataOwnedByAtlas = false;
        cfg.RasterizerMultiply = 1.5f;
        cfg.SizePixels = std::ceilf(f.size);
        cfg.PixelSnapH = true;
        cfg.OversampleH = 4;
        cfg.OversampleV = 4;
        cfg.FontLoaderFlags = ImGuiFreeTypeLoaderFlags_ForceAutoHint | ImGuiFreeTypeLoaderFlags_LightHinting;

        ImFont *font = nullptr;
        std::filesystem::path const font_path{f.font_path};

        if (std::filesystem::exists(font_path)) {
            const auto path_str = font_path.string();
            font = io.Fonts->AddFontFromFileTTF(path_str.c_str(), cfg.SizePixels, &cfg);
        }

        io.Fonts->Flags |= ImFontAtlasFlags_NoPowerOfTwoHeight;

        unsigned char *pixels;
        int width;
        int height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

        const auto *as_bytes = std::bit_cast<const std::byte *>(pixels);

        info("Font atlas size: {} x {}", width, height);

        auto created_image = renderer.image_storage().create_image(
                ImageCreateInfo{
                        .extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1},
                        .format = VK_FORMAT_R8G8B8A8_UNORM,
                        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .descriptor_views = image_descriptor_view_bit(ImageDescriptorView::sampled_2d),
                        .mip_levels = 1,
                        .array_layers = 1,
                        .debug_name = "imgui_fonts",
                },
                std::span<const std::byte>(as_bytes,
                                           static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 4));

        if (!created_image) {
            error("(ImGui) Failed to create font atlas image: {}", describe(created_image.error()));
            return;
        }

        font_texture = *created_image;
        io.Fonts->TexID = font_texture.index;
        io.FontDefault = font;

        auto created_sampler = renderer.sampler_storage().create_sampler(SamplerCreateInfo{
                .mag_filter = VK_FILTER_LINEAR,
                .min_filter = VK_FILTER_LINEAR,
                .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .compare_op = VK_COMPARE_OP_ALWAYS,
                .max_lod = VK_LOD_CLAMP_NONE,
                .sampler_class = SamplerClass::regular,
                .debug_name = "imgui_font_sampler",
        });

        if (!created_sampler) {
            error("(ImGui) Failed to create font sampler");
            return;
        }

        sampler = *created_sampler;
    }

} // namespace gui
