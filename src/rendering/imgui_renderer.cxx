#include <volk.h>

#include "rendering/imgui_renderer.hxx"

#include <backends/imgui_impl_glfw.h>
#include <misc/freetype/imgui_freetype.h>

#include <imgui.h>

#include <ImGuizmo.h>
#include <array>
#include <bit>
#include <filesystem>
#include <implot.h>
#include <unordered_map>
#include <utility>

// Font Awesome 6 icon ranges and the compressed solid font, both from ImGuiNotify (toast icons).
#include "IconsFontAwesome6.h"
#include "fa-solid-900.h"

#include "core/error_describe.hxx"
#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"
#include "gpu/context.hxx"
#include "rendering/renderer.hxx"

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
            std::uint32_t sampler_id{0};
            // See gui::linear_source_texture_bit.
            std::uint32_t already_linear{0};
        };

        auto apply_dark_theme() -> void {
            ImGui::StyleColorsDark();
            ImGuiStyle &style = ImGui::GetStyle();

            style.WindowPadding = {8.F, 8.F};
            style.FramePadding = {6.F, 4.F};
            style.CellPadding = {6.F, 4.F};
            style.ItemSpacing = {8.F, 4.F};
            style.ItemInnerSpacing = {4.F, 4.F};
            style.IndentSpacing = 16.F;
            style.ScrollbarSize = 12.F;
            style.GrabMinSize = 8.F;

            style.WindowRounding = 4.F;
            style.ChildRounding = 4.F;
            style.FrameRounding = 3.F;
            style.PopupRounding = 4.F;
            style.ScrollbarRounding = 6.F;
            style.GrabRounding = 3.F;
            style.TabRounding = 4.F;

            style.WindowBorderSize = 1.F;
            style.ChildBorderSize = 1.F;
            style.PopupBorderSize = 1.F;
            style.FrameBorderSize = 0.F;
            style.TabBorderSize = 0.F;

            auto *c = style.Colors;
            c[ImGuiCol_Text] = {0.82F, 0.82F, 0.82F, 1.00F};
            c[ImGuiCol_TextDisabled] = {0.42F, 0.42F, 0.44F, 1.00F};
            c[ImGuiCol_WindowBg] = {0.13F, 0.13F, 0.14F, 1.00F};
            c[ImGuiCol_ChildBg] = {0.10F, 0.10F, 0.11F, 1.00F};
            c[ImGuiCol_PopupBg] = {0.11F, 0.11F, 0.12F, 0.96F};
            c[ImGuiCol_Border] = {0.25F, 0.25F, 0.27F, 0.60F};
            c[ImGuiCol_BorderShadow] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_FrameBg] = {0.18F, 0.18F, 0.20F, 1.00F};
            c[ImGuiCol_FrameBgHovered] = {0.24F, 0.24F, 0.26F, 1.00F};
            c[ImGuiCol_FrameBgActive] = {0.28F, 0.28F, 0.31F, 1.00F};
            c[ImGuiCol_TitleBg] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TitleBgActive] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TitleBgCollapsed] = {0.09F, 0.09F, 0.10F, 0.75F};
            c[ImGuiCol_MenuBarBg] = {0.11F, 0.11F, 0.12F, 1.00F};
            c[ImGuiCol_ScrollbarBg] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_ScrollbarGrab] = {0.28F, 0.28F, 0.30F, 1.00F};
            c[ImGuiCol_ScrollbarGrabHovered] = {0.34F, 0.34F, 0.37F, 1.00F};
            c[ImGuiCol_ScrollbarGrabActive] = {0.40F, 0.40F, 0.44F, 1.00F};
            c[ImGuiCol_CheckMark] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_SliderGrab] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_SliderGrabActive] = {0.46F, 0.54F, 0.80F, 1.00F};
            c[ImGuiCol_Button] = {0.24F, 0.24F, 0.27F, 1.00F};
            c[ImGuiCol_ButtonHovered] = {0.26F, 0.59F, 0.98F, 0.55F};
            c[ImGuiCol_ButtonActive] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_Header] = {0.26F, 0.59F, 0.98F, 0.25F};
            c[ImGuiCol_HeaderHovered] = {0.26F, 0.59F, 0.98F, 0.50F};
            c[ImGuiCol_HeaderActive] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_Separator] = {0.25F, 0.25F, 0.27F, 0.60F};
            c[ImGuiCol_SeparatorHovered] = {0.26F, 0.59F, 0.98F, 0.60F};
            c[ImGuiCol_SeparatorActive] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_ResizeGrip] = {0.26F, 0.59F, 0.98F, 0.20F};
            c[ImGuiCol_ResizeGripHovered] = {0.26F, 0.59F, 0.98F, 0.67F};
            c[ImGuiCol_ResizeGripActive] = {0.26F, 0.59F, 0.98F, 0.95F};
            c[ImGuiCol_Tab] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TabHovered] = {0.30F, 0.30F, 0.34F, 1.00F};
            c[ImGuiCol_TabActive] = {0.20F, 0.20F, 0.23F, 1.00F};
            c[ImGuiCol_TabUnfocused] = {0.09F, 0.09F, 0.10F, 1.00F};
            c[ImGuiCol_TabUnfocusedActive] = {0.14F, 0.14F, 0.16F, 1.00F};
            c[ImGuiCol_DockingPreview] = {0.26F, 0.59F, 0.98F, 0.60F};
            c[ImGuiCol_DockingEmptyBg] = {0.10F, 0.10F, 0.11F, 1.00F};
            c[ImGuiCol_PlotLines] = {0.61F, 0.61F, 0.61F, 1.00F};
            c[ImGuiCol_PlotLinesHovered] = {1.00F, 0.43F, 0.35F, 1.00F};
            c[ImGuiCol_PlotHistogram] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_PlotHistogramHovered] = {1.00F, 0.43F, 0.35F, 1.00F};
            c[ImGuiCol_TableHeaderBg] = {0.13F, 0.13F, 0.15F, 1.00F};
            c[ImGuiCol_TableBorderStrong] = {0.25F, 0.25F, 0.27F, 1.00F};
            c[ImGuiCol_TableBorderLight] = {0.20F, 0.20F, 0.22F, 1.00F};
            c[ImGuiCol_TableRowBg] = {0.00F, 0.00F, 0.00F, 0.00F};
            c[ImGuiCol_TableRowBgAlt] = {1.00F, 1.00F, 1.00F, 0.03F};
            c[ImGuiCol_TextSelectedBg] = {0.26F, 0.59F, 0.98F, 0.35F};
            c[ImGuiCol_DragDropTarget] = {0.26F, 0.59F, 0.98F, 0.90F};
            c[ImGuiCol_NavHighlight] = {0.26F, 0.59F, 0.98F, 1.00F};
            c[ImGuiCol_NavWindowingHighlight] = {1.00F, 1.00F, 1.00F, 0.70F};
            c[ImGuiCol_NavWindowingDimBg] = {0.80F, 0.80F, 0.80F, 0.20F};
            c[ImGuiCol_ModalWindowDimBg] = {0.10F, 0.10F, 0.10F, 0.45F};
        }

        auto create_pipeline(Renderer &r, VkFormat fb) -> std::expected<PipelineNodeHandle, RendererError> {
            return r.register_pipeline(PipelineRegisterInfo{
                    .stages =
                            {
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/gui.slang",
                                            .entry_point = "main_vs",
                                            .stage = renderer::ShaderStage::vertex,
                                    },
                                    renderer::ShaderCompileRequest{
                                            .source_path = "assets/shaders/gui.slang",
                                            .entry_point = "main_fs",
                                            .stage = renderer::ShaderStage::fragment,
                                    },
                            },
                    .push_constant_ranges = {global_push_constant_range},
                    .colour_formats = {fb},
                    .dynamic_states = {},
                    .depth_format = VK_FORMAT_UNDEFINED,
                    .stencil_format = VK_FORMAT_UNDEFINED,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .blending = true,
                    .debug_name = "gui.pipeline",
            });
        }
    } // namespace

    ImGuiRenderer::ImGuiRenderer(Renderer &r, FontChoice const &font) :
        ImGuiRenderer(r.context().window, r.context().swapchain.frame_count(), r, font) {}

    ImGuiRenderer::ImGuiRenderer(GLFWwindow *w, std::uint32_t initial_slot_count, Renderer &r, FontChoice const &font) :
        renderer(r) {

        std::ignore = ImGui::CreateContext();
        std::ignore = ImPlot::CreateContext();
        apply_dark_theme();

        ImGuiIO &io = ImGui::GetIO();
        io.BackendRendererName = "imgui-custom-vulkan";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            io.BackendFlags |= ImGuiBackendFlags_PlatformHasViewports;
            io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

            ImGuiStyle &style = ImGui::GetStyle();
            style.WindowRounding = 0.0F;
            style.Colors[ImGuiCol_WindowBg].w = 1.0F;
        }

        update_font(font);
        ImGui_ImplGlfw_InitForVulkan(w, true);
        slots_per_frame = std::max(1u, initial_slot_count);
        drawables.resize(static_cast<std::size_t>(frames_in_flight) * slots_per_frame);
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
        auto const &dim = std::get<VkExtent2D>(fb);

        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize =
                ImVec2(static_cast<float>(dim.width) / display_scale, static_cast<float>(dim.height) / display_scale);
        io.DisplayFramebufferScale = ImVec2(display_scale, display_scale);

        if (force_recompile_primary || !main_pipeline.valid()) {
            auto created = create_pipeline(renderer, std::get<1>(fb));

            if (!created) {
                error("[ImGui] Failed to create UI pipeline");
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
            std::vector<DrawableData> new_drawables(static_cast<std::size_t>(frames_in_flight) * new_slots_per_frame);

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

    auto ImGuiRenderer::render_draw_data(VkCommandBuffer cmd, ImDrawData *dd, ShaderObjectSet const &pipeline,
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
        vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vkCmdSetViewportWithCount(cmd, 1, &vp);

        // Shader objects bake no blend state, so set it here.
        VkBool32 const blend_enable = VK_TRUE;
        VkColorBlendEquationEXT const blend_equation{
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .alphaBlendOp = VK_BLEND_OP_ADD,
        };
        VkColorComponentFlags const write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blend_enable);
        vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blend_equation);
        vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &write_mask);

        const float L = dd->DisplayPos.x;
        const float R = dd->DisplayPos.x + dd->DisplaySize.x;
        const float T = dd->DisplayPos.y;
        const float B = dd->DisplayPos.y + dd->DisplaySize.y;
        const ImVec2 clip_offset = dd->DisplayPos;
        const ImVec2 clip_scale = dd->FramebufferScale;

        DrawableData &drawable = acquire_draw_slot();

        if (std::cmp_less(drawable.index_count, dd->TotalIdxCount)) {
            auto const size = static_cast<std::size_t>(dd->TotalIdxCount * 4) * sizeof(ImDrawIdx);
            auto const actual_size = next_power_of_two(size);
            info("[ImGui] Reallocating index buffer to {}", human_readable_bytes(actual_size));

            auto created = Buffer::create(renderer.context(), BufferCreateInfo{
                                                                      .size = actual_size,
                                                                      .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                                      .memory = BufferMemory::upload,
                                                                      .debug_name = "imgui_index_buffer",
                                                              });

            if (!created) {
                error("[ImGui] Failed to allocate index buffer");
                return;
            }

            drawable.index = std::make_unique<Buffer>(std::move(*created));
            drawable.index_count = static_cast<std::uint32_t>(actual_size / sizeof(ImDrawIdx));
        }

        if (static_cast<std::int32_t>(drawable.vertex_count) < dd->TotalVtxCount) {
            auto const size = static_cast<std::size_t>(dd->TotalVtxCount * 4) * sizeof(ImDrawVert);
            auto const actual_size = next_power_of_two(size);
            info("[ImGui] Reallocating vertex buffer to {}", human_readable_bytes(actual_size));

            auto created =
                    Buffer::create(renderer.context(), BufferCreateInfo{
                                                               .size = actual_size,
                                                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                               .memory = BufferMemory::upload,
                                                               .debug_name = "imgui_vertex_buffer",
                                                       });

            if (!created) {
                error("[ImGui] Failed to allocate vertex buffer");
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
                auto const *imgui_cmd = dd->CmdLists[n];
                all_vtx.insert(all_vtx.end(), imgui_cmd->VtxBuffer.Data,
                               imgui_cmd->VtxBuffer.Data + imgui_cmd->VtxBuffer.Size);
                all_itx.insert(all_itx.end(), imgui_cmd->IdxBuffer.Data,
                               imgui_cmd->IdxBuffer.Data + imgui_cmd->IdxBuffer.Size);
            }

            if (!drawable.vertex->write(0, std::span<const ImDrawVert>(all_vtx))) {
                error("[ImGui] Failed to write vertex buffer");
            }
            if (!drawable.index->write(0, std::span<const ImDrawIdx>(all_itx))) {
                error("[ImGui] Failed to write index buffer");
            }
        }

        vkCmdBindIndexBuffer(cmd, drawable.index->buffer, 0, VK_INDEX_TYPE_UINT16);
        pipeline.bind(cmd);

        renderer.resource_table().bind(cmd, frame_index, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout());

        std::uint32_t index_offset = 0;
        std::uint32_t vertex_offset = 0;

        for (int n = 0; n < dd->CmdListsCount; n++) {
            auto const *command_list = dd->CmdLists[n];

            for (int cmd_i = 0; cmd_i < command_list->CmdBuffer.Size; cmd_i++) {
                auto const &imgui_cmd = command_list->CmdBuffer[cmd_i];

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

                // Strip gui::linear_source_texture_bit to get the bindless index.
                auto const raw_tex_id = static_cast<std::uint64_t>(imgui_cmd.GetTexID());
                bool const already_linear = (raw_tex_id & linear_source_texture_bit) != 0;

                PC pc{
                        .lrtb = {L, R, T, B},
                        .vb = drawable.vertex->device_address,
                        .base_vertex = vertex_offset + imgui_cmd.VtxOffset,
                        .texture_id = static_cast<std::uint32_t>(raw_tex_id & 0xFFFFFFFFULL),
                        .sampler_id = sampler.index,
                        .already_linear = already_linear ? 1U : 0U,
                };

                vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);

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
                vkCmdSetScissorWithCount(cmd, 1, &scissor);

                vkCmdDrawIndexed(cmd, imgui_cmd.ElemCount, 1, index_offset + imgui_cmd.IdxOffset, 0, 0);
            }

            index_offset += static_cast<std::uint32_t>(command_list->IdxBuffer.Size);
            vertex_offset += static_cast<std::uint32_t>(command_list->VtxBuffer.Size);
        }
    }

    auto ImGuiRenderer::set_app_name(const std::string_view name) -> void {
        config_name = std::format("{}.ini", name);
        config_path = std::make_unique<std::filesystem::path>(config_name);
    }

    auto ImGuiRenderer::update_font(FontChoice const &f) -> void {
        ImGuiIO &io = ImGui::GetIO();
        ImFontConfig cfg{};
        cfg.FontDataOwnedByAtlas = false;
        cfg.RasterizerMultiply = 1.5F;
        cfg.SizePixels = std::ceil(f.size);
        cfg.PixelSnapH = true;
        cfg.OversampleH = 4;
        cfg.OversampleV = 4;
        cfg.FontLoaderFlags = ImGuiFreeTypeLoaderFlags_ForceAutoHint | ImGuiFreeTypeLoaderFlags_LightHinting;

        ImFont *font = nullptr;
        std::filesystem::path const font_path{f.font_path};

        if (std::filesystem::exists(font_path)) {
            auto const path_str = font_path.string();
            font = io.Fonts->AddFontFromFileTTF(path_str.c_str(), cfg.SizePixels, &cfg);

            // Font Awesome glyphs for the toast icons, merged into the UI font. This backend builds a fixed atlas (no
            // ImGuiBackendFlags_RendererHasTextures), so the glyph ranges must be given up front or they render as '?'.
            static constexpr std::array<ImWchar, 3> icon_ranges{ICON_MIN_FA, ICON_MAX_16_FA, 0};
            ImFontConfig icon_cfg{};
            icon_cfg.MergeMode = true;
            icon_cfg.PixelSnapH = true;
            icon_cfg.GlyphMinAdvanceX = cfg.SizePixels * 2.0F / 3.0F;
            static_cast<void>(io.Fonts->AddFontFromMemoryCompressedTTF(
                    fa_solid_900_compressed_data, static_cast<int>(fa_solid_900_compressed_size),
                    cfg.SizePixels * 2.0F / 3.0F, &icon_cfg, icon_ranges.data()));
        }

        io.Fonts->Flags |= ImFontAtlasFlags_NoPowerOfTwoHeight;

        unsigned char *pixels;
        int width;
        int height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

        auto const *as_bytes = std::bit_cast<std::byte const *>(pixels);

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
            error("[ImGui] Failed to create font atlas image: {}", describe(created_image.error()));
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
            error("[ImGui] Failed to create font sampler");
            return;
        }

        sampler = *created_sampler;
    }

} // namespace gui
